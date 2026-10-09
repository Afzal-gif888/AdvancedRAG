"""Fixed-window rate limiting.

- memory: per-process counters (development / single instance)
- redis:  shared counters (required outside development; see config.check_safety)

Failure policy: if the limiter backend errors, development fails open (logged) while
staging/production fail CLOSED with 503 — a broken limiter must not silently disable
brute-force protection.
"""
import asyncio
import logging
import math
import time
from typing import Callable, Dict, Optional, Tuple

from fastapi import Request

from app.core.config import settings
from app.core.exceptions import RateLimited, ServiceUnavailable

logger = logging.getLogger("advancerag.ratelimit")


class MemoryBackend:
    def __init__(self):
        self._counters: Dict[str, Tuple[int, float]] = {}
        self._lock = asyncio.Lock()

    async def hit(self, key: str, window: int) -> Tuple[int, float]:
        now = time.monotonic()
        async with self._lock:
            count, reset_at = self._counters.get(key, (0, now + window))
            if now >= reset_at:
                count, reset_at = 0, now + window
            count += 1
            self._counters[key] = (count, reset_at)
            if len(self._counters) > 100_000:
                self._counters = {k: v for k, v in self._counters.items() if v[1] > now}
            return count, reset_at - now


class RedisBackend:
    def __init__(self, url: str):
        import redis.asyncio as redis

        self._redis = redis.from_url(url, socket_connect_timeout=2, socket_timeout=2)

    async def hit(self, key: str, window: int) -> Tuple[int, float]:
        k = f"rl:{key}"
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.incr(k)
            pipe.expire(k, window, nx=True)
            pipe.ttl(k)
            count, _, ttl = await pipe.execute()
        return int(count), float(max(ttl, 1))

    async def ping(self) -> bool:
        try:
            return bool(await self._redis.ping())
        except Exception:
            return False

    async def aclose(self) -> None:
        await self._redis.aclose()


_backend = None


def get_backend():
    global _backend
    if _backend is None:
        if settings.RATE_LIMIT_BACKEND == "redis":
            if not settings.REDIS_URL:
                raise RuntimeError("RATE_LIMIT_BACKEND=redis requires REDIS_URL")
            _backend = RedisBackend(settings.REDIS_URL.get_secret_value())
        else:
            _backend = MemoryBackend()
    return _backend


def set_backend(backend) -> None:
    global _backend
    _backend = backend


def client_ip(request: Request) -> str:
    """The real client address. Each trusted proxy appends the address it received the request
    from, so with N trusted hops the client is the N-th entry from the right; anything further
    left was written by the client and can be forged."""
    hops = settings.proxy_hops
    if hops:
        chain = [p.strip() for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()]
        if len(chain) >= hops:
            return chain[-hops][:64]
    return request.client.host if request.client else "unknown"


def _fail_closed() -> bool:
    return settings.ENVIRONMENT in ("staging", "production")


async def check(key: str, times: int, seconds: int) -> None:
    """Count one hit against `key`; raise 429 when over `times` per `seconds`."""
    if not settings.RATE_LIMIT_ENABLED:
        return
    try:
        count, retry_after = await get_backend().hit(key, seconds)
    except Exception:
        logger.exception("rate_limiter_unavailable", extra={"fail_closed": _fail_closed()})
        if _fail_closed():
            raise ServiceUnavailable("Service temporarily unavailable. Please try again shortly.", code="RATE_LIMITER_UNAVAILABLE")
        return
    if count > times:
        raise RateLimited(headers={"Retry-After": str(math.ceil(retry_after))})


class RateLimit:
    """Per-IP dependency: `Depends(RateLimit("login", times=10, seconds=300))`."""

    def __init__(self, scope: str, times: int, seconds: int, key_func: Optional[Callable[[Request], str]] = None):
        self.scope, self.times, self.seconds, self.key_func = scope, times, seconds, key_func

    async def __call__(self, request: Request) -> None:
        discriminator = self.key_func(request) if self.key_func else client_ip(request)
        await check(f"{self.scope}:ip:{discriminator}", self.times, self.seconds)


async def hit_user_limit(scope: str, user_id, times: int, seconds: int) -> None:
    """Per-account limit for authenticated endpoints."""
    await check(f"{scope}:user:{user_id}", times, seconds)


async def hit_identity_limit(scope: str, identity: str, times: int, seconds: int) -> None:
    """Per-target-account limit for unauthenticated endpoints keyed by an identifier (e.g. the
    normalised email). Stops credential stuffing spread across many IPs against one account."""
    await check(f"{scope}:id:{identity}", times, seconds)
