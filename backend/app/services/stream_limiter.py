"""Per-user cap on concurrent chat streams.

Each open stream holds a connection and can trigger several LLM calls (rewrite, grade, generate, verify), so
one account opening many streams could exhaust the shared model quota and the instance.

Two backends, chosen by RATE_LIMIT_BACKEND (the same switch as the request rate limiter):

- memory: per process. A deployment with N replicas allows N times the cap. Fine for one instance and development.
- redis:  one sorted set per user shared by every replica, updated by an atomic Lua script that uses the Redis server's
  clock, so the cap is exact across processes. If Redis cannot be reached the cap falls back to the per-process counter
  (and is logged): losing the shared limit must not take chat down, and the request rate limiter, which uses the same
  Redis, already fails closed outside development.

Every slot also expires after `STALE_GENERATION_SECONDS`, so a stream that never releases its slot (a client that vanished
before the generator started, a crashed replica) cannot lock a user out for good. `release` is deliberately best-effort and
synchronous (it runs in a generator's `finally`, possibly during cancellation); a lost release only lasts until expiry.
"""
import asyncio
import logging
import time
import uuid
from collections import defaultdict
from typing import DefaultDict, List, Union

from app.core.config import settings
from app.core.exceptions import RateLimited

logger = logging.getLogger("advancerag.streams")

_slots: DefaultDict[uuid.UUID, List[float]] = defaultdict(list)
_tasks: set = set()

# KEYS[1] = zset of slot tokens scored by expiry (ms, Redis clock). ARGV: limit, token, ttl_ms.
_ACQUIRE = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
local limit = tonumber(ARGV[1])
if limit > 0 and redis.call('ZCARD', KEYS[1]) >= limit then return 0 end
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[3]), ARGV[2])
redis.call('PEXPIRE', KEYS[1], ARGV[3])
return 1
"""

_redis = None


def _client():
    global _redis
    if _redis is None:
        import redis.asyncio as redis

        _redis = redis.from_url(settings.REDIS_URL.get_secret_value(), socket_connect_timeout=2, socket_timeout=2)
    return _redis


def _shared() -> bool:
    return settings.RATE_LIMIT_BACKEND == "redis" and settings.REDIS_URL is not None


def _key(user_id: uuid.UUID) -> str:
    return f"streams:{user_id}"


def _too_many(limit: int) -> RateLimited:
    return RateLimited(
        f"You already have {limit} answers being generated. Wait for one to finish, then try again.",
        code="TOO_MANY_STREAMS",
        headers={"Retry-After": "5"},
    )


def _prune(user_id: uuid.UUID, now: float) -> List[float]:
    live = [t for t in _slots[user_id] if t > now]
    _slots[user_id] = live
    return live


def _acquire_local(user_id: uuid.UUID, limit: int) -> float:
    now = time.monotonic()
    live = _prune(user_id, now)
    if limit and len(live) >= limit:
        raise _too_many(limit)
    token = now + settings.STALE_GENERATION_SECONDS
    _slots[user_id].append(token)
    return token


async def acquire(user_id: uuid.UUID) -> Union[float, str]:
    """Reserve a slot or raise 429. Returns the slot's token for `release`."""
    limit = settings.MAX_CONCURRENT_STREAMS_PER_USER
    if _shared():
        token = uuid.uuid4().hex
        try:
            ok = await _client().eval(_ACQUIRE, 1, _key(user_id), limit, token, int(settings.STALE_GENERATION_SECONDS * 1000))
        except Exception:
            logger.warning("stream_limit_shared_backend_unavailable_using_process_cap", exc_info=True)
        else:
            if not ok:
                raise _too_many(limit)
            return token
    return _acquire_local(user_id, limit)


def release(user_id: uuid.UUID, token: Union[float, str]) -> None:
    if isinstance(token, str):
        async def _release() -> None:
            try:
                await _client().zrem(_key(user_id), token)
            except Exception:
                logger.warning("stream_slot_release_failed_it_will_expire", exc_info=True)

        try:
            task = asyncio.get_running_loop().create_task(_release())
        except RuntimeError:
            return
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
        return
    try:
        _slots[user_id].remove(token)
    except ValueError:
        pass
    if not _slots[user_id]:
        _slots.pop(user_id, None)


async def active(user_id: uuid.UUID) -> int:
    if _shared():
        try:
            r = _client()
            now_ms = int((await r.time())[0]) * 1000
            return int(await r.zcount(_key(user_id), now_ms, "+inf"))
        except Exception:
            pass
    return len(_prune(user_id, time.monotonic()))


def reset() -> None:
    """Forget every slot held by this process (tests)."""
    global _redis
    _slots.clear()
    _redis = None
