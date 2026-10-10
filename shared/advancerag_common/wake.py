"""Wake sleeping dependencies on hosts that put idle services to sleep (Render's free instances
spin down after 15 minutes without traffic).

Render only wakes a sleeping service for requests that come from outside Render: a call from one
Render service to another's public URL gets an immediate 502 and the target stays asleep. So a wake
goes through a relay outside Render (WAKE_RELAY_URL, e.g. the Vercel frontend's `/wake/<name>`
rewrites to `https://<name>.onrender.com/health`), and the caller then polls the target's own
GET /health until it answers. Only the unauthenticated liveness endpoint is ever touched.

- wake_dependencies(): fire-and-forget pings when a service starts, so dependencies wake in parallel.
- ensure_awake():      wait (bounded, exponential backoff) until a dependency answers /health.
"""
import asyncio
import logging
import time
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

import httpx

log = logging.getLogger("advancerag.wake")

# base_url -> monotonic time of the last healthy answer, so a warm dependency costs nothing.
_last_ok: Dict[str, float] = {}


def relay_for(base_url: str, relay: Optional[str]) -> Optional[str]:
    """`https://advancerag-llm.onrender.com` + relay `https://x.vercel.app/wake` -> `.../wake/advancerag-llm`."""
    host = urlparse(base_url).hostname
    if not relay or not host:
        return None
    return f"{relay.rstrip('/')}/{host.split('.')[0]}"


async def _get(url: str, timeout: float) -> Optional[int]:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=min(timeout, 10.0))) as client:
            return (await client.get(url)).status_code
    except httpx.HTTPError:
        return None


async def _ping(base_url: str, timeout: float, relay: Optional[str]) -> None:
    via = relay_for(base_url, relay)
    status = await _get(via or base_url.rstrip("/") + "/health", timeout)
    log.info("dependency_wake", extra={"target": base_url, "via_relay": bool(via), "status": status})


def wake_dependencies(base_urls: Iterable[str], timeout: float = 300.0, relay: Optional[str] = None) -> List[asyncio.Task]:
    """Start background pings; keep the returned tasks referenced and cancel them on shutdown."""
    return [asyncio.create_task(_ping(u, timeout, relay)) for u in base_urls if u]


async def ensure_awake(
    base_url: str,
    *,
    budget: float,
    relay: Optional[str] = None,
    fresh_for: float = 60.0,
    probe_timeout: float = 10.0,
    max_backoff: float = 15.0,
) -> bool:
    """True once `base_url` answers GET /health with 200, False when `budget` seconds run out.

    Answers at once (no request) if the dependency was healthy within `fresh_for` seconds. Otherwise
    probes it directly; if it does not answer, asks the relay to wake it (once, again after 60 s of
    no progress) and keeps probing with exponential backoff (1, 2, 4 ... max_backoff seconds)."""
    if budget <= 0:
        return True
    now = time.monotonic()
    if now - _last_ok.get(base_url, float("-inf")) < fresh_for:
        return True
    deadline, delay, relayed_at = now + budget, 1.0, None
    health = base_url.rstrip("/") + "/health"
    via = relay_for(base_url, relay)
    relay_task: Optional[asyncio.Task] = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            status = await _get(health, max(1.0, min(probe_timeout, remaining)))
            if status == 200:
                _last_ok[base_url] = time.monotonic()
                if relayed_at is not None:
                    log.info("dependency_awake", extra={"target": base_url, "waited_s": round(budget - remaining, 1)})
                return True
            if via and (relayed_at is None or time.monotonic() - relayed_at > 60):
                # The relay request may itself wait for the cold start; the probes decide when it is up.
                relayed_at = time.monotonic()
                relay_task = asyncio.create_task(_get(via, 90.0))
                log.info("dependency_waking", extra={"target": base_url, "probe_status": status})
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.warning("dependency_wake_timeout", extra={"target": base_url, "budget_s": budget, "probe_status": status})
                return False
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, max_backoff)
    finally:
        if relay_task is not None and not relay_task.done():
            relay_task.cancel()


def mark_unhealthy(base_url: str) -> None:
    """Forget a cached healthy answer (e.g. after a call to the dependency failed)."""
    _last_ok.pop(base_url, None)
