"""Wake sleeping dependencies when a service starts (opt-in: WAKE_DEPENDENCIES_ON_START=true).

On hosts that put idle services to sleep (Render's free instances spin down after 15 minutes and take
about a minute, RAG Core several minutes, to come back), a request chain backend -> agent-core ->
rag-core/llm would otherwise wake them one after another. When this service starts, it sends one
GET /health to each dependency in the background, so they start in parallel. Never blocks startup,
never retries, and only touches the unauthenticated liveness endpoint.
"""
import asyncio
import logging
from typing import Iterable, List

import httpx

log = logging.getLogger("advancerag.wake")


async def _ping(base_url: str, timeout: float) -> None:
    url = base_url.rstrip("/") + "/health"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url)
        log.info("dependency_wake", extra={"target": base_url, "status": r.status_code})
    except (httpx.HTTPError, asyncio.CancelledError) as exc:
        log.info("dependency_wake", extra={"target": base_url, "status": type(exc).__name__})


def wake_dependencies(base_urls: Iterable[str], timeout: float = 300.0) -> List[asyncio.Task]:
    """Start background pings; keep the returned tasks referenced and cancel them on shutdown."""
    return [asyncio.create_task(_ping(u, timeout)) for u in base_urls if u]
