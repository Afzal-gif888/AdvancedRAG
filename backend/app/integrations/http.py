"""Shared outbound HTTP plumbing: pooled clients, correlation-ID propagation, service auth."""
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

import httpx
from advancerag_common.observability import METRICS

from app.core.config import settings
from app.core.context import get_request_id

logger = logging.getLogger("advancerag.integrations")

_clients: Dict[str, httpx.AsyncClient] = {}


def get_client(name: str, base_url: str, timeout: float) -> httpx.AsyncClient:
    client = _clients.get(name)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        _clients[name] = client
    return client


async def close_clients() -> None:
    for client in list(_clients.values()):
        await client.aclose()
    _clients.clear()


def outbound_headers() -> Dict[str, str]:
    headers = {}
    request_id = get_request_id()
    if request_id:
        headers["X-Request-ID"] = request_id
    if settings.internal_token:
        headers["Authorization"] = f"Bearer {settings.internal_token}"
    return headers


@asynccontextmanager
async def timed(service: str, operation: str, extra: Optional[dict] = None):
    """Log latency and outcome of an outbound call (no payloads)."""
    started = time.perf_counter()
    outcome = "ok"
    try:
        yield
    except Exception as exc:
        outcome = type(exc).__name__
        raise
    finally:
        METRICS.observe_ms(f"{service}_call", round((time.perf_counter() - started) * 1000, 2), operation=operation, outcome=outcome)
        logger.info(
            "outbound_call",
            extra={
                "service": service,
                "operation": operation,
                "outcome": outcome,
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                **(extra or {}),
            },
        )


async def ping(name: str, base_url: str, path: str = "/health") -> bool:
    """True only for HTTP 200 (a 401 means our service token is wrong: not ready).

    Uses its own pooled client: the pool for real traffic must keep its own (long) timeouts no
    matter whether a readiness probe or a chat request happened to create it first."""
    try:
        response = await get_client(f"{name}:health", base_url, 10.0).get(path, headers=outbound_headers(), timeout=10.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False
