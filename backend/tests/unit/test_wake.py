"""advancerag_common.wake: opt-in wake-up of sleeping dependencies (Render free) on service start."""
import asyncio

import httpx

from advancerag_common import wake


async def test_pings_health_of_each_dependency_and_never_raises(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if "down" in request.url.host:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"status": "ok"})

    real = httpx.AsyncClient
    monkeypatch.setattr(wake.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    tasks = wake.wake_dependencies(["https://agent.example.com/", "https://down.example.com", ""])
    await asyncio.gather(*tasks)  # an unreachable dependency is only logged
    assert sorted(seen) == ["https://agent.example.com/health", "https://down.example.com/health"]


def test_backend_and_agent_core_keep_it_off_by_default():
    from app.core.config import Settings

    assert Settings.model_fields["WAKE_DEPENDENCIES_ON_START"].default is False
