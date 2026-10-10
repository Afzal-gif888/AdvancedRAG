"""advancerag_common.wake: waking sleeping dependencies (Render free) through a relay outside the host."""
import asyncio

import httpx
import pytest

from advancerag_common import wake


@pytest.fixture(autouse=True)
def _fresh_cache():
    wake._last_ok.clear()
    yield
    wake._last_ok.clear()


def _mock(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(wake.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


def _no_sleep(monkeypatch):
    real_sleep = asyncio.sleep
    sleeps = []

    async def fast(seconds):
        sleeps.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(wake.asyncio, "sleep", fast)
    return sleeps


async def test_pings_health_of_each_dependency_and_never_raises(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if "down" in request.url.host:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json={"status": "ok"})

    _mock(monkeypatch, handler)
    tasks = wake.wake_dependencies(["https://agent.example.com/", "https://down.example.com", ""])
    await asyncio.gather(*tasks)  # an unreachable dependency is only logged
    assert sorted(seen) == ["https://agent.example.com/health", "https://down.example.com/health"]


async def test_startup_pings_go_through_the_relay_when_configured(monkeypatch):
    seen = []
    _mock(monkeypatch, lambda r: seen.append(str(r.url)) or httpx.Response(200))
    await asyncio.gather(*wake.wake_dependencies(["https://advancerag-llm.onrender.com"], relay="https://web.example.com/wake/"))
    assert seen == ["https://web.example.com/wake/advancerag-llm"]


def test_relay_path_is_the_first_host_label():
    assert wake.relay_for("https://advancerag-rag-core.onrender.com/", "https://x.vercel.app/wake") == \
        "https://x.vercel.app/wake/advancerag-rag-core"
    assert wake.relay_for("https://advancerag-rag-core.onrender.com", None) is None


async def test_sleeping_dependency_is_woken_through_the_relay_then_polled_until_healthy(monkeypatch):
    """Render answers in-platform calls to a sleeping service with an immediate 502 and does not wake it;
    only the relay (outside Render) wakes it. The caller keeps probing /health with backoff."""
    state = {"awake": False, "probes": 0, "relayed": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "web.example.com":
            state["relayed"] += 1
            state["awake"] = True
            return httpx.Response(200)
        state["probes"] += 1
        return httpx.Response(200 if state["awake"] and state["probes"] >= 3 else 502)

    _mock(monkeypatch, handler)
    sleeps = _no_sleep(monkeypatch)
    ok = await wake.ensure_awake("https://advancerag-agent-core.onrender.com", budget=120, relay="https://web.example.com/wake")
    assert ok and state["relayed"] == 1 and state["probes"] == 3
    assert sleeps == [1.0, 2.0]  # exponential backoff between probes


async def test_gives_up_after_the_budget(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(wake.time, "monotonic", lambda: clock["t"])
    real_sleep = asyncio.sleep

    async def advance(seconds):
        clock["t"] += seconds
        await real_sleep(0)

    monkeypatch.setattr(wake.asyncio, "sleep", advance)
    probes = []
    _mock(monkeypatch, lambda r: probes.append(r.url.host) or httpx.Response(502))
    ok = await wake.ensure_awake("https://advancerag-llm.onrender.com", budget=30, relay=None)
    assert ok is False
    assert 3 <= len(probes) <= 8  # bounded: backoff 1, 2, 4, 8, 15 ... within 30 s


async def test_recently_healthy_dependency_costs_no_request(monkeypatch):
    calls = []
    _mock(monkeypatch, lambda r: calls.append(1) or httpx.Response(200))
    assert await wake.ensure_awake("https://a.example.com", budget=10)
    assert await wake.ensure_awake("https://a.example.com", budget=10)
    assert len(calls) == 1
    wake.mark_unhealthy("https://a.example.com")
    assert await wake.ensure_awake("https://a.example.com", budget=10)
    assert len(calls) == 2


async def test_zero_budget_means_off(monkeypatch):
    _mock(monkeypatch, lambda r: pytest.fail("no request expected"))
    assert await wake.ensure_awake("https://a.example.com", budget=0)


def test_backend_and_agent_core_keep_it_off_by_default():
    from app.core.config import Settings

    fields = Settings.model_fields
    assert fields["WAKE_DEPENDENCIES_ON_START"].default is False
    assert fields["CHAT_WAKE_SECONDS"].default == 0 and fields["RAG_WAKE_SECONDS"].default == 0
    assert fields["WAKE_RELAY_URL"].default is None and fields["KEEP_AWAKE_URL"].default is None
