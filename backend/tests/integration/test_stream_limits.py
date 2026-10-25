"""Concurrent-stream cap (DoS protection), slot release on every exit path, and stream failure behaviour."""
import asyncio
import os
import uuid

import pytest

from app.core.config import settings
from app.core.exceptions import RateLimited
from app.integrations.agent_core import StreamEvent, get_agent_client
from app.main import app
from app.services import stream_limiter
from conftest import FakeAgent, create_conversation, parse_sse, register_and_login

API = "/api/v1"


@pytest.fixture(autouse=True)
def clean_limiter():
    stream_limiter.reset()
    yield
    stream_limiter.reset()


class BlockingAgent(FakeAgent):
    """Holds every stream open until `gate` is set (a slow model)."""

    def __init__(self):
        super().__init__()
        self.gate = asyncio.Event()
        self.started = 0

    async def stream(self, req):
        self.started += 1
        yield StreamEvent("agent.started", {})
        await self.gate.wait()
        yield StreamEvent("content.delta", {"delta": "done"})
        yield StreamEvent("generation.completed", {"citations": [], "metadata": {}})


@pytest.fixture
def blocking_agent():
    agent = BlockingAgent()
    app.dependency_overrides[get_agent_client] = lambda: agent
    yield agent
    app.dependency_overrides.pop(get_agent_client, None)


async def _post(client, auth, conv, text="hi"):
    return await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": text}, headers=auth)


async def test_fourth_concurrent_stream_is_refused_and_leaves_no_messages(client, auth, blocking_agent, monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONCURRENT_STREAMS_PER_USER", 3)
    convs = [await create_conversation(client, auth) for _ in range(4)]
    tasks = [asyncio.create_task(_post(client, auth, c, f"q{i}")) for i, c in enumerate(convs[:3])]
    while blocking_agent.started < 3:
        await asyncio.sleep(0.01)
    refused = await _post(client, auth, convs[3], "one too many")
    assert refused.status_code == 429 and refused.json()["error"]["code"] == "TOO_MANY_STREAMS"
    assert int(refused.headers["retry-after"]) > 0
    msgs = (await client.get(f"{API}/conversations/{convs[3]['id']}/messages", headers=auth)).json()
    assert msgs["total"] == 0, "a refused turn must not persist the user message"
    blocking_agent.gate.set()
    done = await asyncio.gather(*tasks)
    assert [r.status_code for r in done] == [200, 200, 200]
    me = (await client.get(f"{API}/auth/me", headers=auth)).json()["id"]
    assert await stream_limiter.active(uuid.UUID(me)) == 0


async def test_slot_is_released_after_a_normal_stream_and_after_a_failed_one(client, auth, agent, monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONCURRENT_STREAMS_PER_USER", 1)
    conv = await create_conversation(client, auth)
    assert (await _post(client, auth, conv)).status_code == 200
    agent.mode = "error_event"
    r = await _post(client, auth, conv)
    assert r.status_code == 200 and parse_sse(r.text)[-1]["event"] == "error"
    agent.mode = "ok"
    assert (await _post(client, auth, conv)).status_code == 200, "the slot was released after the failed stream"


async def test_slot_is_released_when_the_turn_cannot_be_prepared(client, auth, agent, monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONCURRENT_STREAMS_PER_USER", 1)
    missing = {"id": str(uuid.uuid4())}
    assert (await _post(client, auth, missing)).status_code == 404
    conv = await create_conversation(client, auth)
    assert (await _post(client, auth, conv)).status_code == 200


async def test_limit_is_per_user(make_client, blocking_agent, monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONCURRENT_STREAMS_PER_USER", 1)
    a, b = make_client(), make_client()
    ha, hb = await register_and_login(a), await register_and_login(b)
    ca, cb = await create_conversation(a, ha), await create_conversation(b, hb)
    ta = asyncio.create_task(_post(a, ha, ca))
    while blocking_agent.started < 1:
        await asyncio.sleep(0.01)
    tb = asyncio.create_task(_post(b, hb, cb))                      # another user is not affected by A's open stream
    while blocking_agent.started < 2:
        await asyncio.sleep(0.01)
    assert (await _post(a, ha, await create_conversation(a, ha))).status_code == 429
    blocking_agent.gate.set()
    assert [r.status_code for r in await asyncio.gather(ta, tb)] == [200, 200]


async def test_a_zero_limit_disables_the_cap(client, auth, blocking_agent, monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONCURRENT_STREAMS_PER_USER", 0)
    tasks = [asyncio.create_task(_post(client, auth, await create_conversation(client, auth))) for _ in range(5)]
    while blocking_agent.started < 5:
        await asyncio.sleep(0.01)
    blocking_agent.gate.set()
    assert all(r.status_code == 200 for r in await asyncio.gather(*tasks))


async def test_a_leaked_slot_expires(monkeypatch):
    monkeypatch.setattr(settings, "MAX_CONCURRENT_STREAMS_PER_USER", 1)
    monkeypatch.setattr(settings, "STALE_GENERATION_SECONDS", 0)       # slots live for 0 seconds in this test
    user = uuid.uuid4()
    await stream_limiter.acquire(user)                                  # never released (a vanished client)
    await asyncio.sleep(0.01)
    await stream_limiter.acquire(user)                                  # the stale slot no longer blocks the user


# ---------------------------------------------------------------- shared (Redis) cap across replicas

REDIS_URL = os.environ.get("TEST_REDIS_URL")
needs_redis = pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")


@pytest.fixture
def shared(monkeypatch):
    from pydantic import SecretStr

    monkeypatch.setattr(settings, "RATE_LIMIT_BACKEND", "redis")
    monkeypatch.setattr(settings, "REDIS_URL", SecretStr(REDIS_URL or "redis://127.0.0.1:1"))
    monkeypatch.setattr(settings, "MAX_CONCURRENT_STREAMS_PER_USER", 2)
    stream_limiter.reset()
    yield
    stream_limiter.reset()


@needs_redis
async def test_the_cap_is_shared_by_every_replica(shared):
    user = uuid.uuid4()
    a = await stream_limiter.acquire(user)                             # replica 1
    stream_limiter.reset()                                              # a second process has its own memory and connection
    b = await stream_limiter.acquire(user)                             # replica 2
    stream_limiter.reset()
    with pytest.raises(RateLimited):
        await stream_limiter.acquire(user)                              # a third stream, on either replica, is refused
    assert await stream_limiter.active(user) == 2
    stream_limiter.release(user, a)
    await asyncio.sleep(0.2)                                            # the release is a background task
    assert await stream_limiter.active(user) == 1
    await stream_limiter.acquire(user)                                  # the freed slot is usable again
    stream_limiter.release(user, b)


@needs_redis
async def test_shared_slots_expire_on_the_redis_clock(shared, monkeypatch):
    monkeypatch.setattr(settings, "STALE_GENERATION_SECONDS", 0.05)
    monkeypatch.setattr(settings, "MAX_CONCURRENT_STREAMS_PER_USER", 1)
    user = uuid.uuid4()
    await stream_limiter.acquire(user)
    with pytest.raises(RateLimited):
        await stream_limiter.acquire(user)
    await asyncio.sleep(0.2)
    await stream_limiter.acquire(user)


@needs_redis
async def test_many_simultaneous_acquires_never_exceed_the_cap(shared):
    user = uuid.uuid4()
    results = await asyncio.gather(*[stream_limiter.acquire(user) for _ in range(12)], return_exceptions=True)
    assert sum(not isinstance(r, Exception) for r in results) == 2, "the Lua script makes check-and-reserve atomic"


async def test_an_unreachable_redis_falls_back_to_the_process_cap(shared, caplog, monkeypatch):
    from pydantic import SecretStr

    monkeypatch.setattr(settings, "REDIS_URL", SecretStr("redis://127.0.0.1:1"))   # nothing listens here, whatever TEST_REDIS_URL says
    stream_limiter.reset()
    user = uuid.uuid4()
    await stream_limiter.acquire(user)
    await stream_limiter.acquire(user)
    with pytest.raises(RateLimited):
        await stream_limiter.acquire(user)                              # still capped, per process
    assert any("process_cap" in r.message for r in caplog.records)


# ---------------------------------------------------------------- the same Lua script on an in-process Redis (always runs)

@pytest.fixture
def replicas(monkeypatch):
    """Two 'replicas' = two clients of one in-memory Redis server (fakeredis with the real Lua engine)."""
    fakeredis = pytest.importorskip("fakeredis")
    pytest.importorskip("lupa")
    from pydantic import SecretStr

    server = fakeredis.FakeServer()
    clients = [fakeredis.FakeAsyncRedis(server=server), fakeredis.FakeAsyncRedis(server=server)]
    current = {"i": 0}
    monkeypatch.setattr(settings, "RATE_LIMIT_BACKEND", "redis")
    monkeypatch.setattr(settings, "REDIS_URL", SecretStr("redis://fake"))
    monkeypatch.setattr(settings, "MAX_CONCURRENT_STREAMS_PER_USER", 2)
    monkeypatch.setattr(stream_limiter, "_client", lambda: clients[current["i"]])
    return current


async def test_in_process_redis_cap_is_shared_by_two_replicas(replicas):
    user = uuid.uuid4()
    replicas["i"] = 0
    a = await stream_limiter.acquire(user)
    replicas["i"] = 1
    b = await stream_limiter.acquire(user)
    with pytest.raises(RateLimited):
        await stream_limiter.acquire(user)                              # replica 2 sees replica 1's slot
    replicas["i"] = 0
    with pytest.raises(RateLimited):
        await stream_limiter.acquire(user)
    assert await stream_limiter.active(user) == 2
    stream_limiter.release(user, a)
    await asyncio.sleep(0.1)
    assert await stream_limiter.active(user) == 1
    await stream_limiter.acquire(user)
    stream_limiter.release(user, b)


async def test_in_process_redis_check_and_reserve_is_atomic_under_concurrency(replicas):
    user = uuid.uuid4()
    results = await asyncio.gather(*[stream_limiter.acquire(user) for _ in range(20)], return_exceptions=True)
    assert sum(not isinstance(r, Exception) for r in results) == 2
    assert all(isinstance(r, RateLimited) for r in results if isinstance(r, Exception))


async def test_in_process_redis_other_users_are_independent_and_zero_means_unlimited(replicas, monkeypatch):
    a, b = uuid.uuid4(), uuid.uuid4()
    await stream_limiter.acquire(a)
    await stream_limiter.acquire(a)
    await stream_limiter.acquire(b)                                     # b is unaffected by a's slots
    monkeypatch.setattr(settings, "MAX_CONCURRENT_STREAMS_PER_USER", 0)
    for _ in range(5):
        await stream_limiter.acquire(a)


async def test_in_process_redis_slots_expire(replicas, monkeypatch):
    monkeypatch.setattr(settings, "STALE_GENERATION_SECONDS", 0.05)
    monkeypatch.setattr(settings, "MAX_CONCURRENT_STREAMS_PER_USER", 1)
    user = uuid.uuid4()
    await stream_limiter.acquire(user)
    with pytest.raises(RateLimited):
        await stream_limiter.acquire(user)
    await asyncio.sleep(0.2)
    await stream_limiter.acquire(user)
