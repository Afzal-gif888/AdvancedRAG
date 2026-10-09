"""Integration: the real LLM service against scripts/mock_freellmapi.py (the same mock gateway
the e2e stack uses) over real HTTP — /v1/models, chat completions and streaming.

Deterministic CI coverage of the OpenAI-compatible contract. It does NOT prove connectivity to
a production FreeLLMAPI; see docs/production-freellmapi.md for that verification.
"""
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

from llm_service.providers.base import GenerationRequest, LLMError
from llm_service.providers.freellmapi import FreeLLMAPIProvider
from mock_openai import ServerThread, free_port

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "mock_freellmapi.py"
KEY = "mock-gateway-key-0123456789"


@pytest.fixture(scope="module")
def gateway():
    spec = importlib.util.spec_from_file_location("mock_freellmapi", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    port = free_port()
    with ServerThread(module.build_app(KEY), port):
        yield f"http://127.0.0.1:{port}/v1"


def _provider(url, key=KEY, model="auto"):
    return FreeLLMAPIProvider(base_url=url, api_key=key, default_model=model, timeout=10, max_retries=0)


async def test_models_and_readiness(gateway):
    p = _provider(gateway)
    status = await p.health_check()
    await p.aclose()
    assert status.ok and status.valid_response and status.model_available and status.models_available == 3


async def test_chat_completion_and_streaming(gateway):
    p = _provider(gateway)
    result = await p.generate(GenerationRequest(messages=[{"role": "user", "content": "ping"}]))
    pieces = [x async for x in p.stream(GenerationRequest(messages=[{"role": "user", "content": "ping"}]))]
    await p.aclose()
    assert "ping" in result.content and result.model == "mock/auto"
    streamed = "".join(x.text for x in pieces if x.text)
    assert streamed == result.content and sum(1 for x in pieces if x.text) > 1
    assert pieces[-1].final.finish_reason == "stop" and pieces[-1].final.usage.total_tokens


async def test_wrong_key_rejected(gateway):
    p = _provider(gateway, key="wrong")
    with pytest.raises(LLMError) as exc:
        await p.generate(GenerationRequest(messages=[{"role": "user", "content": "x"}]))
    status = await p.health_check()
    await p.aclose()
    assert exc.value.code == "LLM_AUTH_FAILED" and not status.authenticated


async def test_llm_service_stream_endpoint_end_to_end(gateway):
    """LLM service HTTP API -> provider -> mock gateway: internal SSE contract only."""
    from llm_service import main
    from llm_service.providers import get_provider

    p = _provider(gateway)
    main.app.dependency_overrides[get_provider] = lambda: p
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://llm") as c:
            r = await c.post("/v1/stream", json={"messages": [{"role": "user", "content": "hello there"}]})
            ready = await c.get("/ready")
    finally:
        main.app.dependency_overrides.clear()
        await p.aclose()
    events = [(b.splitlines()[0][7:], json.loads(b.splitlines()[1][6:])) for b in r.text.strip().split("\n\n")]
    assert {e for e, _ in events} == {"token", "done"} and events[-1][0] == "done"
    assert "hello there" in "".join(d["delta"] for e, d in events if e == "token")
    assert "choices" not in r.text and KEY not in r.text  # gateway wire format and key never leak
    assert ready.status_code == 200 and KEY not in ready.text
