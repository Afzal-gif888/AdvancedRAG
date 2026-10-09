import json

import pytest

from mock_openai import scenario


def _events(text):
    out = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


async def test_generate_endpoint(api):
    r = await api.post("/v1/generate", json={"messages": [{"role": "user", "content": "hi"}], "model": "auto:fast"})
    assert r.status_code == 200
    body = r.json()
    assert body["content"] == "Hello from the mock model." and body["usage"]["total_tokens"] == 11 and body["latency_ms"] >= 0
    assert scenario.last_body["model"] == "auto:fast"


async def test_stream_endpoint_translates_to_internal_events(api):
    r = await api.post("/v1/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    events = _events(r.text)
    assert [e for e, _ in events][-1] == "done"
    assert "".join(d["delta"] for e, d in events if e == "token") == "Hello from the mock model."
    # Provider wire format (choices/delta) never leaks through.
    assert "choices" not in r.text


async def test_errors_are_safe(api):
    scenario.fail_times, scenario.status = 5, 503
    r = await api.post("/v1/generate", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 503
    err = r.json()["error"]
    assert err["code"] == "LLM_PROVIDER_UNAVAILABLE" and err["retryable"] is True
    assert "10.0.0.5" not in r.text
    scenario.reset()
    scenario.fail_times, scenario.status = 5, 503
    r = await api.post("/v1/stream", json={"messages": [{"role": "user", "content": "hi"}]})
    assert _events(r.text)[-1][0] == "error"


async def test_validation(api):
    assert (await api.post("/v1/generate", json={"messages": []})).status_code == 422
    assert (await api.post("/v1/generate", json={"messages": [{"role": "user", "content": "x"}], "model": "bad model; drop"})).status_code == 422


async def test_ready_and_health(api):
    assert (await api.get("/health")).json() == {"status": "ok"}
    r = await api.get("/ready")
    assert r.status_code == 200 and r.json()["checks"] == {"configured": True, "reachable": True, "authenticated": True, "models_response_valid": True, "model_available": True}
    assert "127.0.0.1" not in r.text


async def test_internal_token_enforced(mock_server, monkeypatch):
    import httpx

    from llm_service import main
    from advancerag_common.service import internal_auth

    secured = main.FastAPI()
    secured.add_api_route("/x", lambda: {"ok": True}, dependencies=[main.Depends(internal_auth("s" * 40))])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=secured), base_url="http://t") as c:
        assert (await c.get("/x")).status_code == 401
        assert (await c.get("/x", headers={"Authorization": "Bearer wrong"})).status_code == 401
        assert (await c.get("/x", headers={"Authorization": "Bearer " + "s" * 40})).status_code == 200


@pytest.mark.parametrize(
    "overrides,fragment",
    [
        ({"FREELLMAPI_BASE_URL": "http://localhost:31415/v1"}, "localhost"),
        ({"FREELLMAPI_BASE_URL": "http://llm.example.com/v1"}, "HTTPS"),
        ({"FREELLMAPI_API_KEY": ""}, "FREELLMAPI_API_KEY"),
        ({"INTERNAL_SERVICE_TOKEN": "short"}, "INTERNAL_SERVICE_TOKEN"),
    ],
)
def test_production_rejects_unsafe_config(overrides, fragment):
    from llm_service.config import Settings

    base = {
        "ENVIRONMENT": "production",
        "FREELLMAPI_BASE_URL": "https://llm.example.com/v1",
        "FREELLMAPI_API_KEY": "k" * 20,
        "INTERNAL_SERVICE_TOKEN": "t" * 40,
        "_env_file": None,
    }
    base.update(overrides)
    with pytest.raises(ValueError) as exc:
        Settings(**base).check_safety()
    assert fragment in str(exc.value)
    assert "k" * 20 not in str(exc.value)


def test_production_accepts_secure_remote_endpoint():
    from llm_service.config import Settings

    s = Settings(ENVIRONMENT="production", FREELLMAPI_BASE_URL="https://llm.example.com/v1/", FREELLMAPI_API_KEY="k" * 20, INTERNAL_SERVICE_TOKEN="t" * 40, _env_file=None).check_safety()
    assert s.FREELLMAPI_BASE_URL == "https://llm.example.com/v1"
    assert "k" * 20 not in repr(s)  # SecretStr


def test_environment_is_required(monkeypatch):
    from llm_service.config import Settings

    monkeypatch.delenv("ENVIRONMENT", raising=False)
    with pytest.raises(ValueError):
        Settings(_env_file=None)
