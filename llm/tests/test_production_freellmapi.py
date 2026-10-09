"""Production FreeLLMAPI contract: configuration validation, authentication, model readiness,
bounded retries, safe errors and secret hygiene. Runs against the local mock gateway — this
proves behaviour, not production connectivity."""
import json
import logging

import httpx
import pytest

from llm_service.config import LOCAL_DEVELOPMENT_URL, ConfigError, Settings
from llm_service.providers.base import GenerationRequest, LLMError
from mock_openai import VALID_KEY, free_port, scenario

MSGS = [{"role": "user", "content": "hi"}]
KEY = "freellmapi-" + "k" * 30


def _prod(**overrides):
    base = {
        "ENVIRONMENT": "production",
        "FREELLMAPI_BASE_URL": "https://llm.example.com/v1",
        "FREELLMAPI_API_KEY": KEY,
        "INTERNAL_SERVICE_TOKEN": "t" * 40,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)


def _rejected(settings: Settings) -> str:
    with pytest.raises(ConfigError) as exc:
        settings.check_safety()
    message = str(exc.value)
    assert message.startswith("LLM configuration invalid:")
    assert KEY not in message
    return message


# ------------------------------------------------------------------ configuration

@pytest.mark.parametrize("environment", ["production", "staging"])
def test_https_remote_endpoint_accepted(environment):
    s = _prod(ENVIRONMENT=environment, FREELLMAPI_BASE_URL="https://freellmapi.example.com/v1/").check_safety()
    assert s.freellmapi_base_url == "https://freellmapi.example.com/v1"


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:31415/v1",
    "https://127.0.0.1/v1",
    "https://127.0.0.2/v1",
    "http://localhost:31415/v1",
    "https://LOCALHOST/v1",
    "https://0.0.0.0/v1",
    "http://[::1]:31415/v1",
    "http://host.docker.internal:31415/v1",
    "https://api.localhost/v1",
])
def test_production_rejects_local_endpoints(url):
    assert "HTTPS non-local endpoint" in _rejected(_prod(FREELLMAPI_BASE_URL=url))


def test_production_rejects_plain_http():
    assert "plain http is refused" in _rejected(_prod(FREELLMAPI_BASE_URL="http://freellmapi.example.com/v1"))


@pytest.mark.parametrize("value", [None, "", "   "])
def test_production_requires_url_and_never_falls_back_to_localhost(value):
    s = _prod(FREELLMAPI_BASE_URL=value)
    assert "FREELLMAPI_BASE_URL is required" in _rejected(s)
    with pytest.raises(ConfigError):
        s.freellmapi_base_url  # noqa: B018 — must not silently return the desktop URL


def test_production_requires_key():
    assert "FREELLMAPI_API_KEY is required" in _rejected(_prod(FREELLMAPI_API_KEY=""))


@pytest.mark.parametrize("url", ["ftp://llm.example.com/v1", "llm.example.com/v1", "https:///v1"])
def test_malformed_url_rejected_in_every_environment(url):
    _rejected(_prod(FREELLMAPI_BASE_URL=url))
    _rejected(_prod(ENVIRONMENT="development", FREELLMAPI_BASE_URL=url))


def test_development_allows_desktop_app_and_defaults_to_it():
    dev = Settings(ENVIRONMENT="development", FREELLMAPI_BASE_URL="http://127.0.0.1:31415/v1", _env_file=None).check_safety()
    assert dev.freellmapi_base_url == "http://127.0.0.1:31415/v1"
    default = Settings(ENVIRONMENT="development", _env_file=None).check_safety()
    assert default.freellmapi_base_url == LOCAL_DEVELOPMENT_URL == "http://127.0.0.1:31415/v1"


def test_url_and_key_load_from_environment(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("FREELLMAPI_BASE_URL", "https://gw.example.com/v1")
    monkeypatch.setenv("FREELLMAPI_API_KEY", KEY)
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "t" * 40)
    s = Settings(_env_file=None).check_safety()
    assert s.freellmapi_base_url == "https://gw.example.com/v1" and s.FREELLMAPI_API_KEY.get_secret_value() == KEY
    assert KEY not in repr(s) and KEY not in str(s.model_dump())


def test_railway_private_http_is_opt_in_and_narrow():
    private = "http://freellmapi.railway.internal:3001/v1"
    _rejected(_prod(FREELLMAPI_BASE_URL=private))  # off by default
    assert _prod(FREELLMAPI_BASE_URL=private, FREELLMAPI_ALLOW_RAILWAY_PRIVATE_HTTP=True).check_safety()
    # The flag never allows other plain-http hosts or localhost.
    _rejected(_prod(FREELLMAPI_BASE_URL="http://freellmapi.example.com/v1", FREELLMAPI_ALLOW_RAILWAY_PRIVATE_HTTP=True))
    _rejected(_prod(FREELLMAPI_BASE_URL="http://127.0.0.1:31415/v1", FREELLMAPI_ALLOW_RAILWAY_PRIVATE_HTTP=True))


def test_invalid_model_name_rejected():
    with pytest.raises(ValueError):
        _prod(FREELLMAPI_MODEL="auto; rm -rf")


# ------------------------------------------------------------------ authentication & secrets

async def test_bearer_key_and_request_id_sent(make_provider):
    from advancerag_common.observability import request_id_var

    token = request_id_var.set("req-abc12345")
    try:
        await make_provider().generate(GenerationRequest(messages=MSGS))
    finally:
        request_id_var.reset(token)
    assert scenario.last_auth == f"Bearer {VALID_KEY}"
    assert scenario.last_request_id == "req-abc12345"
    assert VALID_KEY not in json.dumps(scenario.last_body)


async def test_key_never_logged_even_at_debug(make_provider, caplog):
    caplog.set_level(logging.DEBUG)  # includes httpx/httpcore debug output
    scenario.fail_times, scenario.status = 1, 503
    await make_provider().generate(GenerationRequest(messages=MSGS))
    with pytest.raises(LLMError):
        await make_provider(api_key="freellmapi-wrong-key-123").generate(GenerationRequest(messages=MSGS))
    await make_provider().health_check()
    assert VALID_KEY not in caplog.text and "freellmapi-wrong-key-123" not in caplog.text


def test_log_formatter_redacts_authorization_headers():
    from advancerag_common.observability import JsonFormatter

    record = logging.LogRecord("x", logging.INFO, "", 0, "headers: Authorization: Bearer %s", (KEY,), None)
    record.authorization = f"Bearer {KEY}"
    out = JsonFormatter("llm").format(record)
    assert KEY not in out and "[REDACTED]" in out


async def test_errors_never_contain_key_or_url(make_provider, mock_server):
    for status in (401, 429, 500, 503):
        scenario.reset()
        scenario.fail_times, scenario.status = 10, status
        with pytest.raises(LLMError) as exc:
            await make_provider(max_retries=0).generate(GenerationRequest(messages=MSGS))
        text = f"{exc.value.code} {exc.value.message} {exc.value!r}"
        assert VALID_KEY not in text and mock_server not in text and "10.0.0.5" not in text
    assert VALID_KEY not in repr(make_provider())


# ------------------------------------------------------------------ models / readiness

async def test_auto_model_available(make_provider):
    status = await make_provider(default_model="auto").health_check()
    assert status.ok and status.valid_response and status.model_available


@pytest.mark.parametrize("models,model,expected", [
    (["auto", "groq/llama-3.3-70b"], "groq/llama-3.3-70b", True),   # explicit listed id
    (["auto"], "auto:fast", True),                                   # routing alias, gateway routes
    (["groq/llama-3.3-70b"], "auto:fast", False),                    # no router behind the alias
    (["groq/llama-3.3-70b"], "auto", False),                         # gateway does not support auto
    (["auto"], "openrouter/missing-model", False),                   # typo / removed model
])
async def test_configured_model_must_be_offered(make_provider, models, model, expected):
    scenario.models = models
    status = await make_provider(default_model=model).health_check()
    assert status.model_available is expected and status.ok is expected
    if not expected:
        assert "FREELLMAPI_MODEL" in status.detail


async def test_models_response_must_be_openai_compatible(make_provider):
    scenario.mode = "models_invalid"
    status = await make_provider().health_check()
    assert status.reachable and status.authenticated and status.valid_response is False and not status.ok


async def test_models_endpoint_error(make_provider):
    scenario.mode, scenario.status = "models_error", 500
    status = await make_provider().health_check()
    assert status.reachable and not status.ok and "HTTP 500" in status.detail


async def test_models_auth_failure(make_provider):
    status = await make_provider(api_key="wrong").health_check()
    assert status.reachable and not status.authenticated and not status.ok


async def test_readiness_never_runs_a_completion(make_provider):
    await make_provider().health_check()
    assert scenario.calls == 0  # /models only; the chat check is the explicit --chat diagnostic


async def _ready(provider):
    from llm_service import main
    from llm_service.providers import get_provider

    main.app.dependency_overrides[get_provider] = lambda: provider
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://llm") as c:
            return await c.get("/ready")
    finally:
        main.app.dependency_overrides.clear()
        await provider.aclose()


async def test_ready_endpoint_success_and_no_secrets(make_provider, mock_server):
    r = await _ready(make_provider())
    assert r.status_code == 200 and r.json()["status"] == "ready"
    assert VALID_KEY not in r.text and mock_server not in r.text and "Bearer" not in r.text


async def test_ready_endpoint_gateway_unavailable(make_provider):
    r = await _ready(make_provider(base_url=f"http://127.0.0.1:{free_port()}/v1"))
    assert r.status_code == 503 and r.json()["checks"]["reachable"] is False


async def test_ready_endpoint_invalid_credentials(make_provider):
    r = await _ready(make_provider(api_key="freellmapi-revoked"))
    assert r.status_code == 503 and r.json()["checks"]["authenticated"] is False
    assert "freellmapi-revoked" not in r.text


async def test_ready_endpoint_invalid_model(make_provider):
    r = await _ready(make_provider(default_model="no-such-model"))
    assert r.status_code == 503 and r.json()["checks"]["model_available"] is False


# ------------------------------------------------------------------ chat errors & retries

@pytest.mark.parametrize("status,code", [(500, "LLM_PROVIDER_UNAVAILABLE"), (502, "LLM_PROVIDER_UNAVAILABLE"),
                                         (503, "LLM_PROVIDER_UNAVAILABLE"), (504, "LLM_PROVIDER_UNAVAILABLE"),
                                         (429, "LLM_RATE_LIMITED")])
async def test_transient_statuses_retry_with_bound(make_provider, status, code):
    scenario.fail_times, scenario.status, scenario.retry_after = 10, status, 0
    with pytest.raises(LLMError) as exc:
        await make_provider(max_retries=2).generate(GenerationRequest(messages=MSGS))
    assert exc.value.code == code and exc.value.retryable
    assert scenario.calls == 3  # 1 + max_retries, never more


@pytest.mark.parametrize("status,code", [(400, "LLM_BAD_REQUEST"), (401, "LLM_AUTH_FAILED"), (403, "LLM_AUTH_FAILED"),
                                         (404, "LLM_BAD_REQUEST"), (422, "LLM_BAD_REQUEST")])
async def test_client_errors_are_not_retried(make_provider, status, code):
    scenario.mode, scenario.status = "error_status", status
    with pytest.raises(LLMError) as exc:
        await make_provider(max_retries=3).generate(GenerationRequest(messages=MSGS))
    assert exc.value.code == code and not exc.value.retryable and scenario.calls == 1


async def test_transient_failure_recovers_in_stream(make_provider):
    scenario.fail_times, scenario.status = 1, 500
    pieces = [p async for p in make_provider().stream(GenerationRequest(messages=MSGS))]
    assert "".join(p.text for p in pieces if p.text) == "Hello from the mock model." and scenario.calls == 2


async def test_stream_timeout(make_provider):
    scenario.delay = 1.0
    with pytest.raises(LLMError) as exc:
        [p async for p in make_provider(timeout=0.3).stream(GenerationRequest(messages=MSGS))]
    assert exc.value.code == "LLM_TIMEOUT" and scenario.calls == 1


async def test_connection_failure_retries_bounded(make_provider):
    with pytest.raises(LLMError) as exc:
        await make_provider(base_url=f"http://127.0.0.1:{free_port()}/v1", max_retries=2).generate(GenerationRequest(messages=MSGS))
    assert exc.value.code == "LLM_UNAVAILABLE" and exc.value.retryable


def test_backoff_is_exponential_and_capped(make_provider):
    p = make_provider(backoff=1.0, max_backoff=8.0, max_retry_after=10.0)
    assert [p.retry_delay(a) for a in range(6)] == [1.0, 2.0, 4.0, 8.0, 8.0, 8.0]
    assert p.retry_delay(0, httpx.Response(429, headers={"retry-after": "3600"})) == 10.0
    assert p.retry_delay(0, httpx.Response(429, headers={"retry-after": "2"})) == 2.0
    assert p.retry_delay(1, httpx.Response(429, headers={"retry-after": "soon"})) == 2.0


async def test_retry_and_status_metrics(make_provider):
    from advancerag_common.observability import METRICS

    scenario.fail_times, scenario.status = 1, 503
    await make_provider().generate(GenerationRequest(messages=MSGS))
    rendered = METRICS.render()
    assert 'llm_retries_total{reason="http_503"}' in rendered
    assert 'llm_upstream_responses_total{status_class="5xx"}' in rendered
    assert 'llm_upstream_responses_total{status_class="2xx"}' in rendered
    assert VALID_KEY not in rendered


async def test_stream_success_metric_records_model(api):
    from advancerag_common.observability import METRICS

    await api.post("/v1/stream", json={"messages": MSGS})
    assert 'llm_call_ms_count{kind="stream",model="mock-provider/auto",outcome="ok"}' in METRICS.render()
