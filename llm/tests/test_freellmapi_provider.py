import json
import logging

import pytest

from llm_service.providers.base import GenerationRequest, LLMError
from mock_openai import scenario

MSGS = [{"role": "user", "content": "hi"}]


async def test_completion(make_provider):
    result = await make_provider().generate(GenerationRequest(messages=MSGS, temperature=0.2))
    assert result.content == "Hello from the mock model."
    assert result.model == "mock-provider/auto" and result.finish_reason == "stop"
    assert result.usage.total_tokens == 11
    assert scenario.last_body["model"] == "auto" and scenario.last_body["temperature"] == 0.2
    assert scenario.last_body["stream"] is False


@pytest.mark.parametrize("model", ["auto:fast", "auto:smart", "some-provider/model-x"])
async def test_model_is_configurable_per_call_and_default(make_provider, model):
    await make_provider(default_model="auto:smart").generate(GenerationRequest(messages=MSGS))
    assert scenario.last_body["model"] == "auto:smart"
    await make_provider().generate(GenerationRequest(messages=MSGS, model=model))
    assert scenario.last_body["model"] == model


async def test_streaming_yields_tokens_then_final(make_provider):
    pieces = [p async for p in make_provider().stream(GenerationRequest(messages=MSGS))]
    text = "".join(p.text for p in pieces if p.text)
    assert text == "Hello from the mock model."
    assert sum(1 for p in pieces if p.text) == 5  # multiple chunks
    final = pieces[-1].final
    assert final.finish_reason == "stop" and final.model == "mock-provider/auto"


async def test_invalid_key_is_auth_failed_and_not_retried(make_provider, caplog):
    caplog.set_level(logging.DEBUG)
    with pytest.raises(LLMError) as exc:
        await make_provider(api_key="wrong-key").generate(GenerationRequest(messages=MSGS))
    assert exc.value.code == "LLM_AUTH_FAILED" and not exc.value.retryable
    assert scenario.calls == 1
    # Neither our key nor the provider's error text leaks into the error or logs.
    assert "wrong-key" not in exc.value.message and "sk-should-not-leak" not in caplog.text and "wrong-key" not in caplog.text


async def test_missing_key_is_not_configured(make_provider):
    with pytest.raises(LLMError) as exc:
        await make_provider(api_key=None).generate(GenerationRequest(messages=MSGS))
    assert exc.value.code == "LLM_NOT_CONFIGURED"
    assert scenario.calls == 0


async def test_retries_transient_5xx_then_succeeds(make_provider):
    scenario.fail_times, scenario.status = 2, 503
    result = await make_provider(max_retries=2).generate(GenerationRequest(messages=MSGS))
    assert result.content and scenario.calls == 3


async def test_rate_limit_honours_retry_after_then_gives_up(make_provider):
    scenario.fail_times, scenario.status, scenario.retry_after = 10, 429, 0
    with pytest.raises(LLMError) as exc:
        await make_provider(max_retries=2).generate(GenerationRequest(messages=MSGS))
    assert exc.value.code == "LLM_RATE_LIMITED" and exc.value.retryable
    assert scenario.calls == 3  # bounded retries


async def test_bad_model_is_not_retried(make_provider):
    scenario.mode, scenario.status = "error_status", 404
    with pytest.raises(LLMError) as exc:
        await make_provider().generate(GenerationRequest(messages=MSGS))
    assert exc.value.code == "LLM_BAD_REQUEST" and scenario.calls == 1


async def test_timeout(make_provider):
    scenario.delay = 1.0
    with pytest.raises(LLMError) as exc:
        await make_provider(timeout=0.3).generate(GenerationRequest(messages=MSGS))
    assert exc.value.code == "LLM_TIMEOUT"
    assert scenario.calls == 1  # read timeouts are not retried


async def test_provider_unreachable(make_provider):
    from mock_openai import free_port

    with pytest.raises(LLMError) as exc:
        await make_provider(base_url=f"http://127.0.0.1:{free_port()}/v1", max_retries=1).generate(GenerationRequest(messages=MSGS))
    assert exc.value.code == "LLM_UNAVAILABLE" and exc.value.retryable


async def test_malformed_response(make_provider):
    scenario.mode = "malformed"
    with pytest.raises(LLMError) as exc:
        await make_provider().generate(GenerationRequest(messages=MSGS))
    assert exc.value.code == "LLM_INVALID_RESPONSE"
    with pytest.raises(LLMError) as exc:
        [p async for p in make_provider().stream(GenerationRequest(messages=MSGS))]
    assert exc.value.code == "LLM_INVALID_RESPONSE"


async def test_stream_retries_only_before_first_token(make_provider):
    scenario.fail_times, scenario.status = 1, 502
    pieces = [p async for p in make_provider().stream(GenerationRequest(messages=MSGS))]
    assert "".join(p.text for p in pieces if p.text) == "Hello from the mock model."
    assert scenario.calls == 2


async def test_stream_interrupted_mid_way(make_provider):
    scenario.mode = "stream_break"
    received = []
    with pytest.raises(LLMError) as exc:
        async for p in make_provider().stream(GenerationRequest(messages=MSGS)):
            if p.text:
                received.append(p.text)
    assert received  # partial tokens were delivered before the failure
    assert exc.value.code in ("LLM_STREAM_INTERRUPTED", "LLM_INVALID_RESPONSE")
    assert scenario.calls == 1  # never retried after tokens were sent


async def test_optional_tools_and_structured_output_pass_through(make_provider):
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    await make_provider().generate(GenerationRequest(messages=MSGS, tools=tools, tool_choice="auto", response_format={"type": "json_object"}))
    assert scenario.last_body["tools"] == tools and scenario.last_body["response_format"] == {"type": "json_object"}


async def test_health_check(make_provider):
    status = await make_provider().health_check()
    assert status.ok and status.models_available == 3
    deep = await make_provider().health_check(deep=True)
    assert deep.ok and deep.completion_ok
    bad = await make_provider(api_key="nope").health_check()
    assert bad.reachable and not bad.authenticated and not bad.ok
    none = await make_provider(api_key=None).health_check()
    assert not none.configured and none.reachable and not none.ok  # server up, key missing


def test_request_body_never_contains_key(make_provider):
    body = make_provider()._body(GenerationRequest(messages=MSGS), stream=True)
    assert "test-freellmapi-key" not in json.dumps(body)
