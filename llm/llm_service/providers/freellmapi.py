"""FreeLLMAPI adapter (OpenAI-compatible `/chat/completions`, `/models`).

The only module that knows FreeLLMAPI's wire format. Retries happen here (the layer that
understands provider semantics), not in Agent Core or the backend.

Works the same against the desktop app (development) and a hosted FreeLLMAPI behind HTTPS
(production): only the base URL and the unified key differ. Provider keys (OpenRouter, Groq,
...) live inside FreeLLMAPI; this service only ever holds the unified FreeLLMAPI key.
"""
import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, Optional

import httpx
from advancerag_common.observability import METRICS, get_request_id

from llm_service.providers.base import (
    LLM_AUTH_FAILED,
    LLM_BAD_REQUEST,
    LLM_INVALID_RESPONSE,
    LLM_NOT_CONFIGURED,
    LLM_PROVIDER_ERROR,
    LLM_PROVIDER_UNAVAILABLE,
    LLM_RATE_LIMITED,
    LLM_STREAM_INTERRUPTED,
    LLM_TIMEOUT,
    LLM_UNAVAILABLE,
    GenerationRequest,
    GenerationResult,
    HealthStatus,
    LLMError,
    LLMProvider,
    StreamDelta,
    Usage,
)

log = logging.getLogger("advancerag.llm.freellmapi")

# Transient: retried with bounded exponential backoff. Everything else (400/401/403/404/422...)
# is a configuration or request problem that a retry cannot fix.
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _status_class(status: int) -> str:
    return f"{status // 100}xx"


def is_routing_alias(model: str) -> bool:
    """FreeLLMAPI routing profiles (`auto:fast`, `auto:<profile>`) are accepted by the router but
    not listed by /models; they are valid whenever the gateway offers `auto` itself."""
    return model.startswith("auto:")


class FreeLLMAPIProvider(LLMProvider):
    name = "freellmapi"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str],
        default_model: str = "auto",
        timeout: float = 90.0,
        connect_timeout: float = 5.0,
        max_retries: int = 2,
        backoff: float = 1.0,
        max_backoff: float = 8.0,
        max_retry_after: float = 10.0,
        stream_usage: bool = False,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.default_model = default_model
        self.max_retries, self.backoff, self.max_backoff, self.max_retry_after = max_retries, backoff, max_backoff, max_retry_after
        self.stream_usage = stream_usage
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            transport=transport,
        )

    def __repr__(self) -> str:  # never include the key
        return f"FreeLLMAPIProvider(default_model={self.default_model!r})"

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ helpers
    def _headers(self) -> Dict[str, str]:
        if not self._api_key:
            raise LLMError(LLM_NOT_CONFIGURED, "The LLM provider is not configured.", status_code=503)
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        # Correlation id only (validated upstream to [A-Za-z0-9._-]{8,64}); never a secret.
        request_id = get_request_id()
        if request_id:
            headers["X-Request-ID"] = request_id
        return headers

    def _body(self, req: GenerationRequest, stream: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": req.model or self.default_model, "messages": req.messages, "stream": stream}
        for key in ("temperature", "max_tokens", "response_format", "tools", "tool_choice"):
            value = getattr(req, key)
            if value is not None:
                body[key] = value
        if stream and self.stream_usage:
            body["stream_options"] = {"include_usage": True}
        return body

    def _status_error(self, response: httpx.Response) -> LLMError:
        status = response.status_code
        # Log only a short provider error type — never the body, headers or request.
        provider_type = None
        try:
            err = response.json().get("error")
            provider_type = (err.get("type") or err.get("code")) if isinstance(err, dict) else None
        except Exception:
            pass
        log.warning("provider_error_status", extra={"status": status, "provider_error_type": str(provider_type)[:60] if provider_type else None})
        if status in (401, 403):
            return LLMError(LLM_AUTH_FAILED, "The LLM provider rejected the configured credentials.", status_code=502)
        if status == 429:
            return LLMError(LLM_RATE_LIMITED, "The LLM provider is rate limiting requests.", retryable=True, status_code=503)
        if status in (400, 404, 413, 422):
            return LLMError(LLM_BAD_REQUEST, "The LLM provider rejected the request (check the model name and parameters).", status_code=502)
        if status >= 500:
            return LLMError(LLM_PROVIDER_UNAVAILABLE, "The LLM provider is temporarily unavailable.", retryable=True, status_code=503)
        return LLMError(LLM_PROVIDER_ERROR, "The LLM provider returned an unexpected response.", status_code=502)

    def retry_delay(self, attempt: int, response: Optional[httpx.Response] = None) -> float:
        """Exponential backoff capped at `max_backoff`; a provider Retry-After is honoured up to
        `max_retry_after`. Both caps keep a burst of failures from turning into a retry storm."""
        if response is not None and response.headers.get("retry-after"):
            try:
                return max(0.0, min(float(response.headers["retry-after"]), self.max_retry_after))
            except ValueError:
                pass
        return min(self.backoff * (2 ** attempt), self.max_backoff)

    async def _send(self, req: GenerationRequest, *, stream: bool) -> httpx.Response:
        """POST with bounded retries for connection failures and retryable statuses. For streams,
        the returned response is open (caller must close) — retries only happen before any token."""
        headers = self._headers()
        body = self._body(req, stream)
        last_error: Optional[LLMError] = None
        for attempt in range(self.max_retries + 1):
            response = None
            reason = None
            try:
                request = self._client.build_request("POST", "/chat/completions", json=body, headers=headers)
                response = await self._client.send(request, stream=stream)
            except httpx.TimeoutException as exc:
                # A read timeout means the model is slow, not down: retrying would double latency.
                if isinstance(exc, httpx.ConnectTimeout):
                    last_error = LLMError(LLM_UNAVAILABLE, "The LLM provider could not be reached.", retryable=True, status_code=503)
                    reason = "connect_timeout"
                else:
                    METRICS.inc("llm_timeouts", kind="stream" if stream else "generate")
                    raise LLMError(LLM_TIMEOUT, "The LLM provider timed out.", retryable=True, status_code=504) from exc
            except httpx.HTTPError as exc:
                last_error = LLMError(LLM_UNAVAILABLE, "The LLM provider could not be reached.", retryable=True, status_code=503)
                reason = "connection_error"
                log.warning("provider_connection_error", extra={"error_type": type(exc).__name__, "attempt": attempt + 1})
            else:
                METRICS.inc("llm_upstream_responses", status_class=_status_class(response.status_code))
                if response.is_success:
                    return response
                if stream:
                    await response.aread()
                    await response.aclose()
                last_error = self._status_error(response)
                if response.status_code not in _RETRYABLE_STATUS:
                    raise last_error
                reason = f"http_{response.status_code}"
            if attempt < self.max_retries:
                METRICS.inc("llm_retries", reason=reason or "unknown")
                await asyncio.sleep(self.retry_delay(attempt, response))
        assert last_error is not None
        raise last_error

    @staticmethod
    def _usage(raw: Any) -> Usage:
        if not isinstance(raw, dict):
            return Usage()
        return Usage(raw.get("prompt_tokens"), raw.get("completion_tokens"), raw.get("total_tokens"))

    # ------------------------------------------------------------------ API
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        response = await self._send(request, stream=False)
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice.get("message") or {}
            content = message.get("content")
            tool_calls = message.get("tool_calls")
            if content is None and not tool_calls:
                raise ValueError("empty choice")
            return GenerationResult(
                content=content or "",
                model=data.get("model"),
                finish_reason=choice.get("finish_reason"),
                usage=self._usage(data.get("usage")),
                tool_calls=tool_calls,
            )
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError(LLM_INVALID_RESPONSE, "The LLM provider returned a malformed response.", status_code=502) from exc

    async def stream(self, request: GenerationRequest) -> AsyncIterator[StreamDelta]:
        response = await self._send(request, stream=True)
        model: Optional[str] = None
        finish_reason: Optional[str] = None
        usage = Usage()
        received_any = False
        try:
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise LLMError(LLM_INVALID_RESPONSE, "The LLM provider sent a malformed stream.", status_code=502) from exc
                if not isinstance(chunk, dict):
                    raise LLMError(LLM_INVALID_RESPONSE, "The LLM provider sent a malformed stream.", status_code=502)
                if chunk.get("error"):
                    raise LLMError(LLM_PROVIDER_ERROR, "The LLM provider reported an error during generation.", status_code=502)
                model = chunk.get("model") or model
                if chunk.get("usage"):
                    usage = self._usage(chunk["usage"])
                for choice in chunk.get("choices") or []:
                    if not isinstance(choice, dict):
                        continue
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        received_any = True
                        yield StreamDelta(text=delta)
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        except httpx.TimeoutException as exc:
            METRICS.inc("llm_timeouts", kind="stream")
            raise LLMError(LLM_TIMEOUT, "The LLM provider timed out.", retryable=True, status_code=504) from exc
        except httpx.HTTPError as exc:
            code = LLM_STREAM_INTERRUPTED if received_any else LLM_UNAVAILABLE
            raise LLMError(code, "The connection to the LLM provider was interrupted.", retryable=True, status_code=503) from exc
        finally:
            await response.aclose()
        yield StreamDelta(final=GenerationResult(content="", model=model, finish_reason=finish_reason, usage=usage))

    async def health_check(self, *, deep: bool = False) -> HealthStatus:
        """Readiness: reachable, key accepted, /models is OpenAI-compatible JSON and offers the
        configured model. `deep=True` additionally runs one tiny completion (diagnostics only —
        never on every probe). Details never include URLs or secrets."""
        status = HealthStatus(configured=bool(self._api_key))
        if not self._api_key:
            # Still report reachability so "server down" and "key missing" are distinguishable.
            try:
                await self._client.get("/models", timeout=5.0)
                status.reachable = True
                status.detail = "FREELLMAPI_API_KEY is not set"
            except httpx.HTTPError:
                status.detail = "FREELLMAPI_API_KEY is not set and FreeLLMAPI is not reachable"
            return status
        try:
            response = await self._client.get("/models", headers=self._headers(), timeout=5.0)
            status.reachable = True
        except httpx.HTTPError:
            status.detail = "FreeLLMAPI is not reachable"
            return status
        if response.status_code in (401, 403):
            status.detail = "FreeLLMAPI rejected the API key"
            return status
        if response.is_success:
            status.authenticated = True
            ids = self._model_ids(response)
            if ids is None:
                status.valid_response = False
                status.detail = "FreeLLMAPI /models did not return OpenAI-compatible JSON"
                return status
            status.valid_response = True
            status.models_available = len(ids)
            model = self.default_model
            status.model_available = model in ids or (is_routing_alias(model) and "auto" in ids)
            if not status.model_available:
                status.detail = f"Configured model {model!r} is not offered by FreeLLMAPI (check FREELLMAPI_MODEL)"
                return status
        elif response.status_code == 404:
            # Gateway without a models listing: authentication is proven by a real call below.
            status.authenticated = deep
        else:
            status.detail = f"Models endpoint returned HTTP {response.status_code}"
            return status
        if deep:
            started = time.perf_counter()
            try:
                result = await self.generate(GenerationRequest(messages=[{"role": "user", "content": "Reply with the single word: OK"}], max_tokens=5))
                status.completion_ok = bool(result.content.strip())
                status.authenticated = True
                status.detail = f"completion in {round((time.perf_counter() - started) * 1000)} ms via model {result.model or 'unknown'}"
            except LLMError as exc:
                status.completion_ok = False
                status.authenticated = exc.code != LLM_AUTH_FAILED and status.authenticated
                status.detail = exc.code
        return status

    @staticmethod
    def _model_ids(response: httpx.Response) -> Optional[set]:
        """Model ids from an OpenAI-style `{"data": [{"id": ...}, ...]}`; None if malformed."""
        try:
            data = response.json().get("data")
        except (ValueError, AttributeError):
            return None
        if not isinstance(data, list) or not all(isinstance(m, dict) and isinstance(m.get("id"), str) for m in data):
            return None
        return {m["id"] for m in data}
