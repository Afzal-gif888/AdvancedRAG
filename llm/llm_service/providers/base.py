"""Provider-neutral LLM interface. Everything above this layer depends on these types only."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional


class LLMError(Exception):
    """Safe, provider-neutral error. `code` and `message` may be shown to callers;
    raw provider responses never are."""

    def __init__(self, code: str, message: str, *, retryable: bool = False, status_code: int = 502):
        super().__init__(message)
        self.code, self.message, self.retryable, self.status_code = code, message, retryable, status_code


# Canonical error codes
LLM_NOT_CONFIGURED = "LLM_NOT_CONFIGURED"
LLM_UNAVAILABLE = "LLM_UNAVAILABLE"                    # gateway unreachable (connection failure)
LLM_PROVIDER_UNAVAILABLE = "LLM_PROVIDER_UNAVAILABLE"  # gateway answered 5xx
LLM_TIMEOUT = "LLM_TIMEOUT"
LLM_AUTH_FAILED = "LLM_AUTH_FAILED"                    # 401/403: wrong or revoked key
LLM_RATE_LIMITED = "LLM_RATE_LIMITED"                  # 429
LLM_BAD_REQUEST = "LLM_BAD_REQUEST"                    # 400/404/413/422: bad model or parameters
LLM_INVALID_RESPONSE = "LLM_INVALID_RESPONSE"          # not OpenAI-compatible JSON/SSE
LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"              # error reported inside a stream, other statuses
LLM_STREAM_INTERRUPTED = "LLM_STREAM_INTERRUPTED"


@dataclass
class GenerationRequest:
    messages: List[Dict[str, Any]]
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Optional capabilities, passed through to providers that support them.
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None


@dataclass
class Usage:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class GenerationResult:
    content: str
    model: Optional[str]
    finish_reason: Optional[str]
    usage: Usage = field(default_factory=Usage)
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class StreamDelta:
    """One streamed piece. Exactly one of `text` / `final` is set."""

    text: Optional[str] = None
    final: Optional[GenerationResult] = None


@dataclass
class HealthStatus:
    configured: bool
    reachable: bool = False
    authenticated: bool = False
    models_available: Optional[int] = None
    # None = not checked (gateway without a model listing); False fails readiness.
    valid_response: Optional[bool] = None
    model_available: Optional[bool] = None
    completion_ok: Optional[bool] = None
    detail: Optional[str] = None

    @property
    def ok(self) -> bool:
        return (self.configured and self.reachable and self.authenticated and self.valid_response is not False
                and self.model_available is not False and self.completion_ok is not False)


class LLMProvider(ABC):
    name: str = "base"
    default_model: str = ""

    @abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult: ...

    @abstractmethod
    def stream(self, request: GenerationRequest) -> AsyncIterator[StreamDelta]: ...

    @abstractmethod
    async def health_check(self, *, deep: bool = False) -> HealthStatus: ...

    async def aclose(self) -> None:  # pragma: no cover - optional
        return None
