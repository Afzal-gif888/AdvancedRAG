"""Agent Core error categories.

The wire contract with the backend has five stable codes (the backend drops anything else):

    RAG_SERVICE_UNAVAILABLE   document search failed (down, timeout, invalid or rejected response)
    LLM_UNAVAILABLE           the language model failed (any LLM-layer error not listed below)
    LLM_TIMEOUT               the language model timed out
    LLM_RATE_LIMITED          the language model is rate limited upstream
    GENERATION_FAILED         empty/invalid answer or an unexpected Agent Core error

`reason` keeps the precise internal cause (e.g. `rag_timeout`, `LLM_AUTH_FAILED`,
`llm_stream_interrupted`) for logs and metrics only; it is never shown to users.
"""
from typing import Optional

RAG_SERVICE_UNAVAILABLE = "RAG_SERVICE_UNAVAILABLE"
LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
LLM_TIMEOUT = "LLM_TIMEOUT"
LLM_RATE_LIMITED = "LLM_RATE_LIMITED"
GENERATION_FAILED = "GENERATION_FAILED"

PUBLIC_CODES = {RAG_SERVICE_UNAVAILABLE, LLM_UNAVAILABLE, LLM_TIMEOUT, LLM_RATE_LIMITED, GENERATION_FAILED}

# Internal reasons for RAG failures.
RAG_TIMEOUT = "rag_timeout"
RAG_UNREACHABLE = "rag_unreachable"
RAG_HTTP_ERROR = "rag_http_error"
RAG_INVALID_RESPONSE = "rag_invalid_response"

USER_MESSAGES = {
    RAG_SERVICE_UNAVAILABLE: "Document search is unavailable.",
    LLM_UNAVAILABLE: "The language model is unavailable.",
    LLM_TIMEOUT: "The language model took too long to respond.",
    LLM_RATE_LIMITED: "The AI service is temporarily rate-limited. Please try again shortly.",
    GENERATION_FAILED: "The response could not be generated.",
}


class AgentError(Exception):
    """A safe, categorised failure. `code` and `message` may reach the user; `reason` may not."""

    def __init__(self, code: str, message: Optional[str] = None, retryable: bool = True, reason: Optional[str] = None):
        assert code in PUBLIC_CODES, code
        self.code = code
        self.message = message or USER_MESSAGES[code]
        self.retryable = retryable
        self.reason = reason or code
        super().__init__(self.message)


def rag_error(reason: str) -> AgentError:
    return AgentError(RAG_SERVICE_UNAVAILABLE, reason=reason)


def llm_error(upstream_code: Optional[str], *, retryable: bool = True) -> AgentError:
    """Map an LLM-layer code (LLM_RATE_LIMITED, LLM_AUTH_FAILED, LLM_INVALID_RESPONSE,
    LLM_STREAM_INTERRUPTED, ...) to a public category, keeping the original as `reason`."""
    reason = upstream_code or "llm_unknown_error"
    if upstream_code == LLM_RATE_LIMITED:
        return AgentError(LLM_RATE_LIMITED, retryable=True, reason=reason)
    if upstream_code == LLM_TIMEOUT:
        return AgentError(LLM_TIMEOUT, retryable=True, reason=reason)
    # Auth/config/bad-request/invalid-response details stay server-side.
    return AgentError(LLM_UNAVAILABLE, retryable=retryable, reason=reason)
