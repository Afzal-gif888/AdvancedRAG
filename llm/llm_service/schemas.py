from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    model_config = {"extra": "allow"}  # tool_call_id / tool_calls / name pass through

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = Field(default=None, max_length=400_000)


class ChatRequest(BaseModel):
    messages: List[Message] = Field(min_length=1, max_length=400)
    # None = provider default (FREELLMAPI_MODEL). Callers may pick "auto:fast" etc.
    model: Optional[str] = Field(default=None, pattern=r"^[A-Za-z0-9._:/@-]{1,120}$")
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=None, ge=1, le=65_536)
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, max_length=64)
    tool_choice: Optional[Any] = None


class UsageOut(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    content: str
    model: Optional[str]
    finish_reason: Optional[str]
    usage: UsageOut
    tool_calls: Optional[List[Dict[str, Any]]] = None
    latency_ms: float
