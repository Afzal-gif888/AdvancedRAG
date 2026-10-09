import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings


class MessageCreate(BaseModel):
    """Only user content is accepted — role is fixed server-side; assistant messages come from Agent Core."""

    model_config = {"extra": "forbid"}

    content: str = Field(min_length=1)
    # Documents to attach to the conversation before answering (must be owned by the caller).
    document_ids: List[uuid.UUID] = Field(default_factory=list, max_length=50)

    @field_validator("content")
    @classmethod
    def _check(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message must not be empty.")
        if len(v) > settings.MAX_MESSAGE_CHARS:
            raise ValueError(f"Message must be at most {settings.MAX_MESSAGE_CHARS} characters.")
        return v


class Attachment(BaseModel):
    id: uuid.UUID
    filename: str


class Citation(BaseModel):
    model_config = {"extra": "ignore"}

    source_id: Optional[int] = Field(default=None, ge=1, le=1000)  # the [n] marker in the answer text
    document_id: Optional[uuid.UUID] = None
    filename: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = Field(default=None, max_length=200)  # RAG Core chunk reference
    snippet: Optional[str] = Field(default=None, max_length=1000)


class MessageResponse(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    role: Literal["user", "assistant"]
    content: str
    status: str
    attachments: List[Attachment] = []
    citations: List[Citation] = []
    error_code: Optional[str] = None
    created_at: datetime


class SendMessageResponse(BaseModel):
    user_message: MessageResponse
    assistant_message: MessageResponse
