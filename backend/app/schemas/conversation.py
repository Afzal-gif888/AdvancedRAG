import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import ORMModel


def _clean_title(v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    v = " ".join(v.split())
    if not v:
        raise ValueError("Title must not be empty.")
    return v


class ConversationCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)

    @field_validator("title")
    @classmethod
    def _title(cls, v: Optional[str]) -> Optional[str]:
        return _clean_title(v)


class ConversationUpdate(BaseModel):
    model_config = {"extra": "forbid"}

    title: Optional[str] = Field(default=None, max_length=200)
    archived: Optional[bool] = None

    @field_validator("title")
    @classmethod
    def _title(cls, v: Optional[str]) -> Optional[str]:
        return _clean_title(v)


class ConversationResponse(ORMModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    archived_at: Optional[datetime] = None


class ConversationSearchResult(ConversationResponse):
    # Short excerpt of the first matching message, when the match was in message content.
    snippet: Optional[str] = None


class AttachDocumentsRequest(BaseModel):
    document_ids: List[uuid.UUID] = Field(min_length=1, max_length=50)
