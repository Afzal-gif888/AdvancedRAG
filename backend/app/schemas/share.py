from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ShareCreate(BaseModel):
    expires_in_days: Optional[int] = Field(default=None, ge=1, le=365)


class ShareCreated(BaseModel):
    # The raw token is returned exactly once; only its hash is stored.
    token: str
    url: str
    created_at: datetime
    expires_at: Optional[datetime] = None


class ShareStatus(BaseModel):
    active: bool
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None


class SharedMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class SharedConversation(BaseModel):
    """Public view: no ids, emails, documents, metadata or internal state."""

    title: str
    shared_at: datetime
    messages: List[SharedMessage]
