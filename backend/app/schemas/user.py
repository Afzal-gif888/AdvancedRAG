import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import ORMModel


class UserResponse(ORMModel):
    """Safe user representation — never includes hashes, role or security fields."""

    id: uuid.UUID
    email: str
    full_name: Optional[str] = None
    is_verified: bool
    plan: str
    created_at: datetime


class ProfileUpdate(BaseModel):
    # Only whitelisted fields; anything else (id, role, email, password_hash...) is rejected.
    model_config = {"extra": "forbid"}

    full_name: Optional[str] = Field(default=None, max_length=100)

    @field_validator("full_name")
    @classmethod
    def _clean(cls, v: Optional[str]) -> Optional[str]:
        return v.strip() if v is not None else v
