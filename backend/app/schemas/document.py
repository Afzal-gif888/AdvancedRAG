import uuid
from datetime import datetime
from typing import Optional

from app.schemas.common import ORMModel


class DocumentResponse(ORMModel):
    id: uuid.UUID
    filename: str
    mime_type: str
    size_bytes: int
    status: str
    processing_error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
