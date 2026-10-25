import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from app.schemas.common import ORMModel


class DocumentResponse(ORMModel):
    id: uuid.UUID
    filename: str
    mime_type: str
    size_bytes: int
    status: str
    processing_error: Optional[str] = None
    # Where the text came from and what could not be made searchable (null until indexing finished):
    # {method, partial, total_pages, ocr_pages, unprocessed_pages, failed_pages, blank_pages, warnings[], ...}
    extraction: Optional[Dict[str, Any]] = None
    version: int = 1
    logical_id: Optional[uuid.UUID] = None
    replaces_id: Optional[uuid.UUID] = None
    # True when the upload matched a document you already have (same content): no new work was queued.
    duplicate: bool = False
    created_at: datetime
    updated_at: datetime
