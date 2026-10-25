import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import BigInteger, Column, ForeignKey, Index, Integer, String, Table
from sqlalchemy.orm import Mapped, mapped_column

from app.core.time import utcnow
from app.db.base import Base, JSONType, UTCDateTime

UPLOADED = "uploaded"
PROCESSING = "processing"
READY = "ready"
FAILED = "failed"
DELETED = "deleted"
# An older version replaced by a newer one that finished indexing. Hidden from search and listings, kept
# (file and vectors) for a retention window so the replacement can be rolled back.
SUPERSEDED = "superseded"

conversation_documents = Table(
    "conversation_documents",
    Base.metadata,
    Column("conversation_id", ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True),
    Column("document_id", ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True, index=True),
    Column("created_at", UTCDateTime, default=utcnow, nullable=False),
)


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        Index("ix_documents_user_id_created_at", "user_id", "created_at"),
        Index("ix_documents_user_id_checksum_sha256", "user_id", "checksum_sha256"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    # Sanitised display name; original_filename keeps what the client sent (also sanitised).
    filename: Mapped[str] = mapped_column(String(255))
    original_filename: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    checksum_sha256: Mapped[str] = mapped_column(String(64))
    # Server-generated; never derived from the client filename.
    storage_key: Mapped[str] = mapped_column(String(255), unique=True)
    # Which storage backend holds the original (local | s3 | db). Per document, so changing STORAGE_BACKEND
    # never strands documents that were stored with the previous one.
    storage_backend: Mapped[str] = mapped_column(String(16), default="local", server_default="local")
    status: Mapped[str] = mapped_column(String(20), default=UPLOADED, index=True)
    # RAG Core's extraction report (native/OCR pages, blank/unprocessed/failed pages, warnings), set when
    # indexing succeeds. `partial: true` means some content is not searchable although the document is ready.
    extraction: Mapped[Optional[dict]] = mapped_column(JSONType)
    # Versioning: all versions of one logical document share `logical_id` (null on rows created before
    # versions existed = the row is its own logical document).
    logical_id: Mapped[Optional[uuid.UUID]] = mapped_column(index=True)
    version: Mapped[int] = mapped_column(Integer, default=1, server_default="1")
    # The previous version. Deliberately not a foreign key: deleting an old version must never be blocked by,
    # or cascade into, the version that replaced it (readers treat a missing target as "no previous version").
    replaces_id: Mapped[Optional[uuid.UUID]] = mapped_column()
    superseded_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)
    processing_error: Mapped[Optional[str]] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)
