"""Durable object storage inside PostgreSQL (STORAGE_BACKEND=db).

A fallback for deployments without S3-compatible storage and with an ephemeral disk (Render free): the
original file is kept in 1 MiB parts, written in the same transaction as the document and its job, so a
document never exists without its bytes. Meant for modest volumes; object storage (S3/R2) is the better
home for large files (see docs/verification/STORAGE.md).
"""
import uuid
from datetime import datetime

from sqlalchemy import BigInteger, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column

from app.core.time import utcnow
from app.db.base import Base, UTCDateTime


class StoredObject(Base):
    __tablename__ = "stored_objects"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    part_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class StoredObjectPart(Base):
    __tablename__ = "stored_object_parts"

    key: Mapped[str] = mapped_column(ForeignKey("stored_objects.key", ondelete="CASCADE"), primary_key=True)
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    data: Mapped[bytes] = mapped_column(LargeBinary)
