import hashlib
import logging
import re
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass
from typing import BinaryIO, List, Optional, Sequence

import anyio
from fastapi import UploadFile
from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import AppError, Conflict, NotFound, PayloadTooLarge, UnsupportedMediaType, ValidationFailed
from app.core.time import utcnow
from app.db.models import Document, conversation_documents
from app.db.models import document as doc_status
from app.db.models import job as job_types
from app.integrations.storage import Storage, make_storage_key
from app.services import conversation_service, job_service

logger = logging.getLogger("advancerag.documents")

# Server-side MIME types by extension — the browser-supplied Content-Type is never trusted.
MIME_BY_EXT = {
    "pdf": "application/pdf",
    "txt": "text/plain",
    "md": "text/markdown",
    "csv": "text/csv",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
}
TEXT_EXTS = {"txt", "md", "csv"}
_FORBIDDEN_CHARS = re.compile(r'[\x00-\x1f\x7f<>:"/\\|?*]')


class DocumentNotFound(NotFound):
    code = "DOCUMENT_NOT_FOUND"
    message = "Document not found."


def sanitize_filename(raw: Optional[str]) -> str:
    name = unicodedata.normalize("NFKC", raw or "")
    # Keep only the final path component (defeats ../ and C:\ tricks), then strip unsafe characters.
    name = re.split(r"[\\/]", name)[-1]
    name = _FORBIDDEN_CHARS.sub("_", name).strip(" .")
    if len(name) > 200:
        stem, dot, ext = name.rpartition(".")
        name = (stem[: 200 - len(ext) - 1] + "." + ext) if dot and len(ext) <= 10 else name[:200]
    return name


def extension_of(filename: str) -> str:
    _, dot, ext = filename.rpartition(".")
    return ext.lower() if dot else ""


def _content_matches(ext: str, head: bytes, fileobj: BinaryIO) -> bool:
    if ext == "pdf":
        return head.startswith(b"%PDF-")
    if ext == "png":
        return head.startswith(b"\x89PNG\r\n\x1a\n")
    if ext in ("jpg", "jpeg"):
        return head.startswith(b"\xff\xd8\xff")
    if ext == "docx":
        if not head.startswith(b"PK\x03\x04"):
            return False
        try:
            fileobj.seek(0)
            with zipfile.ZipFile(fileobj) as zf:
                return "word/document.xml" in zf.namelist()
        except zipfile.BadZipFile:
            return False
    if ext in TEXT_EXTS:
        return b"\x00" not in head  # binary content masquerading as text
    return False


@dataclass
class ScannedFile:
    size: int
    sha256: str
    content_ok: bool


def _scan(fileobj: BinaryIO, ext: str, max_bytes: int) -> ScannedFile:
    fileobj.seek(0)
    digest, size = hashlib.sha256(), 0
    head = fileobj.read(8192)
    chunk = head
    while chunk:
        size += len(chunk)
        if size > max_bytes:
            return ScannedFile(size, "", False)
        digest.update(chunk)
        chunk = fileobj.read(1024 * 1024)
    ok = _content_matches(ext, head, fileobj)
    fileobj.seek(0)
    return ScannedFile(size, digest.hexdigest(), ok)


async def get_owned(db: AsyncSession, user_id: uuid.UUID, document_id: uuid.UUID) -> Document:
    doc = await db.scalar(
        select(Document).where(
            Document.id == document_id, Document.user_id == user_id, Document.status != doc_status.DELETED
        )
    )
    if doc is None:
        raise DocumentNotFound()
    return doc


async def list_for_user(db: AsyncSession, user_id: uuid.UUID, limit: int, offset: int, status: Optional[str] = None):
    filters = [Document.user_id == user_id, Document.status != doc_status.DELETED]
    if status:
        filters.append(Document.status == status)
    total = await db.scalar(select(func.count()).select_from(Document).where(*filters))
    rows = await db.scalars(select(Document).where(*filters).order_by(Document.created_at.desc(), Document.id).limit(limit).offset(offset))
    return rows.all(), total or 0


async def upload(db: AsyncSession, user_id: uuid.UUID, file: UploadFile, storage: Storage) -> Document:
    filename = sanitize_filename(file.filename)
    ext = extension_of(filename)
    if not filename or ext not in settings.allowed_extensions or ext not in MIME_BY_EXT:
        raise UnsupportedMediaType(
            f"Unsupported file type. Allowed: {', '.join(sorted(settings.allowed_extensions & MIME_BY_EXT.keys()))}."
        )

    scanned = await anyio.to_thread.run_sync(_scan, file.file, ext, settings.MAX_UPLOAD_BYTES)
    if scanned.size > settings.MAX_UPLOAD_BYTES:
        raise PayloadTooLarge(f"Files must be at most {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", code="FILE_TOO_LARGE")
    if scanned.size == 0:
        raise ValidationFailed("The file is empty.", code="EMPTY_FILE")
    if not scanned.content_ok:
        raise UnsupportedMediaType("The file content does not match its extension.", code="FILE_CONTENT_MISMATCH")

    count, used = (
        await db.execute(
            select(func.count(), func.coalesce(func.sum(Document.size_bytes), 0)).where(
                Document.user_id == user_id, Document.status != doc_status.DELETED
            )
        )
    ).one()
    if count >= settings.MAX_DOCUMENTS_PER_USER:
        raise Conflict("You have reached the maximum number of documents.", code="DOCUMENT_QUOTA_EXCEEDED")
    if used + scanned.size > settings.MAX_STORAGE_BYTES_PER_USER:
        raise Conflict("You have reached your storage limit.", code="STORAGE_QUOTA_EXCEEDED")

    doc_id = uuid.uuid4()
    key = make_storage_key(user_id, doc_id)
    await storage.put(key, file.file)
    doc = Document(
        id=doc_id,
        user_id=user_id,
        filename=filename,
        original_filename=filename,
        mime_type=MIME_BY_EXT[ext],
        size_bytes=scanned.size,
        checksum_sha256=scanned.sha256,
        storage_key=key,
        status=doc_status.UPLOADED,
    )
    db.add(doc)
    job_service.enqueue(db, user_id, job_types.DOCUMENT_INGEST, resource_id=doc_id)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        await storage.delete(key)  # no orphaned blobs
        raise
    await db.refresh(doc)
    job_service.notify_worker()
    logger.info("document_uploaded", extra={"document_id": str(doc_id), "size_bytes": scanned.size, "ext": ext})
    return doc


async def reprocess(db: AsyncSession, user_id: uuid.UUID, document_id: uuid.UUID) -> Document:
    doc = await get_owned(db, user_id, document_id)
    if doc.status != doc_status.FAILED:
        raise AppError("Only failed documents can be reprocessed.", code="DOCUMENT_NOT_RETRYABLE", status_code=409)
    doc.status = doc_status.UPLOADED
    doc.processing_error = None
    job_service.enqueue(db, user_id, job_types.DOCUMENT_INGEST, resource_id=doc.id)
    await db.commit()
    await db.refresh(doc)
    job_service.notify_worker()
    return doc


async def _soft_delete(db: AsyncSession, docs: Sequence[Document]) -> None:
    if not docs:
        return
    ids = [d.id for d in docs]
    await db.execute(delete(conversation_documents).where(conversation_documents.c.document_id.in_(ids)))
    for d in docs:
        d.status = doc_status.DELETED
        # Vector + blob removal happens asynchronously; the row is hard-deleted when that succeeds.
        job_service.enqueue(db, d.user_id, job_types.DOCUMENT_DELETE, resource_id=d.id, payload={"storage_key": d.storage_key},
                            max_attempts=settings.DELETE_JOB_MAX_ATTEMPTS)


async def delete_document(db: AsyncSession, user_id: uuid.UUID, document_id: uuid.UUID) -> None:
    doc = await get_owned(db, user_id, document_id)
    await _soft_delete(db, [doc])
    await db.commit()
    job_service.notify_worker()


async def delete_all_for_user(db: AsyncSession, user_id: uuid.UUID) -> None:
    docs = (await db.scalars(select(Document).where(Document.user_id == user_id, Document.status != doc_status.DELETED))).all()
    await _soft_delete(db, docs)


# ---------------------------------------------------------------- conversation links

async def attach_to_conversation(
    db: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID, document_ids: List[uuid.UUID], commit: bool = True
) -> List[Document]:
    await conversation_service.get_owned(db, user_id, conversation_id)
    unique_ids = list(dict.fromkeys(document_ids))
    docs = (
        await db.scalars(
            select(Document).where(
                Document.id.in_(unique_ids), Document.user_id == user_id, Document.status != doc_status.DELETED
            )
        )
    ).all()
    if len(docs) != len(unique_ids):
        # Includes other users' documents — reported as not found, never "forbidden".
        raise DocumentNotFound()
    already = set(
        (
            await db.scalars(
                select(conversation_documents.c.document_id).where(
                    conversation_documents.c.conversation_id == conversation_id,
                    conversation_documents.c.document_id.in_(unique_ids),
                )
            )
        ).all()
    )
    new_rows = [{"conversation_id": conversation_id, "document_id": d.id, "created_at": utcnow()} for d in docs if d.id not in already]
    if new_rows:
        await db.execute(insert(conversation_documents), new_rows)
    if commit:
        await db.commit()
    by_id = {d.id: d for d in docs}
    return [by_id[i] for i in unique_ids]


async def detach_from_conversation(db: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID, document_id: uuid.UUID) -> None:
    await conversation_service.get_owned(db, user_id, conversation_id)
    result = await db.execute(
        delete(conversation_documents).where(
            conversation_documents.c.conversation_id == conversation_id,
            conversation_documents.c.document_id == document_id,
        )
    )
    if result.rowcount == 0:
        raise DocumentNotFound()
    await db.commit()


async def list_library_documents(db: AsyncSession, user_id: uuid.UUID, exclude: Sequence[uuid.UUID], limit: int) -> List[Document]:
    """The user's ready documents (newest first), excluding `exclude`. Ready means RAG Core
    finished indexing it; uploaded/processing/failed/deleted documents are never offered."""
    query = select(Document).where(Document.user_id == user_id, Document.status == doc_status.READY)
    if exclude:
        query = query.where(Document.id.not_in(list(exclude)))
    rows = await db.scalars(query.order_by(Document.created_at.desc(), Document.id).limit(limit))
    return list(rows.all())


async def count_pending_documents(db: AsyncSession, user_id: uuid.UUID) -> int:
    """Knowledge Base documents not searchable yet (uploaded or still being indexed)."""
    return int(await db.scalar(
        select(func.count()).select_from(Document)
        .where(Document.user_id == user_id, Document.status.in_((doc_status.UPLOADED, doc_status.PROCESSING)))
    ) or 0)


async def list_conversation_documents(db: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID) -> List[Document]:
    rows = await db.scalars(
        select(Document)
        .join(conversation_documents, conversation_documents.c.document_id == Document.id)
        .where(
            conversation_documents.c.conversation_id == conversation_id,
            Document.user_id == user_id,
            Document.status != doc_status.DELETED,
        )
        .order_by(conversation_documents.c.created_at)
    )
    return list(rows.all())


async def mark_status(db: AsyncSession, document_id: uuid.UUID, status: str, error: Optional[str] = None) -> None:
    await db.execute(
        update(Document)
        .where(Document.id == document_id, Document.status != doc_status.DELETED)
        .values(status=status, processing_error=error, updated_at=utcnow())
    )
