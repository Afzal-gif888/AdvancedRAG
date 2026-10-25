import hashlib
import logging
import re
import struct
import unicodedata
import uuid
import zipfile
from dataclasses import dataclass
from typing import BinaryIO, List, Optional, Sequence, Tuple

import anyio
from fastapi import UploadFile
from sqlalchemy import delete, func, insert, or_, select, text, update
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
IMAGE_EXTS = {"png", "jpg", "jpeg"}
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


def image_pixels(fileobj: BinaryIO, ext: str) -> Optional[int]:
    """Width x height of a PNG/JPEG read from its header (no decoding). None if the header is not understood.
    Lets the upload reject a decompression bomb before any pixel is decoded anywhere."""
    fileobj.seek(0)
    head = fileobj.read(32)
    if ext == "png":
        if len(head) < 24 or head[12:16] != b"IHDR":
            return None
        w, h = struct.unpack(">II", head[16:24])
        return w * h
    fileobj.seek(2)
    while True:
        marker = fileobj.read(4)
        if len(marker) < 4 or marker[0] != 0xFF:
            return None
        code, length = marker[1], struct.unpack(">H", marker[2:4])[0]
        if 0xC0 <= code <= 0xCF and code not in (0xC4, 0xC8, 0xCC):
            sof = fileobj.read(5)
            if len(sof) < 5:
                return None
            h, w = struct.unpack(">HH", sof[1:5])
            return w * h
        if length < 2:
            return None
        fileobj.seek(length - 2, 1)


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
    # Superseded versions are history, not documents: they appear only under /documents/{id}/versions.
    filters = [Document.user_id == user_id, Document.status.notin_((doc_status.DELETED, doc_status.SUPERSEDED))]
    if status:
        filters.append(Document.status == status)
    total = await db.scalar(select(func.count()).select_from(Document).where(*filters))
    rows = await db.scalars(select(Document).where(*filters).order_by(Document.created_at.desc(), Document.id).limit(limit).offset(offset))
    return rows.all(), total or 0


# ---------------------------------------------------------------- upload

async def _validated_upload(file: UploadFile) -> Tuple[str, str, ScannedFile]:
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
    if ext in IMAGE_EXTS:
        pixels = await anyio.to_thread.run_sync(image_pixels, file.file, ext)
        file.file.seek(0)
        if pixels is None:
            raise UnsupportedMediaType("The image could not be read.", code="FILE_CONTENT_MISMATCH")
        if pixels > settings.MAX_IMAGE_PIXELS:
            raise PayloadTooLarge(f"Images must have at most {settings.MAX_IMAGE_PIXELS // 1_000_000} megapixels.", code="IMAGE_TOO_LARGE")
    return filename, ext, scanned


async def _lock_content(db: AsyncSession, user_id: uuid.UUID, sha256: str) -> None:
    """Serialise concurrent uploads of the same content by the same user (PostgreSQL advisory lock held to the
    end of the transaction), so two simultaneous identical uploads cannot both pass the duplicate check."""
    if db.bind is not None and db.bind.dialect.name == "postgresql":
        key = int.from_bytes(hashlib.sha256(f"{user_id}:{sha256}".encode()).digest()[:8], "big", signed=True)
        await db.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": key})


async def _find_duplicate(db: AsyncSession, user_id: uuid.UUID, sha256: str) -> Optional[Document]:
    return await db.scalar(
        select(Document)
        .where(
            Document.user_id == user_id,
            Document.checksum_sha256 == sha256,
            Document.status.in_((doc_status.UPLOADED, doc_status.PROCESSING, doc_status.READY)),
        )
        .order_by(Document.created_at)
        .limit(1)
    )


async def _check_quota(db: AsyncSession, user_id: uuid.UUID, size: int) -> None:
    count, used = (
        await db.execute(
            select(func.count(), func.coalesce(func.sum(Document.size_bytes), 0)).where(
                Document.user_id == user_id, Document.status != doc_status.DELETED
            )
        )
    ).one()
    if count >= settings.MAX_DOCUMENTS_PER_USER:
        raise Conflict("You have reached the maximum number of documents.", code="DOCUMENT_QUOTA_EXCEEDED")
    if used + size > settings.MAX_STORAGE_BYTES_PER_USER:
        raise Conflict("You have reached your storage limit.", code="STORAGE_QUOTA_EXCEEDED")


async def _store_and_enqueue(db: AsyncSession, user_id: uuid.UUID, file: UploadFile, storage: Storage, *, filename: str, ext: str,
                             scanned: ScannedFile, **extra) -> Document:
    """Stores the bytes, adds the document row and its ingest job, commits once. The job carries no file
    content: the worker reads the object from storage by key (durable, nothing duplicated in the job table)."""
    doc_id = uuid.uuid4()
    key = make_storage_key(user_id, doc_id)
    await storage.put(key, file.file, db=db)  # the db backend writes into this transaction
    doc = Document(
        id=doc_id,
        user_id=user_id,
        filename=filename,
        original_filename=filename,
        mime_type=MIME_BY_EXT[ext],
        size_bytes=scanned.size,
        checksum_sha256=scanned.sha256,
        storage_key=key,
        storage_backend=settings.STORAGE_BACKEND,
        status=doc_status.UPLOADED,
        logical_id=extra.pop("logical_id", doc_id),
        **extra,
    )
    db.add(doc)
    job_service.enqueue(db, user_id, job_types.DOCUMENT_INGEST, resource_id=doc_id, payload={})
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        await storage.delete(key)  # no orphaned blobs (a no-op for the db backend: it rolled back with us)
        raise
    await db.refresh(doc)
    job_service.notify_worker()
    return doc


async def upload_with_status(db: AsyncSession, user_id: uuid.UUID, file: UploadFile, storage: Storage) -> Tuple[Document, bool]:
    """Returns (document, created). Uploading content the user already has (same SHA-256, not deleted/failed)
    returns the existing document and queues nothing."""
    filename, ext, scanned = await _validated_upload(file)
    await _lock_content(db, user_id, scanned.sha256)
    duplicate = await _find_duplicate(db, user_id, scanned.sha256)
    if duplicate is not None:
        await db.commit()  # nothing was written; this ends the transaction and releases the advisory lock
        logger.info("document_duplicate_upload", extra={"document_id": str(duplicate.id)})
        return duplicate, False
    await _check_quota(db, user_id, scanned.size)
    doc = await _store_and_enqueue(db, user_id, file, storage, filename=filename, ext=ext, scanned=scanned)
    logger.info("document_uploaded", extra={"document_id": str(doc.id), "size_bytes": scanned.size, "ext": ext, "storage": doc.storage_backend})
    return doc, True


async def upload(db: AsyncSession, user_id: uuid.UUID, file: UploadFile, storage: Storage) -> Document:
    return (await upload_with_status(db, user_id, file, storage))[0]


# ---------------------------------------------------------------- versions

def logical_id_of(doc: Document) -> uuid.UUID:
    return doc.logical_id or doc.id


async def _versions(db: AsyncSession, user_id: uuid.UUID, logical: uuid.UUID, include_deleted: bool = False) -> List[Document]:
    stmt = select(Document).where(Document.user_id == user_id, or_(Document.logical_id == logical, Document.id == logical))
    if not include_deleted:
        stmt = stmt.where(Document.status != doc_status.DELETED)
    return list((await db.scalars(stmt.order_by(Document.version.desc(), Document.created_at.desc()))).all())


async def list_versions(db: AsyncSession, user_id: uuid.UUID, document_id: uuid.UUID) -> List[Document]:
    doc = await get_owned(db, user_id, document_id)
    return await _versions(db, user_id, logical_id_of(doc))


async def replace(db: AsyncSession, user_id: uuid.UUID, document_id: uuid.UUID, file: UploadFile, storage: Storage) -> Tuple[Document, bool]:
    """Upload a new version of a document. The current version stays searchable until the new one has finished
    indexing; the switch (new -> ready, old -> superseded) is a single transaction, so queries never see both
    versions or neither. Returns (new_version, created); identical content is a no-op."""
    current = await get_owned(db, user_id, document_id)
    if current.status != doc_status.READY:
        raise Conflict("Only a ready document can be replaced.", code="DOCUMENT_NOT_REPLACEABLE")
    filename, ext, scanned = await _validated_upload(file)
    await _lock_content(db, user_id, scanned.sha256)
    if scanned.sha256 == current.checksum_sha256:
        await db.commit()  # nothing was written; releases the advisory lock
        return current, False
    logical = logical_id_of(current)
    family = await _versions(db, user_id, logical)
    if any(d.status in (doc_status.UPLOADED, doc_status.PROCESSING) for d in family):
        raise Conflict("A new version of this document is already being processed.", code="REPLACEMENT_IN_PROGRESS")
    await _check_quota(db, user_id, scanned.size)
    if current.logical_id is None:
        current.logical_id = logical
    doc = await _store_and_enqueue(
        db, user_id, file, storage, filename=filename, ext=ext, scanned=scanned,
        logical_id=logical, version=max(d.version for d in family) + 1, replaces_id=current.id,
    )
    logger.info("document_replacement_uploaded", extra={"document_id": str(doc.id), "replaces": str(current.id), "version": doc.version})
    return doc, True


async def _repoint_conversations(db: AsyncSession, old_id: uuid.UUID, new_id: uuid.UUID) -> None:
    rows = (await db.execute(select(conversation_documents.c.conversation_id).where(conversation_documents.c.document_id == old_id))).all()
    if not rows:
        return
    have = {r[0] for r in (await db.execute(select(conversation_documents.c.conversation_id).where(conversation_documents.c.document_id == new_id))).all()}
    await db.execute(delete(conversation_documents).where(conversation_documents.c.document_id == old_id))
    fresh = [{"conversation_id": r[0], "document_id": new_id, "created_at": utcnow()} for r in rows if r[0] not in have]
    if fresh:
        await db.execute(insert(conversation_documents), fresh)


async def finalize_ready(db: AsyncSession, document_id: uuid.UUID, extraction: Optional[dict]) -> None:
    """Indexing finished and was verified: the document becomes ready and, if it replaces an older version,
    that version is superseded IN THE SAME TRANSACTION (the caller commits once)."""
    now = utcnow()
    await db.execute(
        update(Document)
        .where(Document.id == document_id, Document.status != doc_status.DELETED)
        .values(status=doc_status.READY, processing_error=None, extraction=extraction, updated_at=now)
    )
    doc = await db.scalar(select(Document).where(Document.id == document_id).execution_options(populate_existing=True))
    if doc is None or doc.replaces_id is None:
        return
    old = await db.scalar(select(Document).where(Document.id == doc.replaces_id, Document.status == doc_status.READY))
    if old is not None:
        old.status, old.superseded_at, old.updated_at = doc_status.SUPERSEDED, now, now
        await _repoint_conversations(db, old.id, doc.id)


async def restore_version(db: AsyncSession, user_id: uuid.UUID, document_id: uuid.UUID) -> Document:
    """Make a superseded version current again (rollback, or roll forward). Atomic swap; both versions must
    still exist (superseded versions are purged after SUPERSEDED_RETENTION_DAYS)."""
    target = await db.scalar(
        select(Document).where(Document.id == document_id, Document.user_id == user_id, Document.status == doc_status.SUPERSEDED)
    )
    if target is None:
        raise DocumentNotFound()
    logical = logical_id_of(target)
    now = utcnow()
    current = [d for d in await _versions(db, user_id, logical) if d.status == doc_status.READY]
    for d in current:
        d.status, d.superseded_at, d.updated_at = doc_status.SUPERSEDED, now, now
        await _repoint_conversations(db, d.id, target.id)
    target.status, target.superseded_at, target.updated_at = doc_status.READY, None, now
    await db.commit()
    await db.refresh(target)
    return target


# ---------------------------------------------------------------- reprocess / delete

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
        job_service.enqueue(db, d.user_id, job_types.DOCUMENT_DELETE, resource_id=d.id,
                            payload={"storage_key": d.storage_key, "storage_backend": d.storage_backend},
                            max_attempts=settings.DELETE_JOB_MAX_ATTEMPTS)


async def delete_document(db: AsyncSession, user_id: uuid.UUID, document_id: uuid.UUID) -> None:
    """Deleting a document deletes every version of it (superseded ones included)."""
    doc = await get_owned(db, user_id, document_id)
    family = await _versions(db, user_id, logical_id_of(doc))
    await _soft_delete(db, family or [doc])
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
    """The user's ready documents (newest first), excluding `exclude`. Ready means RAG Core finished indexing it;
    uploaded/processing/failed/superseded/deleted documents are never offered."""
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
