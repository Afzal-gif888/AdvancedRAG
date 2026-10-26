import uuid
from typing import List, Literal, Optional
from urllib.parse import quote

from fastapi import APIRouter, File, Query, Response, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.deps import DB, CurrentUser
from app.integrations.storage import get_storage, get_storage_for
from app.middleware.rate_limit import hit_user_limit
from app.schemas.common import Page, error_responses
from app.schemas.document import DocumentResponse
from app.services import document_service

router = APIRouter()


def _response(doc, duplicate: bool = False) -> DocumentResponse:
    out = DocumentResponse.model_validate(doc)
    return out.model_copy(update={"duplicate": True}) if duplicate else out


@router.get("", response_model=Page[DocumentResponse], responses=error_responses(401))
async def list_documents(
    user: CurrentUser,
    db: DB,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status_filter: Optional[Literal["uploaded", "processing", "ready", "failed"]] = Query(None, alias="status"),
):
    items, total = await document_service.list_for_user(db, user.id, limit, offset, status_filter)
    return Page(items=list(items), total=total, limit=limit, offset=offset)


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    responses={200: {"description": "The same content is already in your Knowledge Base (duplicate=true)."}, **error_responses(401, 409, 413, 415, 422, 429)},
)
async def upload_document(user: CurrentUser, db: DB, response: Response, file: UploadFile = File(...)):
    """Multipart upload (field `file`), one file per request. Returns immediately with
    status `uploaded`; ingestion runs in the background — poll GET /documents/{id}.
    Uploading content you already have returns the existing document with HTTP 200 and `duplicate: true`."""
    await hit_user_limit("upload", user.id, times=60, seconds=600)
    doc, created = await document_service.upload_with_status(db, user.id, file, get_storage())
    if not created:
        response.status_code = status.HTTP_200_OK
    return _response(doc, duplicate=not created)


@router.get("/{document_id}", response_model=DocumentResponse, responses=error_responses(401, 404))
async def get_document(document_id: uuid.UUID, user: CurrentUser, db: DB):
    return await document_service.get_owned(db, user.id, document_id)


@router.get("/{document_id}/content", responses={200: {"content": {"application/octet-stream": {}}}, **error_responses(401, 404)})
async def download_document(document_id: uuid.UUID, user: CurrentUser, db: DB):
    doc = await document_service.get_owned(db, user.id, document_id)
    storage = get_storage_for(doc.storage_backend)
    if not await storage.exists(doc.storage_key):
        raise document_service.DocumentNotFound()
    return StreamingResponse(
        storage.stream(doc.storage_key),
        media_type=doc.mime_type,
        headers={
            # Always a download, never rendered inline by the browser.
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(doc.filename)}",
            "Content-Length": str(doc.size_bytes),
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/{document_id}/reprocess", response_model=DocumentResponse, responses=error_responses(401, 404, 409))
async def reprocess_document(document_id: uuid.UUID, user: CurrentUser, db: DB):
    return await document_service.reprocess(db, user.id, document_id)


@router.post(
    "/{document_id}/replace",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(401, 404, 409, 413, 415, 422, 429),
)
async def replace_document(document_id: uuid.UUID, user: CurrentUser, db: DB, response: Response, file: UploadFile = File(...)):
    """Upload a new version of a ready document. The current version keeps answering questions until the new
    one is fully indexed, then the switch happens atomically. Identical content is a no-op (HTTP 200)."""
    await hit_user_limit("upload", user.id, times=60, seconds=600)
    doc, created = await document_service.replace(db, user.id, document_id, file, get_storage())
    if not created:
        response.status_code = status.HTTP_200_OK
    return _response(doc, duplicate=not created)


@router.get("/{document_id}/versions", response_model=List[DocumentResponse], responses=error_responses(401, 404))
async def list_versions(document_id: uuid.UUID, user: CurrentUser, db: DB):
    """Every version of this document, newest first (superseded versions are kept for a retention window)."""
    return await document_service.list_versions(db, user.id, document_id)


@router.post("/{document_id}/restore", response_model=DocumentResponse, responses=error_responses(401, 404))
async def restore_version(document_id: uuid.UUID, user: CurrentUser, db: DB):
    """Make a superseded version the current one again (rollback)."""
    return await document_service.restore_version(db, user.id, document_id)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, user: CurrentUser, db: DB):
    """Removes the document (all of its versions) from all conversations immediately; the files and vectors are
    deleted from storage / RAG Core in the background."""
    await document_service.delete_document(db, user.id, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
