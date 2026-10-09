import uuid
from typing import List, Literal, Optional

from fastapi import APIRouter, Query, Response, status

from app.api.deps import DB, CurrentUser
from app.middleware.rate_limit import hit_user_limit
from app.schemas.common import Page, error_responses
from app.schemas.conversation import AttachDocumentsRequest, ConversationCreate, ConversationResponse, ConversationUpdate
from app.schemas.document import DocumentResponse
from app.schemas.share import ShareCreate, ShareCreated, ShareStatus
from app.services import conversation_service, document_service, share_service

router = APIRouter()


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED, responses=error_responses(401, 422))
async def create_conversation(user: CurrentUser, db: DB, data: Optional[ConversationCreate] = None):
    return await conversation_service.create(db, user.id, data or ConversationCreate())


@router.get("", response_model=Page[ConversationResponse], responses=error_responses(401, 422))
async def list_conversations(
    user: CurrentUser,
    db: DB,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    q: Optional[str] = Query(None, max_length=200, description="Filter by title"),
    archived: Literal["exclude", "only", "include"] = "exclude",
    sort: Literal["updated_at", "created_at", "title"] = "updated_at",
    order: Literal["asc", "desc"] = "desc",
):
    items, total = await conversation_service.list_for_user(
        db, user.id, limit=limit, offset=offset, q=q, archived=archived, sort=sort, order=order
    )
    return Page(items=items, total=total, limit=limit, offset=offset)


@router.get("/{conversation_id}", response_model=ConversationResponse, responses=error_responses(401, 404))
async def get_conversation(conversation_id: uuid.UUID, user: CurrentUser, db: DB):
    return await conversation_service.get_owned(db, user.id, conversation_id)


@router.patch("/{conversation_id}", response_model=ConversationResponse, responses=error_responses(401, 404, 422))
async def update_conversation(conversation_id: uuid.UUID, data: ConversationUpdate, user: CurrentUser, db: DB):
    return await conversation_service.update(db, user.id, conversation_id, data)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT, responses=error_responses(401, 404))
async def delete_conversation(conversation_id: uuid.UUID, user: CurrentUser, db: DB):
    await conversation_service.delete_conversation(db, user.id, conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------- documents in a conversation

@router.get("/{conversation_id}/documents", response_model=List[DocumentResponse], responses=error_responses(401, 404))
async def list_conversation_documents(conversation_id: uuid.UUID, user: CurrentUser, db: DB):
    await conversation_service.get_owned(db, user.id, conversation_id)
    return await document_service.list_conversation_documents(db, user.id, conversation_id)


@router.post("/{conversation_id}/documents", response_model=List[DocumentResponse], responses=error_responses(401, 404, 422))
async def attach_documents(conversation_id: uuid.UUID, data: AttachDocumentsRequest, user: CurrentUser, db: DB):
    await document_service.attach_to_conversation(db, user.id, conversation_id, data.document_ids)
    return await document_service.list_conversation_documents(db, user.id, conversation_id)


@router.delete("/{conversation_id}/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT, responses=error_responses(401, 404))
async def detach_document(conversation_id: uuid.UUID, document_id: uuid.UUID, user: CurrentUser, db: DB):
    await document_service.detach_from_conversation(db, user.id, conversation_id, document_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------------------------------------------------------------- sharing

@router.post("/{conversation_id}/share", response_model=ShareCreated, status_code=status.HTTP_201_CREATED, responses=error_responses(401, 404, 429))
async def create_share(conversation_id: uuid.UUID, user: CurrentUser, db: DB, data: Optional[ShareCreate] = None):
    """Creates a public read-only link. Replaces any existing link for this conversation."""
    await hit_user_limit("share_create", user.id, times=30, seconds=3600)
    return await share_service.create(db, user.id, conversation_id, (data or ShareCreate()).expires_in_days)


@router.get("/{conversation_id}/share", response_model=ShareStatus, responses=error_responses(401, 404))
async def get_share_status(conversation_id: uuid.UUID, user: CurrentUser, db: DB):
    return await share_service.status(db, user.id, conversation_id)


@router.delete("/{conversation_id}/share", status_code=status.HTTP_204_NO_CONTENT, responses=error_responses(401, 404))
async def revoke_share(conversation_id: uuid.UUID, user: CurrentUser, db: DB):
    await share_service.revoke(db, user.id, conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
