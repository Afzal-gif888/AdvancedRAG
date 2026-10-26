import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, List, Optional, cast

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import AppError, NotFound
from app.core.time import utcnow
from app.db.models import Conversation, Document, Message, User
from app.db.models import message as msg_status
from app.db.models.conversation import DEFAULT_TITLE
from app.integrations.agent_core import AgentRequest
from app.schemas.message import MessageCreate, MessageResponse
from app.services import conversation_service, document_service

TITLE_MAX = 60


def to_response(m: Message) -> MessageResponse:
    meta = m.metadata_ or {}
    return MessageResponse(
        id=m.id,
        conversation_id=m.conversation_id,
        role=cast(Any, m.role),
        content=m.content,
        status=m.status,
        attachments=meta.get("attachments", []),
        citations=meta.get("citations", []),
        error_code=meta.get("error_code"),
        created_at=m.created_at,
    )


async def list_messages(db: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID, limit: int, offset: int, order: str = "asc"):
    """`order=desc` + offset pages backwards from the newest message (used for "load earlier")."""
    await conversation_service.get_owned(db, user_id, conversation_id)
    base = select(Message).where(Message.conversation_id == conversation_id)
    total = await db.scalar(select(func.count()).select_from(base.subquery()))
    ordering = (Message.created_at.desc(), Message.id.desc()) if order == "desc" else (Message.created_at.asc(), Message.id)
    rows = await db.scalars(base.order_by(*ordering).limit(limit).offset(offset))
    return [to_response(m) for m in rows.all()], total or 0


async def last_message_time(db: AsyncSession, conversation_id: uuid.UUID):
    return await db.scalar(select(func.max(Message.created_at)).where(Message.conversation_id == conversation_id))


async def next_message_time(db: AsyncSession, conversation_id: uuid.UUID):
    """Strictly increasing per conversation, so ordering, history cut-offs and share
    snapshots never depend on clock resolution (callers hold the conversation row lock)."""
    now = utcnow()
    last = await last_message_time(db, conversation_id)
    return max(now, last + timedelta(milliseconds=1)) if last else now


@dataclass
class PreparedTurn:
    user_message: Message
    assistant_message: Message
    agent_request: AgentRequest


async def _history(db: AsyncSession, conversation_id: uuid.UUID, before) -> List[dict]:
    if settings.CHAT_HISTORY_CONTEXT_MESSAGES == 0:
        return []
    rows = await db.scalars(
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
            Message.status == msg_status.COMPLETED,
            Message.created_at < before,
        )
        .order_by(Message.created_at.desc())
        .limit(settings.CHAT_HISTORY_CONTEXT_MESSAGES)
    )
    return [_history_item(m) for m in reversed(rows.all())]


def _history_item(m: Message) -> dict:
    """One earlier message for Agent Core. Assistant turns also carry the citations they showed and,
    for exact table analysis, the computed results stored with them, so a follow-up such as "give
    the above as a table" can reuse them (same conversation only; loaded for its owner)."""
    item = {"role": m.role, "content": m.content}
    if m.role == msg_status.ASSISTANT:
        meta = m.metadata_ or {}
        if meta.get("citations"):
            item["citations"] = meta["citations"]
        if isinstance(meta.get("structured_result"), dict):
            item["structured"] = meta["structured_result"]
    return item


async def _knowledge_base(db: AsyncSession, user: User) -> List[dict]:
    """Every ready document the user owns: the Knowledge Base searched in every conversation.
    Ready means RAG Core finished indexing and verified it; nothing has to be attached."""
    if not settings.LIBRARY_DISCOVERY_ENABLED:
        return []
    docs = await document_service.list_library_documents(db, user.id, (), settings.LIBRARY_DISCOVERY_MAX_DOCUMENTS)
    return [{"document_id": str(d.id), "filename": d.filename} for d in docs]


async def _build_agent_request(db: AsyncSession, user: User, conv: Conversation, user_msg: Message, assistant_msg: Message) -> AgentRequest:
    # Retrieval scope is the user's Knowledge Base, not this conversation's attachments: the
    # same documents are available in every chat, new or old (RAG Core re-checks ownership).
    return AgentRequest(
        user_id=user.id,
        conversation_id=conv.id,
        message_id=assistant_msg.id,
        message=user_msg.content,
        document_ids=[],
        history=await _history(db, conv.id, user_msg.created_at),
        context={"pending_document_count": await document_service.count_pending_documents(db, user.id)},
        library_documents=await _knowledge_base(db, user),
    )


async def prepare_turn(db: AsyncSession, user: User, conversation_id: uuid.UUID, data: MessageCreate) -> PreparedTurn:
    """Atomically persist the user message + a pending assistant placeholder."""
    conv = await conversation_service.get_owned(db, user.id, conversation_id, for_update=True)

    attachments: List[Document] = []
    if data.document_ids:
        # Raises 404 if any id is not owned by this user.
        attachments = await document_service.attach_to_conversation(db, user.id, conv.id, data.document_ids, commit=False)

    now = await next_message_time(db, conv.id)
    user_msg = Message(
        conversation_id=conv.id,
        user_id=user.id,
        role=msg_status.USER,
        content=data.content,
        status=msg_status.COMPLETED,
        metadata_={"attachments": [{"id": str(d.id), "filename": d.filename} for d in attachments]} if attachments else None,
        created_at=now,
    )
    # Strictly after the user message, even on coarse clocks.
    assistant_msg = Message(
        conversation_id=conv.id,
        user_id=user.id,
        role=msg_status.ASSISTANT,
        content="",
        status=msg_status.PENDING,
        created_at=now + timedelta(milliseconds=1),
    )
    db.add_all([user_msg, assistant_msg])

    if conv.title == DEFAULT_TITLE:
        first_line = " ".join(data.content.split())
        conv.title = (first_line[:TITLE_MAX] + "…") if len(first_line) > TITLE_MAX else first_line
    conv.updated_at = now
    await db.flush()

    request = await _build_agent_request(db, user, conv, user_msg, assistant_msg)
    await db.commit()
    return PreparedTurn(user_msg, assistant_msg, request)


async def prepare_retry(db: AsyncSession, user: User, conversation_id: uuid.UUID, message_id: uuid.UUID) -> PreparedTurn:
    """Re-run generation for a failed/cancelled assistant message (the latest one only)."""
    conv = await conversation_service.get_owned(db, user.id, conversation_id, for_update=True)
    assistant_msg = await db.scalar(
        select(Message).where(Message.id == message_id, Message.conversation_id == conv.id, Message.role == msg_status.ASSISTANT)
    )
    if assistant_msg is None:
        raise NotFound("Message not found.", code="MESSAGE_NOT_FOUND")
    if assistant_msg.status not in (msg_status.FAILED, msg_status.CANCELLED):
        raise AppError("Only failed responses can be retried.", code="MESSAGE_NOT_RETRYABLE", status_code=409)
    latest = await db.scalar(
        select(Message.id).where(Message.conversation_id == conv.id).order_by(Message.created_at.desc()).limit(1)
    )
    if latest != assistant_msg.id:
        raise AppError("Only the latest response can be retried.", code="MESSAGE_NOT_RETRYABLE", status_code=409)
    user_msg = await db.scalar(
        select(Message)
        .where(Message.conversation_id == conv.id, Message.role == msg_status.USER, Message.created_at < assistant_msg.created_at)
        .order_by(Message.created_at.desc())
        .limit(1)
    )
    if user_msg is None:
        raise AppError("Nothing to retry.", code="MESSAGE_NOT_RETRYABLE", status_code=409)

    assistant_msg.status = msg_status.PENDING
    assistant_msg.content = ""
    assistant_msg.metadata_ = None
    conv.updated_at = utcnow()
    request = await _build_agent_request(db, user, conv, user_msg, assistant_msg)
    await db.commit()
    return PreparedTurn(user_msg, assistant_msg, request)


async def finalize_assistant(
    db: AsyncSession,
    message_id: uuid.UUID,
    *,
    status: str,
    content: str = "",
    citations: Optional[list] = None,
    metadata: Optional[dict] = None,
    error_code: Optional[str] = None,
) -> Optional[Message]:
    msg = await db.get(Message, message_id)
    if msg is None:  # conversation deleted mid-generation
        return None
    meta = dict(metadata or {})
    if citations:
        meta["citations"] = citations
    if error_code:
        meta["error_code"] = error_code
    msg.status = status
    msg.content = content
    msg.metadata_ = meta or None
    await db.commit()
    return msg


async def update_streaming(db: AsyncSession, message_id: uuid.UUID, content: str) -> None:
    """Persist partial output while generating (only while still pending/streaming)."""
    msg = await db.get(Message, message_id)
    if msg is not None and msg.status in (msg_status.PENDING, msg_status.STREAMING):
        msg.status = msg_status.STREAMING
        msg.content = content
        await db.commit()


async def mark_stale_generations(db: AsyncSession, older_than_seconds: int, limit: int = 500) -> int:
    """Crash recovery: generations stuck in pending/streaming (process died, deploy, lost
    connection) become `failed` with INTERRUPTED. Flushed partial content is kept; the user
    can retry from the UI."""
    cutoff = utcnow() - timedelta(seconds=older_than_seconds)
    stale = (
        await db.scalars(
            select(Message)
            .where(
                Message.role == msg_status.ASSISTANT,
                Message.status.in_([msg_status.PENDING, msg_status.STREAMING]),
                Message.updated_at < cutoff,
            )
            .limit(limit)
        )
    ).all()
    for msg in stale:
        msg.status = msg_status.FAILED
        msg.metadata_ = {**(msg.metadata_ or {}), "error_code": "INTERRUPTED"}
    await db.commit()
    return len(stale)
