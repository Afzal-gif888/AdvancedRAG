import uuid
from typing import Literal, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFound
from app.core.time import utcnow
from app.db.models import Conversation
from app.db.models.conversation import DEFAULT_TITLE
from app.schemas.conversation import ConversationCreate, ConversationUpdate

ArchivedFilter = Literal["exclude", "only", "include"]
SortField = Literal["updated_at", "created_at", "title"]


class ConversationNotFound(NotFound):
    code = "CONVERSATION_NOT_FOUND"
    message = "Conversation not found."


def escape_like(term: str) -> str:
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


async def get_owned(db: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID, *, for_update: bool = False) -> Conversation:
    """The single ownership gate for conversations. Another user's conversation is
    indistinguishable from a missing one (404)."""
    stmt = select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
    if for_update:
        stmt = stmt.with_for_update()
    conv = await db.scalar(stmt)
    if conv is None:
        raise ConversationNotFound()
    return conv


async def create(db: AsyncSession, user_id: uuid.UUID, data: ConversationCreate) -> Conversation:
    conv = Conversation(user_id=user_id, title=data.title or DEFAULT_TITLE)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return conv


async def list_for_user(
    db: AsyncSession,
    user_id: uuid.UUID,
    *,
    limit: int,
    offset: int,
    q: Optional[str] = None,
    archived: ArchivedFilter = "exclude",
    sort: SortField = "updated_at",
    order: Literal["asc", "desc"] = "desc",
):
    filters = [Conversation.user_id == user_id]
    if archived == "exclude":
        filters.append(Conversation.archived_at.is_(None))
    elif archived == "only":
        filters.append(Conversation.archived_at.is_not(None))
    if q:
        filters.append(Conversation.title.ilike(f"%{escape_like(q)}%", escape="\\"))

    column = getattr(Conversation, sort)
    ordering = column.desc() if order == "desc" else column.asc()
    total = await db.scalar(select(func.count()).select_from(Conversation).where(*filters))
    rows = await db.scalars(select(Conversation).where(*filters).order_by(ordering, Conversation.id).limit(limit).offset(offset))
    return rows.all(), total or 0


async def update(db: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID, data: ConversationUpdate) -> Conversation:
    conv = await get_owned(db, user_id, conversation_id)
    if data.title is not None:
        conv.title = data.title
        conv.updated_at = utcnow()
    if data.archived is not None:
        conv.archived_at = (conv.archived_at or utcnow()) if data.archived else None
    await db.commit()
    await db.refresh(conv)
    return conv


async def delete_conversation(db: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
    conv = await get_owned(db, user_id, conversation_id)
    # Messages, shares and document links cascade at the database level.
    await db.delete(conv)
    await db.commit()


async def delete_all_for_user(db: AsyncSession, user_id: uuid.UUID) -> None:
    await db.execute(delete(Conversation).where(Conversation.user_id == user_id))
