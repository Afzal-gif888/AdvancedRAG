import uuid
from typing import Dict, List, Tuple

from sqlalchemy import exists, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Conversation, Message
from app.schemas.conversation import ConversationSearchResult
from app.services.conversation_service import escape_like

SNIPPET_RADIUS = 60


def _snippet(content: str, term: str) -> str:
    idx = content.lower().find(term.lower())
    if idx < 0:
        return content[: SNIPPET_RADIUS * 2]
    start, end = max(0, idx - SNIPPET_RADIUS), min(len(content), idx + len(term) + SNIPPET_RADIUS)
    return ("…" if start else "") + content[start:end].replace("\n", " ") + ("…" if end < len(content) else "")


async def search_chats(db: AsyncSession, user_id: uuid.UUID, q: str, limit: int, offset: int) -> Tuple[List[ConversationSearchResult], int]:
    """Case-insensitive match on titles and message content, always scoped to the caller.

    Bounded by the user's own data (indexed on user_id). For very large histories, swap the
    ILIKE for PostgreSQL full-text search (tsvector + GIN) behind this same function.
    """
    pattern = f"%{escape_like(q)}%"
    message_match = exists().where(
        Message.conversation_id == Conversation.id,
        Message.content.ilike(pattern, escape="\\"),
    )
    filters = [Conversation.user_id == user_id, or_(Conversation.title.ilike(pattern, escape="\\"), message_match)]

    total = await db.scalar(select(func.count()).select_from(Conversation).where(*filters)) or 0
    convs = (
        await db.scalars(
            select(Conversation).where(*filters).order_by(Conversation.updated_at.desc(), Conversation.id).limit(limit).offset(offset)
        )
    ).all()

    snippets: Dict[uuid.UUID, str] = {}
    if convs:
        rows = await db.execute(
            select(Message.conversation_id, Message.content)
            .where(Message.conversation_id.in_([c.id for c in convs]), Message.content.ilike(pattern, escape="\\"))
            .order_by(Message.created_at)
            .limit(limit * 5)
        )
        for conv_id, content in rows.all():
            snippets.setdefault(conv_id, _snippet(content, q))

    results = [
        ConversationSearchResult.model_validate(c, from_attributes=True).model_copy(update={"snippet": snippets.get(c.id)})
        for c in convs
    ]
    return results, total
