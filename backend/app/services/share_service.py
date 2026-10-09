import uuid
from datetime import timedelta
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import security
from app.core.config import settings
from app.core.exceptions import NotFound
from app.core.time import utcnow
from app.db.models import Conversation, Message, Share
from app.db.models import message as msg_status
from app.schemas.share import ShareCreated, SharedConversation, SharedMessage, ShareStatus
from app.services import conversation_service, message_service


class ShareNotFound(NotFound):
    code = "SHARE_NOT_FOUND"
    message = "This shared link is invalid or no longer available."


def _is_live(share: Share, now) -> bool:
    return share.is_active and (share.expires_at is None or share.expires_at > now)


async def _deactivate(db: AsyncSession, conversation_id: uuid.UUID, user_id: uuid.UUID) -> int:
    result = await db.execute(
        update(Share)
        .where(Share.conversation_id == conversation_id, Share.owner_user_id == user_id, Share.is_active.is_(True))
        .values(is_active=False, revoked_at=utcnow())
    )
    return result.rowcount


async def create(db: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID, expires_in_days: Optional[int]) -> ShareCreated:
    await conversation_service.get_owned(db, user_id, conversation_id, for_update=True)
    # One live link per conversation; issuing a new one replaces (revokes) the old one.
    await _deactivate(db, conversation_id, user_id)
    raw = security.generate_token()
    now = utcnow()
    # Snapshot = newest existing message; anything sent later gets a strictly greater timestamp.
    last = await message_service.last_message_time(db, conversation_id)
    share = Share(
        conversation_id=conversation_id,
        owner_user_id=user_id,
        share_token_hash=security.hash_token(raw),
        is_active=True,
        snapshot_at=last or now,
        created_at=now,
        expires_at=now + timedelta(days=expires_in_days) if expires_in_days else None,
    )
    db.add(share)
    await db.commit()
    return ShareCreated(token=raw, url=f"{settings.frontend_url}/share/{raw}", created_at=share.created_at, expires_at=share.expires_at)


async def status(db: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID) -> ShareStatus:
    await conversation_service.get_owned(db, user_id, conversation_id)
    share = await db.scalar(
        select(Share)
        .where(Share.conversation_id == conversation_id, Share.owner_user_id == user_id, Share.is_active.is_(True))
        .order_by(Share.created_at.desc())
        .limit(1)
    )
    if share is None or not _is_live(share, utcnow()):
        return ShareStatus(active=False)
    return ShareStatus(active=True, created_at=share.created_at, expires_at=share.expires_at)


async def revoke(db: AsyncSession, user_id: uuid.UUID, conversation_id: uuid.UUID) -> None:
    await conversation_service.get_owned(db, user_id, conversation_id)
    await _deactivate(db, conversation_id, user_id)
    await db.commit()


async def get_public(db: AsyncSession, raw_token: str) -> SharedConversation:
    if not raw_token or len(raw_token) > 200:
        raise ShareNotFound()
    row = (
        await db.execute(
            select(Share, Conversation)
            .join(Conversation, Conversation.id == Share.conversation_id)
            .where(Share.share_token_hash == security.hash_token(raw_token))
        )
    ).first()
    # Revoked, expired and unknown tokens are indistinguishable.
    if row is None or not _is_live(row[0], utcnow()):
        raise ShareNotFound()
    share, conv = row
    messages = await db.scalars(
        select(Message)
        .where(
            Message.conversation_id == conv.id,
            Message.status == msg_status.COMPLETED,
            Message.created_at <= share.snapshot_at,
        )
        .order_by(Message.created_at, Message.id)
        .limit(1000)
    )
    return SharedConversation(
        title=conv.title,
        shared_at=share.created_at,
        messages=[SharedMessage(role=m.role, content=m.content, created_at=m.created_at) for m in messages.all()],
    )
