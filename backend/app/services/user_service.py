from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User
from app.schemas.settings import SettingsUpdate, UserSettings
from app.schemas.user import ProfileUpdate
from app.services import conversation_service, document_service, job_service


async def update_profile(db: AsyncSession, user: User, data: ProfileUpdate) -> User:
    if "full_name" in data.model_fields_set:
        user.full_name = data.full_name or None
    await db.commit()
    await db.refresh(user)
    return user


def get_settings(user: User) -> UserSettings:
    # Tolerates unknown/legacy keys in storage.
    return UserSettings.model_validate({k: v for k, v in (user.settings or {}).items() if k in UserSettings.model_fields})


async def update_settings(db: AsyncSession, user: User, data: SettingsUpdate) -> UserSettings:
    merged = get_settings(user).model_copy(update=data.model_dump(exclude_unset=True, exclude_none=True))
    user.settings = merged.model_dump()  # reassign so the JSON column is flagged dirty
    await db.commit()
    return merged


async def clear_all_data(db: AsyncSession, user: User) -> None:
    """Deletes every conversation (with messages/shares) and every document of the user."""
    await document_service.delete_all_for_user(db, user.id)
    await conversation_service.delete_all_for_user(db, user.id)
    await db.commit()
    job_service.notify_worker()
