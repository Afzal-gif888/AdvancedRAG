from fastapi import APIRouter

from app.api.deps import DB, CurrentUser
from app.schemas.common import error_responses
from app.schemas.settings import SettingsUpdate, UserSettings
from app.services import user_service

router = APIRouter()


@router.get("", response_model=UserSettings, responses=error_responses(401))
async def read_settings(user: CurrentUser):
    return user_service.get_settings(user)


@router.patch("", response_model=UserSettings, responses=error_responses(401, 422))
async def update_settings(data: SettingsUpdate, user: CurrentUser, db: DB):
    return await user_service.update_settings(db, user, data)
