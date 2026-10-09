from fastapi import APIRouter, Response, status

from app.api.deps import DB, CurrentUser
from app.schemas.common import error_responses
from app.schemas.user import ProfileUpdate, UserResponse
from app.services import user_service

# Passwords cannot be changed from the profile. The only way to set a new password is the
# emailed reset link (/auth/forgot-password), which also signs out every session.
router = APIRouter()

@router.get("", response_model=UserResponse, responses=error_responses(401))
async def read_profile(user: CurrentUser):
    return user


@router.patch("", response_model=UserResponse, responses=error_responses(401, 422))
async def update_profile(data: ProfileUpdate, user: CurrentUser, db: DB):
    return await user_service.update_profile(db, user, data)


@router.delete("/data", status_code=status.HTTP_204_NO_CONTENT, responses=error_responses(401))
async def clear_all_data(user: CurrentUser, db: DB):
    """Deletes all of the user's conversations and documents. The account itself is kept."""
    await user_service.clear_all_data(db, user)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
