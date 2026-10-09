from fastapi import APIRouter, Depends

from app.api.deps import DB
from app.middleware.rate_limit import RateLimit
from app.schemas.common import error_responses
from app.schemas.share import SharedConversation
from app.services import share_service

router = APIRouter() 


@router.get(

    "/{share_token}",
    response_model=SharedConversation,
    responses=error_responses(404, 429),
    dependencies=[Depends(RateLimit("share_view", times=120, seconds=60))],
)
async def view_shared_conversation(share_token: str, db: DB):
    """Public, unauthenticated. Returns only the shared conversation's title and messages
    as of the moment the link was created."""
    return await share_service.get_public(db, share_token)

