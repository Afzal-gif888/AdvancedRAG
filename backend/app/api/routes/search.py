from fastapi import APIRouter, Query

from app.api.deps import DB, CurrentUser
from app.schemas.common import Page, error_responses
from app.schemas.conversation import ConversationSearchResult
from app.services import search_service

router = APIRouter()


@router.get("/chats", response_model=Page[ConversationSearchResult], responses=error_responses(401, 422))
async def search_chats(
    user: CurrentUser,
    db: DB,
    q: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0, le=1000),
):
    """Searches the caller's conversation titles and message content."""
    items, total = await search_service.search_chats(db, user.id, q.strip() or q, limit, offset)
    return Page(items=items, total=total, limit=limit, offset=offset)
