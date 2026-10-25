import uuid
from typing import Literal

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import StreamingResponse

from app.api.deps import DB, CurrentUser
from app.core.exceptions import AppError
from app.integrations.agent_core import AgentCoreClient, get_agent_client
from app.middleware.rate_limit import hit_user_limit
from app.schemas.common import Page, error_responses
from app.schemas.message import MessageCreate, MessageResponse, SendMessageResponse
from app.services import agent_service, message_service, stream_limiter

router = APIRouter()

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


async def _limited(events, user_id, token):
    """Relay the stream and give the user's slot back however it ends (finished, failed, client gone)."""
    try:
        async for chunk in events:
            yield chunk
    finally:
        stream_limiter.release(user_id, token)


async def _message_rate_limit(user: CurrentUser) -> None:
    await hit_user_limit("messages", user.id, times=30, seconds=60)


@router.get("", response_model=Page[MessageResponse], responses=error_responses(401, 404))
async def list_messages(
    conversation_id: uuid.UUID,
    user: CurrentUser,
    db: DB,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    order: Literal["asc", "desc"] = "asc",
):
    items, total = await message_service.list_messages(db, user.id, conversation_id, limit, offset, order)
    return Page(items=items, total=total, limit=limit, offset=offset)


@router.post(
    "",
    response_model=SendMessageResponse,
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(401, 404, 422, 429, 502, 503),
    dependencies=[Depends(_message_rate_limit)],
)
async def send_message(
    conversation_id: uuid.UUID,
    data: MessageCreate,
    user: CurrentUser,
    db: DB,
    agent: AgentCoreClient = Depends(get_agent_client),
):
    """Non-streaming turn. On AI failure the user message is kept, the assistant message is
    stored as `failed`, and the error carries both ids so the client can offer a retry."""
    turn = await message_service.prepare_turn(db, user, conversation_id, data)
    try:
        assistant = await agent_service.run_turn(agent, turn)
    except AppError as exc:
        exc.details = {"user_message_id": str(turn.user_message.id), "assistant_message_id": str(turn.assistant_message.id)}
        raise
    return SendMessageResponse(
        user_message=message_service.to_response(turn.user_message),
        assistant_message=message_service.to_response(assistant),
    )


@router.post(
    "/stream",
    responses={200: {"content": {"text/event-stream": {}}}, **error_responses(401, 404, 422, 429)},
    dependencies=[Depends(_message_rate_limit)],
)
async def stream_message(
    conversation_id: uuid.UUID,
    data: MessageCreate,
    user: CurrentUser,
    db: DB,
    agent: AgentCoreClient = Depends(get_agent_client),
):
    """Server-Sent Events. Validation/ownership errors are normal JSON errors (before the
    stream starts); AI failures arrive as an `error` event."""
    token = await stream_limiter.acquire(user.id)  # before anything is persisted: a refused turn leaves no messages behind
    try:
        turn = await message_service.prepare_turn(db, user, conversation_id, data)
    except BaseException:
        stream_limiter.release(user.id, token)
        raise
    return StreamingResponse(_limited(agent_service.stream_turn(agent, turn), user.id, token), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post(
    "/{message_id}/retry",
    responses={200: {"content": {"text/event-stream": {}}}, **error_responses(401, 404, 409, 429)},
    dependencies=[Depends(_message_rate_limit)],
)
async def retry_message(
    conversation_id: uuid.UUID,
    message_id: uuid.UUID,
    user: CurrentUser,
    db: DB,
    agent: AgentCoreClient = Depends(get_agent_client),
):
    """Re-generates the latest failed/cancelled assistant message (SSE, same events as /stream)."""
    token = await stream_limiter.acquire(user.id)
    try:
        turn = await message_service.prepare_retry(db, user, conversation_id, message_id)
    except BaseException:
        stream_limiter.release(user.id, token)
        raise
    return StreamingResponse(_limited(agent_service.stream_turn(agent, turn), user.id, token), media_type="text/event-stream", headers=SSE_HEADERS)
