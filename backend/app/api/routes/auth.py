from typing import cast

from fastapi import APIRouter, BackgroundTasks, Depends, Request, Response, status

from app.api.deps import DB, CurrentAuth, CurrentUser
from app.core.config import settings
from app.core.exceptions import Unauthorized
from app.integrations.email import get_email_service
from app.middleware.error_handler import error_response
from app.middleware.rate_limit import RateLimit, client_ip, hit_identity_limit
from app.schemas.auth import (
    ForgotPasswordRequest,
    LoginRequest,
    MessageOut,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
)
from app.schemas.common import error_responses
from app.schemas.user import UserResponse
from app.services import auth_service

router = APIRouter()

COOKIE_PATH = f"{settings.API_V1_STR}/auth"
GENERIC_RESET_RESPONSE = "If an account exists for this email, a password reset link has been sent."


def set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        value=token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 3600,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.COOKIE_SAMESITE,
        domain=settings.COOKIE_DOMAIN,
        # Only sent to /auth endpoints, never to the rest of the API.
        path=COOKIE_PATH,
    )


def clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.REFRESH_COOKIE_NAME,
        path=COOKIE_PATH,
        domain=settings.COOKIE_DOMAIN,
        secure=settings.cookie_secure,
        httponly=True,
        samesite=settings.COOKIE_SAMESITE,
    )


def _token_response(issued: auth_service.IssuedSession) -> TokenResponse:
    return TokenResponse(
        access_token=issued.access_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.model_validate(issued.user),
    )


async def _start_session(db, user, request: Request, response: Response) -> TokenResponse:
    issued = await auth_service.issue_session(db, user, request.headers.get("user-agent"), client_ip(request))
    set_refresh_cookie(response, cast(str, issued.refresh_token))
    return _token_response(issued)


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    responses=error_responses(409, 422, 429),
    dependencies=[Depends(RateLimit("register", times=10, seconds=3600))],
)
async def register(data: RegisterRequest, request: Request, response: Response, db: DB):
    """Creates the account (name, email, password) and signs the user in."""
    await hit_identity_limit("register", data.email, times=5, seconds=3600)
    user = await auth_service.register(db, data)
    return await _start_session(db, user, request, response)


@router.post(
    "/login",
    response_model=TokenResponse,
    responses=error_responses(401, 422, 429),
    dependencies=[Depends(RateLimit("login", times=10, seconds=300))],
)
async def login(data: LoginRequest, request: Request, response: Response, db: DB):
    """Email + password. Returns an access token and sets the HttpOnly refresh cookie."""
    # Per-IP (dependency) plus per-account: stops one account being attacked from many IPs.
    await hit_identity_limit("login", data.email, times=10, seconds=900)
    user = await auth_service.authenticate(db, data.email, data.password)
    return await _start_session(db, user, request, response)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    responses=error_responses(401, 429),
    dependencies=[Depends(RateLimit("refresh", times=60, seconds=60))],
)
async def refresh(request: Request, response: Response, db: DB):
    """Reads the HttpOnly refresh cookie, rotates it, and returns a new access token."""
    raw = request.cookies.get(settings.REFRESH_COOKIE_NAME)
    try:
        issued = await auth_service.refresh(db, raw)
    except Unauthorized as exc:
        # Return (not raise) so the cookie deletion is part of the error response.
        failed = error_response(exc)
        clear_refresh_cookie(failed)
        return failed
    if issued.refresh_token:
        set_refresh_cookie(response, cast(str, issued.refresh_token))
    return _token_response(issued)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request, db: DB):
    """Revokes the current session server-side (identified by the refresh cookie) and clears it."""
    await auth_service.revoke_session_by_refresh_token(db, request.cookies.get(settings.REFRESH_COOKIE_NAME))
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    clear_refresh_cookie(response)
    return response


@router.post("/logout-all", status_code=status.HTTP_204_NO_CONTENT, responses=error_responses(401))
async def logout_all(auth: CurrentAuth, db: DB):
    await auth_service.revoke_all_sessions(db, auth.user.id, "logout_all")
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    clear_refresh_cookie(response)
    return response


@router.get("/me", response_model=UserResponse, responses=error_responses(401))
async def me(user: CurrentUser):
    return user


@router.post(
    "/forgot-password",
    response_model=MessageOut,
    dependencies=[Depends(RateLimit("forgot_password", times=5, seconds=3600))],
)
async def forgot_password(data: ForgotPasswordRequest, db: DB, background: BackgroundTasks):
    await hit_identity_limit("forgot_password", data.email, times=3, seconds=3600)
    result = await auth_service.request_password_reset(db, data.email)
    if result:
        # Sent after the response so timing does not reveal whether the account exists.
        background.add_task(auth_service.send_password_reset_email, get_email_service(), *result)
    return MessageOut(message=GENERIC_RESET_RESPONSE)


@router.post(
    "/reset-password",
    response_model=MessageOut,
    responses=error_responses(400, 422, 429),
    dependencies=[Depends(RateLimit("reset_password", times=10, seconds=3600))],
)
async def reset_password(data: ResetPasswordRequest, db: DB):
    await auth_service.reset_password(db, data.token, data.new_password)
    return MessageOut(message="Your password has been reset. Please sign in.")
