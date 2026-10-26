import logging
import re
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import security
from app.core.config import settings
from app.core.context import get_request_id
from app.core.exceptions import AppError, Conflict, Unauthorized
from app.db.base import rowcount
from app.core.time import utcnow
from app.db.models import AuthToken, User, UserSession
from app.db.models.auth_token import PASSWORD_RESET
from app.db import session as db_session
from app.integrations.email import EmailService
from app.schemas.auth import RegisterRequest

logger = logging.getLogger("advancerag.auth")


class InvalidCredentials(Unauthorized):
    code = "INVALID_CREDENTIALS"
    message = "Invalid email or password."


class SessionExpired(Unauthorized):
    code = "SESSION_EXPIRED"
    message = "Your session has expired. Please sign in again."


class InvalidToken(AppError):
    status_code = 400
    code = "INVALID_TOKEN"
    message = "This link is invalid or has expired."


@dataclass
class IssuedSession:
    user: User
    session: UserSession
    access_token: str
    # None when served from the refresh grace window: the browser keeps its current cookie.
    refresh_token: Optional[str]


def _refresh_expiry():
    return utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)


async def register(db: AsyncSession, req: RegisterRequest) -> User:
    """Create the account. An email can be registered once; existing accounts are never overwritten."""
    if await db.scalar(select(User.id).where(User.email == req.email)) is not None:
        raise Conflict("An account with this email already exists.", code="EMAIL_ALREADY_REGISTERED")
    user = User(
        email=req.email,
        password_hash=security.hash_password(req.password),
        full_name=req.full_name,
        role="user",  # never client-controlled
        settings={},
    )
    db.add(user)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise Conflict("An account with this email already exists.", code="EMAIL_ALREADY_REGISTERED")
    await db.refresh(user)
    logger.info("user_registered", extra={"user_id": str(user.id)})
    return user


async def authenticate(db: AsyncSession, email: str, password: str) -> User:
    user = await db.scalar(select(User).where(User.email == email))
    if user is None:
        security.burn_password_check(password)  # equalise timing
        raise InvalidCredentials()
    if not security.verify_password(password, user.password_hash) or not user.is_active:
        raise InvalidCredentials()
    user.last_login_at = utcnow()
    if security.password_needs_rehash(user.password_hash):
        user.password_hash = security.hash_password(password)
    await db.commit()
    return user


async def issue_session(
    db: AsyncSession, user: User, user_agent: Optional[str], ip_address: Optional[str]
) -> IssuedSession:
    session_id = uuid.uuid4()
    expires_at = _refresh_expiry()
    refresh_token = security.create_refresh_token(user.id, session_id, expires_at)
    session = UserSession(
        id=session_id,
        user_id=user.id,
        refresh_token_hash=security.hash_token(refresh_token),
        expires_at=expires_at,
        last_used_at=utcnow(),
        user_agent=(user_agent or "")[:255] or None,
        ip_address=(ip_address or "")[:45] or None,
    )
    db.add(session)
    await db.commit()
    return IssuedSession(user, session, security.create_access_token(user.id, session_id), refresh_token)


async def refresh(db: AsyncSession, raw_refresh_token: Optional[str]) -> IssuedSession:
    """Rotate the refresh token. Presenting a superseded token revokes the session (reuse detection)."""
    if not raw_refresh_token:
        raise SessionExpired()
    claims = security.decode_token(raw_refresh_token, security.REFRESH)
    if claims is None:
        raise SessionExpired()

    session = await db.scalar(
        select(UserSession).where(UserSession.id == claims.session_id).with_for_update()
    )
    now = utcnow()
    if session is None or session.user_id != claims.user_id or not session.is_active_at(now):
        raise SessionExpired()

    presented = security.hash_token(raw_refresh_token)
    in_grace = (
        presented == session.previous_refresh_token_hash
        and session.rotated_at is not None
        and (now - session.rotated_at).total_seconds() <= settings.REFRESH_REUSE_GRACE_SECONDS
    )
    if presented != session.refresh_token_hash and not in_grace:
        # A validly signed token that was rotated away earlier: likely stolen. Kill the session.
        session.revoked_at = now
        session.revoked_reason = "refresh_token_reuse"
        await db.commit()
        logger.warning("refresh_token_reuse_detected", extra={"user_id": str(session.user_id), "session_id": str(session.id)})
        raise Unauthorized("Your session is no longer valid. Please sign in again.", code="REFRESH_TOKEN_REUSED")

    user = await db.get(User, session.user_id)
    if user is None or not user.is_active:
        raise SessionExpired()

    if in_grace:
        # Concurrent refresh (another tab of the same browser rotated a moment ago). Issue an
        # access token only: no new refresh token, so the replay window cannot be extended.
        session.last_used_at = now
        await db.commit()
        logger.info("refresh_grace_used", extra={"session_id": str(session.id)})
        return IssuedSession(user, session, security.create_access_token(user.id, session.id), None)

    expires_at = _refresh_expiry()
    new_refresh = security.create_refresh_token(user.id, session.id, expires_at)
    # Compare-and-set on the presented hash: of two concurrent refreshes exactly one rotates,
    # even where SELECT ... FOR UPDATE is unavailable. The loser takes the grace path.
    result = await db.execute(
        update(UserSession)
        .where(UserSession.id == session.id, UserSession.refresh_token_hash == presented)
        .values(
            previous_refresh_token_hash=presented,
            refresh_token_hash=security.hash_token(new_refresh),
            rotated_at=now,
            expires_at=expires_at,
            last_used_at=now,
        )
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    if rowcount(result) != 1:
        logger.info("refresh_grace_used", extra={"session_id": str(session.id), "reason": "concurrent_rotation"})
        return IssuedSession(user, session, security.create_access_token(user.id, session.id), None)
    return IssuedSession(user, session, security.create_access_token(user.id, session.id), new_refresh)


async def revoke_session_by_refresh_token(db: AsyncSession, raw_refresh_token: Optional[str]) -> None:
    claims = security.decode_token(raw_refresh_token, security.REFRESH) if raw_refresh_token else None
    if claims is None:
        return
    await revoke_session(db, claims.session_id, claims.user_id, "logout")


async def revoke_session(db: AsyncSession, session_id: uuid.UUID, user_id: uuid.UUID, reason: str) -> None:
    await db.execute(
        update(UserSession)
        .where(UserSession.id == session_id, UserSession.user_id == user_id, UserSession.revoked_at.is_(None))
        .values(revoked_at=utcnow(), revoked_reason=reason)
    )
    await db.commit()


async def revoke_all_sessions(db: AsyncSession, user_id: uuid.UUID, reason: str, commit: bool = True) -> None:
    await db.execute(
        update(UserSession)
        .where(UserSession.user_id == user_id, UserSession.revoked_at.is_(None))
        .values(revoked_at=utcnow(), revoked_reason=reason)
    )
    if commit:
        await db.commit()


# ---------------------------------------------------------------- single-use tokens

async def _create_auth_token(db: AsyncSession, user_id: uuid.UUID, purpose: str, ttl: timedelta) -> str:
    now = utcnow()
    # Invalidate earlier outstanding tokens of the same purpose.
    await db.execute(
        update(AuthToken)
        .where(AuthToken.user_id == user_id, AuthToken.purpose == purpose, AuthToken.used_at.is_(None))
        .values(used_at=now)
    )
    raw = security.generate_token()
    db.add(AuthToken(user_id=user_id, purpose=purpose, token_hash=security.hash_token(raw), expires_at=now + ttl))
    await db.commit()
    return raw


# security.generate_token() output: URL-safe base64. Anything else is rejected before any lookup,
# with the same INVALID_TOKEN answer as an unknown, used or expired token.
_TOKEN_FORMAT = re.compile(r"^[A-Za-z0-9_-]{32,128}$")


async def _consume_auth_token(db: AsyncSession, raw: str, purpose: str) -> User:
    if not _TOKEN_FORMAT.match(raw or ""):
        raise InvalidToken()
    now = utcnow()
    token = await db.scalar(
        select(AuthToken)
        .where(AuthToken.token_hash == security.hash_token(raw), AuthToken.purpose == purpose)
        .with_for_update()
    )
    if token is None or token.used_at is not None or token.expires_at <= now:
        raise InvalidToken()
    user = await db.get(User, token.user_id)
    if user is None or not user.is_active:
        raise InvalidToken()
    token.used_at = now
    return user


async def request_password_reset(db: AsyncSession, email: str) -> Optional[tuple]:
    """Returns (email, raw_token) when an email should be sent; None otherwise. Caller responds generically."""
    user = await db.scalar(select(User).where(User.email == email))
    if user is None or not user.is_active:
        return None
    raw = await _create_auth_token(
        db, user.id, PASSWORD_RESET, timedelta(minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)
    )
    return user.email, raw


async def reset_password(db: AsyncSession, raw_token: str, new_password: str) -> None:
    user = await _consume_auth_token(db, raw_token, PASSWORD_RESET)
    user.password_hash = security.hash_password(new_password)
    user.password_changed_at = utcnow()
    # Proving control of the mailbox also verifies it.
    user.is_verified = True
    await revoke_all_sessions(db, user.id, "password_reset", commit=False)
    await db.commit()
    logger.info("password_reset_completed", extra={"user_id": str(user.id)})


# ---------------------------------------------------------------- emails

async def send_password_reset_email(email_service: EmailService, email: str, raw_token: str) -> None:
    """Runs after the (generic) response so timing does not reveal whether the account exists.
    On delivery failure the token is voided (no live token for an email that never arrived) and a
    structured error is logged with the request id — never the token, the link or provider details."""
    reset_url = f"{settings.frontend_url}/reset-password?token={raw_token}"
    try:
        await email_service.send_password_reset_email(email, reset_url, settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)
    except Exception as exc:
        logger.error("password_reset_email_failed", extra={
            "request_id": get_request_id(),
            "provider": getattr(getattr(email_service, "provider", None), "name", type(email_service).__name__),
            "error_type": type(exc).__name__,
        })
        try:
            async with db_session.session_scope() as db:
                await db.execute(
                    update(AuthToken)
                    .where(AuthToken.token_hash == security.hash_token(raw_token), AuthToken.used_at.is_(None))
                    .values(used_at=utcnow())
                )
                await db.commit()
        except Exception as db_exc:  # pragma: no cover - logged; the token still expires on its own
            logger.error("password_reset_token_void_failed", extra={"error_type": type(db_exc).__name__})
        return
    logger.info("password_reset_email_sent", extra={"to_domain": email.split("@")[-1]})
