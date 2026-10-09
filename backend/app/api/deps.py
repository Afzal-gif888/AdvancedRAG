from dataclasses import dataclass
from typing import Annotated, Optional

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import security
from app.core.exceptions import Unauthorized
from app.core.time import utcnow
from app.db.models import User, UserSession
from app.db.session import get_db

bearer_scheme = HTTPBearer(auto_error=False, description="Access token from /auth/login or /auth/refresh")

DB = Annotated[AsyncSession, Depends(get_db)]


@dataclass
class AuthContext:
    user: User
    session: UserSession


async def get_auth_context(
    db: DB, credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> AuthContext:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise Unauthorized()
    claims = security.decode_token(credentials.credentials, security.ACCESS)
    if claims is None:
        raise Unauthorized("Invalid or expired access token.", code="INVALID_TOKEN")

    # The session check makes logout / logout-all / password change take effect immediately.
    row = (
        await db.execute(
            select(User, UserSession)
            .join(UserSession, UserSession.user_id == User.id)
            .where(User.id == claims.user_id, UserSession.id == claims.session_id)
        )
    ).first()
    if row is None:
        raise Unauthorized("Invalid or expired access token.", code="INVALID_TOKEN")
    user, session = row
    if not user.is_active or not session.is_active_at(utcnow()):
        raise Unauthorized("Your session has ended. Please sign in again.", code="SESSION_EXPIRED")
    return AuthContext(user=user, session=session)


async def get_current_user(ctx: AuthContext = Depends(get_auth_context)) -> User:
    return ctx.user


CurrentUser = Annotated[User, Depends(get_current_user)]
CurrentAuth = Annotated[AuthContext, Depends(get_auth_context)]
