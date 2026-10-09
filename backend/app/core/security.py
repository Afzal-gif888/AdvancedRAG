import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.core.config import settings
from app.core.time import utcnow

# argon2-cffi's PasswordHasher uses Argon2id with RFC 9106 "low memory" parameters by default.
_hasher = PasswordHasher()
# Used to equalise timing when the account does not exist.
_DUMMY_HASH = _hasher.hash(secrets.token_urlsafe(16))

ACCESS = "access"
REFRESH = "refresh"


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: Optional[str]) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def password_needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


def burn_password_check(password: str) -> None:
    """Spend the same time as a real verification (used when the user does not exist)."""
    verify_password(password, None)


def generate_token(nbytes: int = 32) -> str:
    return secrets.token_urlsafe(nbytes)


def hash_token(token: str) -> str:
    """Deterministic hash for high-entropy opaque tokens (share/reset/verify/refresh).

    These tokens carry 256 bits of randomness, so a fast unsalted hash is safe (no brute force
    is feasible) and lets us look them up by hash. Not keyed, so rotating JWT secrets does not
    invalidate share links.
    """
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass
class TokenClaims:
    user_id: uuid.UUID
    session_id: uuid.UUID
    jti: Optional[str]
    expires_at: datetime


def create_access_token(user_id: uuid.UUID, session_id: uuid.UUID) -> str:
    now = utcnow()
    payload = {
        "sub": str(user_id),
        "sid": str(session_id),
        "typ": ACCESS,
        "iat": now,
        "exp": now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, settings.JWT_SECRET.get_secret_value(), algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(user_id: uuid.UUID, session_id: uuid.UUID, expires_at: datetime) -> str:
    payload = {
        "sub": str(user_id),
        "sid": str(session_id),
        "typ": REFRESH,
        "jti": generate_token(16),
        "iat": utcnow(),
        "exp": expires_at,
    }
    return jwt.encode(payload, settings.JWT_REFRESH_SECRET.get_secret_value(), algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str, token_type: str) -> Optional[TokenClaims]:
    secret = (settings.JWT_SECRET if token_type == ACCESS else settings.JWT_REFRESH_SECRET).get_secret_value()
    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["sub", "sid", "typ", "exp", "iat"]},
        )
        if payload.get("typ") != token_type:
            return None
        return TokenClaims(
            user_id=uuid.UUID(payload["sub"]),
            session_id=uuid.UUID(payload["sid"]),
            jti=payload.get("jti"),
            expires_at=datetime.fromtimestamp(payload["exp"], tz=utcnow().tzinfo),
        )
    except (jwt.PyJWTError, ValueError, TypeError, KeyError):
        return None
