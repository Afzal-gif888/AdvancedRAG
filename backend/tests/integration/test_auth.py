import uuid
from datetime import timedelta

import jwt
from sqlalchemy import select

from app.core.config import settings
from app.core.time import utcnow
from app.db import session as db_session
from app.db.models import User, UserSession
from conftest import OUTBOX, PASSWORD, login, register_and_login

API = "/api/v1"


async def _register(client, email, password=PASSWORD, name="Alice"):
    return await client.post(f"{API}/auth/register", json={"email": email, "password": password, "full_name": name})


# ---------------------------------------------------------------- sign-up (name + email + password)

async def test_register_creates_account_and_signs_in(client):
    r = await _register(client, "  Alice@Example.COM ")
    assert r.status_code == 201
    body = r.json()
    assert body["token_type"] == "bearer" and body["access_token"] and body["expires_in"] == settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    assert body["user"]["email"] == "alice@example.com" and body["user"]["full_name"] == "Alice"
    cookie = r.headers["set-cookie"].lower()
    assert "refresh_token=" in cookie and "httponly" in cookie and "path=/api/v1/auth" in cookie and "samesite=lax" in cookie
    assert (await client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {body['access_token']}"})).status_code == 200
    assert OUTBOX.sent == []  # no email is involved in sign-up

    async with db_session.session_scope() as db:
        user = await db.scalar(select(User).where(User.email == "alice@example.com"))
    assert user.password_hash.startswith("$argon2id$") and PASSWORD not in user.password_hash
    assert user.role == "user"


async def test_register_ignores_client_supplied_role(client):
    r = await client.post(f"{API}/auth/register", json={"email": "r@example.com", "password": PASSWORD, "full_name": "R", "role": "admin", "is_verified": True})
    assert r.status_code == 201
    async with db_session.session_scope() as db:
        user = await db.scalar(select(User).where(User.email == "r@example.com"))
    assert user.role == "user"


async def test_duplicate_email_is_rejected_case_insensitively_and_never_overwritten(client):
    await register_and_login(client, "dup@example.com")
    r = await _register(client, "DUP@example.com", password="Other-pass9", name="Intruder")
    assert r.status_code == 409 and r.json()["error"]["code"] == "EMAIL_ALREADY_REGISTERED"
    assert await login(client, "dup@example.com")  # the original password still works
    assert (await client.post(f"{API}/auth/login", json={"email": "dup@example.com", "password": "Other-pass9"})).status_code == 401


async def test_weak_password_rejected_without_echoing_it(client):
    r = await client.post(f"{API}/auth/register", json={"email": "w@example.com", "password": "abcdefgh", "full_name": "Test User"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "abcdefgh" not in r.text


async def test_name_is_required(client):
    for name in (None, "", "   "):
        body = {"email": "noname@example.com", "password": PASSWORD}
        if name is not None:
            body["full_name"] = name
        r = await client.post(f"{API}/auth/register", json=body)
        assert r.status_code == 422, name


async def test_invalid_email_rejected(client):
    r = await client.post(f"{API}/auth/register", json={"email": "not-an-email", "password": PASSWORD, "full_name": "Test User"})
    assert r.status_code == 422


# ---------------------------------------------------------------- login (email + password)

async def test_login_with_password_starts_a_session(client):
    await register_and_login(client, "me@example.com")
    client.cookies.clear()
    r = await client.post(f"{API}/auth/login", json={"email": "ME@example.com", "password": PASSWORD})
    assert r.status_code == 200 and r.json()["access_token"]
    assert "refresh_token=" in r.headers["set-cookie"]
    me = await client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"})
    assert me.status_code == 200 and me.json()["email"] == "me@example.com"
    async with db_session.session_scope() as db:
        user = await db.scalar(select(User).where(User.email == "me@example.com"))
    assert user.last_login_at is not None
    assert OUTBOX.sent == []


async def test_invalid_credentials_are_generic(client):
    await register_and_login(client, "known@example.com")
    wrong_pw = await client.post(f"{API}/auth/login", json={"email": "known@example.com", "password": "Wrong12345"})
    unknown = await client.post(f"{API}/auth/login", json={"email": "unknown@example.com", "password": "Wrong12345"})
    assert wrong_pw.status_code == unknown.status_code == 401
    assert wrong_pw.json()["error"]["message"] == unknown.json()["error"]["message"] == "Invalid email or password."
    assert wrong_pw.json()["error"]["code"] == unknown.json()["error"]["code"] == "INVALID_CREDENTIALS"
    assert "set-cookie" not in wrong_pw.headers


async def test_otp_endpoints_are_gone(client):
    for path in ("/auth/verify-otp", "/auth/resend-otp"):
        r = await client.post(f"{API}{path}", json={"challenge_id": str(uuid.uuid4()), "code": "123456"})
        assert r.status_code in (404, 405), path


async def test_change_password_endpoint_is_gone(client, auth):
    r = await client.post(f"{API}/profile/change-password", json={"current_password": PASSWORD, "new_password": "NewPassw0rd"}, headers=auth)
    assert r.status_code in (404, 405)


async def test_protected_endpoints_require_auth(client):
    for method, path in [("GET", "/auth/me"), ("GET", "/conversations"), ("GET", "/documents"), ("GET", "/profile"), ("GET", "/settings")]:
        r = await client.request(method, f"{API}{path}")
        assert r.status_code == 401, path
        assert r.json()["error"]["code"] == "UNAUTHORIZED"


async def test_expired_access_token_rejected(client):
    await register_and_login(client, "exp@example.com")
    async with db_session.session_scope() as db:
        s = await db.scalar(select(UserSession))
    token = jwt.encode(
        {"sub": str(s.user_id), "sid": str(s.id), "typ": "access", "iat": utcnow() - timedelta(hours=2), "exp": utcnow() - timedelta(hours=1)},
        settings.JWT_SECRET.get_secret_value(),
        algorithm="HS256",
    )
    r = await client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


async def test_refresh_rotates_token_and_detects_reuse(client, monkeypatch):
    monkeypatch.setattr(settings, "REFRESH_REUSE_GRACE_SECONDS", 0)  # grace window covered in test_refresh_rotation.py
    await register_and_login(client, "rot@example.com")
    first = client.cookies.get("refresh_token")

    r = await client.post(f"{API}/auth/refresh")
    assert r.status_code == 200
    second = client.cookies.get("refresh_token")
    assert second and second != first
    assert (await client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"})).status_code == 200

    # Replaying the superseded token is treated as theft: the whole session dies.
    client.cookies.clear()
    client.cookies.set("refresh_token", first, path="/api/v1/auth")
    r = await client.post(f"{API}/auth/refresh")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "REFRESH_TOKEN_REUSED"

    client.cookies.clear()
    client.cookies.set("refresh_token", second, path="/api/v1/auth")
    r = await client.post(f"{API}/auth/refresh")
    assert r.status_code == 401


async def test_refresh_without_cookie_fails(client):
    r = await client.post(f"{API}/auth/refresh")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "SESSION_EXPIRED"


async def test_logout_revokes_session_server_side(client):
    headers = await register_and_login(client, "out@example.com")
    refresh_cookie = client.cookies.get("refresh_token")
    r = await client.post(f"{API}/auth/logout")
    assert r.status_code == 204
    # The still-unexpired access token no longer works.
    assert (await client.get(f"{API}/auth/me", headers=headers)).status_code == 401
    client.cookies.set("refresh_token", refresh_cookie, path="/api/v1/auth")
    assert (await client.post(f"{API}/auth/refresh")).status_code == 401


async def test_logout_all_revokes_every_device(make_client):
    a, b = make_client(), make_client()
    email = "multi@example.com"
    headers_a = await register_and_login(a, email)
    headers_b = await login(b, email)

    assert (await a.post(f"{API}/auth/logout-all", headers=headers_a)).status_code == 204
    assert (await a.get(f"{API}/auth/me", headers=headers_a)).status_code == 401
    assert (await b.get(f"{API}/auth/me", headers=headers_b)).status_code == 401
    assert (await b.post(f"{API}/auth/refresh")).status_code == 401


async def test_forgot_and_reset_password_flow(client, outbox):
    headers = await register_and_login(client, "reset@example.com")
    unknown = await client.post(f"{API}/auth/forgot-password", json={"email": "nobody@example.com"})
    known = await client.post(f"{API}/auth/forgot-password", json={"email": "reset@example.com"})
    assert unknown.status_code == known.status_code == 200
    assert unknown.json() == known.json()  # no account enumeration
    reset_mails = [m for m in outbox.sent if "Reset" in m.subject]
    assert len(reset_mails) == 1 and reset_mails[0].recipient == "reset@example.com"
    token = outbox.last_token()

    r = await client.post(f"{API}/auth/reset-password", json={"token": token, "new_password": "Brand-new-pw1"})
    assert r.status_code == 200
    # Single use.
    r = await client.post(f"{API}/auth/reset-password", json={"token": token, "new_password": "Another-pw12"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_TOKEN"
    # Existing sessions are revoked; the new password works.
    assert (await client.get(f"{API}/auth/me", headers=headers)).status_code == 401
    assert (await login(client, "reset@example.com", "Brand-new-pw1"))


async def test_reset_token_expires(client, outbox, monkeypatch):
    await client.post(f"{API}/auth/register", json={"email": "slow@example.com", "password": PASSWORD, "full_name": "Test User"})
    await client.post(f"{API}/auth/forgot-password", json={"email": "slow@example.com"})
    token = outbox.last_token()
    from app.db.models import AuthToken

    async with db_session.session_scope() as db:
        for t in (await db.scalars(select(AuthToken))).all():
            t.expires_at = utcnow() - timedelta(minutes=1)
        await db.commit()
    r = await client.post(f"{API}/auth/reset-password", json={"token": token, "new_password": "Brand-new-pw1"})
    assert r.status_code == 400


async def test_new_reset_request_invalidates_previous_token(client, outbox):
    await client.post(f"{API}/auth/register", json={"email": "twice@example.com", "password": PASSWORD, "full_name": "Test User"})
    await client.post(f"{API}/auth/forgot-password", json={"email": "twice@example.com"})
    first = outbox.last_token()
    await client.post(f"{API}/auth/forgot-password", json={"email": "twice@example.com"})
    r = await client.post(f"{API}/auth/reset-password", json={"token": first, "new_password": "Brand-new-pw1"})
    assert r.status_code == 400


async def test_refresh_tokens_never_stored_in_plaintext(client):
    await register_and_login(client, "plain@example.com")
    refresh = client.cookies.get("refresh_token")
    async with db_session.session_scope() as db:
        sessions = (await db.scalars(select(UserSession))).all()
    assert sessions and all(s.refresh_token_hash != refresh for s in sessions)


async def test_inactive_user_cannot_login(client):
    await client.post(f"{API}/auth/register", json={"email": "off@example.com", "password": PASSWORD, "full_name": "Test User"})
    async with db_session.session_scope() as db:
        user = await db.scalar(select(User).where(User.email == "off@example.com"))
        user.is_active = False
        await db.commit()
    r = await client.post(f"{API}/auth/login", json={"email": "off@example.com", "password": PASSWORD})
    assert r.status_code == 401 and r.json()["error"]["code"] == "INVALID_CREDENTIALS"


async def test_profile_update_only_allows_whitelisted_fields(client, auth):
    r = await client.patch(f"{API}/profile", json={"full_name": "  New Name "}, headers=auth)
    assert r.status_code == 200 and r.json()["full_name"] == "New Name"
    for forbidden in ({"email": "x@example.com"}, {"id": str(uuid.uuid4())}, {"role": "admin"}, {"password_hash": "x"}):
        r = await client.patch(f"{API}/profile", json=forbidden, headers=auth)
        assert r.status_code == 422, forbidden
