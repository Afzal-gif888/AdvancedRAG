"""Rotating refresh tokens: concurrency grace window vs. replay detection."""
import asyncio
from datetime import timedelta

from sqlalchemy import select

from app.core.config import settings
from app.core.time import utcnow
from app.db import session as db_session
from app.db.models import UserSession
from conftest import PASSWORD, login, register_and_login

API = "/api/v1"


def _cookie(client):
    return client.cookies.get("refresh_token")


def _use(client, token):
    client.cookies.clear()
    client.cookies.set("refresh_token", token, path="/api/v1/auth")


async def test_normal_refresh_rotates(client):
    await register_and_login(client)
    first = _cookie(client)
    r = await client.post(f"{API}/auth/refresh")
    assert r.status_code == 200 and _cookie(client) != first
    assert "set-cookie" in r.headers


async def test_concurrent_refreshes_from_two_tabs_keep_the_session(make_client):
    a = make_client()
    await register_and_login(a, "tabs@example.com")
    token = _cookie(a)
    b = make_client()
    _use(b, token)  # second tab: same browser cookie

    ra, rb = await asyncio.gather(a.post(f"{API}/auth/refresh"), b.post(f"{API}/auth/refresh"))
    assert {ra.status_code, rb.status_code} == {200}
    # Exactly one response rotated the cookie; the other got an access token only.
    assert sum("set-cookie" in r.headers for r in (ra, rb)) == 1
    for r in (ra, rb):
        assert (await a.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {r.json()['access_token']}"})).status_code == 200
    # The rotated token keeps working afterwards.
    winner = ra if "set-cookie" in ra.headers else rb
    rotated = winner.cookies.get("refresh_token")
    _use(a, rotated)
    assert (await a.post(f"{API}/auth/refresh")).status_code == 200


async def test_reuse_after_grace_window_revokes_session(client):
    await register_and_login(client)
    old = _cookie(client)
    await client.post(f"{API}/auth/refresh")
    new = _cookie(client)
    async with db_session.session_scope() as db:
        s = await db.scalar(select(UserSession))
        s.rotated_at = utcnow() - timedelta(seconds=settings.REFRESH_REUSE_GRACE_SECONDS + 1)
        await db.commit()
    _use(client, old)
    r = await client.post(f"{API}/auth/refresh")
    assert r.status_code == 401 and r.json()["error"]["code"] == "REFRESH_TOKEN_REUSED"
    _use(client, new)
    assert (await client.post(f"{API}/auth/refresh")).status_code == 401  # whole session is dead


async def test_token_two_rotations_old_is_reuse_even_inside_grace(client):
    await register_and_login(client)
    gen0 = _cookie(client)
    await client.post(f"{API}/auth/refresh")
    await client.post(f"{API}/auth/refresh")
    _use(client, gen0)
    r = await client.post(f"{API}/auth/refresh")
    assert r.status_code == 401 and r.json()["error"]["code"] == "REFRESH_TOKEN_REUSED"


async def test_grace_response_never_extends_the_replay_window(client):
    await register_and_login(client)
    old = _cookie(client)
    await client.post(f"{API}/auth/refresh")
    _use(client, old)
    r = await client.post(f"{API}/auth/refresh")
    assert r.status_code == 200 and "set-cookie" not in r.headers


async def test_revoked_and_expired_sessions_cannot_refresh(client, make_client):
    await register_and_login(client, "rev@example.com")
    token = _cookie(client)
    await client.post(f"{API}/auth/logout")
    _use(client, token)
    r = await client.post(f"{API}/auth/refresh")
    assert r.status_code == 401 and r.json()["error"]["code"] == "SESSION_EXPIRED"

    other = make_client()
    await register_and_login(other, "exp@example.com")
    async with db_session.session_scope() as db:
        for s in (await db.scalars(select(UserSession))).all():
            s.expires_at = utcnow() - timedelta(seconds=1)
        await db.commit()
    assert (await other.post(f"{API}/auth/refresh")).status_code == 401


async def test_logout_all_blocks_refresh_on_every_device(make_client):
    a, b = make_client(), make_client()
    headers = await register_and_login(a, "all@example.com")
    await login(b, "all@example.com")
    assert (await a.post(f"{API}/auth/logout-all", headers=headers)).status_code == 204
    assert (await a.post(f"{API}/auth/refresh")).status_code == 401
    assert (await b.post(f"{API}/auth/refresh")).status_code == 401
