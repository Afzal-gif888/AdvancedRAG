"""User A must never reach User B's data, and the API must fail safely."""
import uuid

import pytest

from app.core.config import settings
from conftest import PASSWORD, create_conversation, register_and_login, upload

API = "/api/v1"


@pytest.fixture
async def two_users(make_client, agent):
    a, b = make_client(), make_client()
    ha = await register_and_login(a, "alice@example.com")
    hb = await register_and_login(b, "bob@example.com")
    conv = await create_conversation(a, ha, "Alice private")
    await a.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "alice secret"}, headers=ha)
    doc = (await upload(a, ha, "alice.txt", b"alice data")).json()
    return {"a": a, "ha": ha, "b": b, "hb": hb, "conv": conv, "doc": doc}


async def test_cannot_read_or_modify_other_users_conversations(two_users):
    b, hb, cid = two_users["b"], two_users["hb"], two_users["conv"]["id"]
    for method, path, body in [
        ("GET", f"/conversations/{cid}", None),
        ("PATCH", f"/conversations/{cid}", {"title": "pwned"}),
        ("DELETE", f"/conversations/{cid}", None),
        ("GET", f"/conversations/{cid}/messages", None),
        ("POST", f"/conversations/{cid}/messages", {"content": "inject"}),
        ("POST", f"/conversations/{cid}/messages/stream", {"content": "inject"}),
        ("GET", f"/conversations/{cid}/documents", None),
        ("POST", f"/conversations/{cid}/share", None),
        ("GET", f"/conversations/{cid}/share", None),
        ("DELETE", f"/conversations/{cid}/share", None),
    ]:
        r = await b.request(method, f"{API}{path}", json=body, headers=hb)
        # 404, not 403: existence is not revealed.
        assert r.status_code == 404, (method, path, r.status_code)
        assert "alice" not in r.text.lower()

    assert (await b.get(f"{API}/conversations", headers=hb)).json()["total"] == 0
    assert (await b.get(f"{API}/search/chats", params={"q": "alice"}, headers=hb)).json()["total"] == 0
    # Still intact for the owner.
    a, ha = two_users["a"], two_users["ha"]
    assert (await a.get(f"{API}/conversations/{cid}", headers=ha)).json()["title"] == "Alice private"
    assert (await a.get(f"{API}/conversations/{cid}/messages", headers=ha)).json()["total"] == 2


async def test_cannot_access_or_attach_other_users_documents(two_users):
    b, hb, did = two_users["b"], two_users["hb"], two_users["doc"]["id"]
    for method, path in [
        ("GET", f"/documents/{did}"),
        ("GET", f"/documents/{did}/content"),
        ("DELETE", f"/documents/{did}"),
        ("POST", f"/documents/{did}/reprocess"),
    ]:
        assert (await b.request(method, f"{API}{path}", headers=hb)).status_code == 404, path
    assert (await b.get(f"{API}/documents", headers=hb)).json()["total"] == 0

    own_conv = await create_conversation(b, hb)
    r = await b.post(f"{API}/conversations/{own_conv['id']}/documents", json={"document_ids": [did]}, headers=hb)
    assert r.status_code == 404
    r = await b.post(f"{API}/conversations/{own_conv['id']}/messages/stream", json={"content": "x", "document_ids": [did]}, headers=hb)
    assert r.status_code == 404
    # Nothing was persisted from the rejected message.
    assert (await b.get(f"{API}/conversations/{own_conv['id']}/messages", headers=hb)).json()["total"] == 0
    a, ha = two_users["a"], two_users["ha"]
    assert (await a.get(f"{API}/documents/{did}", headers=ha)).status_code == 200


async def test_cannot_attach_own_document_to_other_users_conversation(two_users):
    b, hb = two_users["b"], two_users["hb"]
    bob_doc = (await upload(b, hb, "bob.txt")).json()
    r = await b.post(f"{API}/conversations/{two_users['conv']['id']}/documents", json={"document_ids": [bob_doc["id"]]}, headers=hb)
    assert r.status_code == 404


async def test_profile_is_always_the_callers_own(two_users):
    b, hb = two_users["b"], two_users["hb"]
    r = await b.patch(f"{API}/profile", json={"full_name": "Bob"}, headers=hb)
    assert r.json()["email"] == "bob@example.com"
    me_a = await two_users["a"].get(f"{API}/profile", headers=two_users["ha"])
    assert me_a.json()["full_name"] == "Test User"


async def test_forged_tokens_rejected(client, auth):
    import jwt

    me = (await client.get(f"{API}/auth/me", headers=auth)).json()
    forged = [
        "Bearer not.a.jwt",
        "Bearer " + jwt.encode({"sub": me["id"], "sid": str(uuid.uuid4()), "typ": "access", "iat": 0, "exp": 9999999999}, settings.JWT_SECRET.get_secret_value(), algorithm="HS256"),
        "Bearer " + jwt.encode({"sub": me["id"], "sid": str(uuid.uuid4()), "typ": "access", "iat": 0, "exp": 9999999999}, None, algorithm="none"),
        "Basic dXNlcjpwYXNz",
    ]
    for header in forged:
        r = await client.get(f"{API}/auth/me", headers={"Authorization": header})
        assert r.status_code == 401, header


async def test_refresh_token_cannot_be_used_as_access_token(client):
    await register_and_login(client, "swap@example.com")
    refresh = client.cookies.get("refresh_token")
    r = await client.get(f"{API}/auth/me", headers={"Authorization": f"Bearer {refresh}"})
    assert r.status_code == 401


async def test_rate_limiting_login(client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    await client.post(f"{API}/auth/register", json={"email": "rl@example.com", "password": PASSWORD, "full_name": "Test User"})
    codes = [(await client.post(f"{API}/auth/login", json={"email": "rl@example.com", "password": "Wrong12345"})).status_code for _ in range(12)]
    assert codes[:10] == [401] * 10
    assert codes[10] == 429
    r = await client.post(f"{API}/auth/login", json={"email": "rl@example.com", "password": PASSWORD})
    assert r.status_code == 429 and int(r.headers["retry-after"]) > 0
    assert r.json()["error"]["code"] == "RATE_LIMITED"


async def test_oversized_json_body_rejected(client, auth):
    r = await client.post(f"{API}/conversations", content=b'{"title": "' + b"a" * (settings.MAX_JSON_BODY_BYTES + 10) + b'"}', headers={**auth, "Content-Type": "application/json"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


async def test_message_length_limit(client, auth, agent):
    conv = await create_conversation(client, auth)
    r = await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "a" * (settings.MAX_MESSAGE_CHARS + 1)}, headers=auth)
    assert r.status_code == 422


async def test_malformed_ids_and_unknown_routes(client, auth):
    assert (await client.get(f"{API}/conversations/not-a-uuid", headers=auth)).status_code == 422
    r = await client.get(f"{API}/nope", headers=auth)
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"


async def test_errors_carry_request_id_and_no_internals(client):
    r = await client.post(f"{API}/auth/login", json={"email": "x@example.com", "password": "Wrong12345"}, headers={"X-Request-ID": "trace-12345678"})
    assert r.headers["x-request-id"] == "trace-12345678"
    assert r.json()["error"]["request_id"] == "trace-12345678"
    # Malicious request ids are replaced, not reflected.
    r = await client.get("/health", headers={"X-Request-ID": "bad\nid"})
    assert r.headers["x-request-id"] != "bad\nid"


async def test_unhandled_exceptions_are_generic(client, auth, monkeypatch):
    from app.services import conversation_service

    async def boom(*a, **k):
        raise RuntimeError("SELECT * FROM users -- secret dsn postgresql://u:p@db")

    monkeypatch.setattr(conversation_service, "list_for_user", boom)
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://t") as c:
        r = await c.get(f"{API}/conversations", headers=auth)
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert "SELECT" not in r.text and "postgresql" not in r.text


async def test_cors_allows_only_configured_origin(client):
    ok = await client.options(f"{API}/auth/login", headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert ok.headers.get("access-control-allow-credentials") == "true"
    bad = await client.options(f"{API}/auth/login", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


async def test_unhandled_error_logs_route_template_not_the_share_token(monkeypatch, caplog):
    """A crash on a public share link must not write the share token (a bearer credential) to logs."""
    import logging

    import httpx

    from app.main import app
    from app.services import share_service

    async def boom(db, token):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(share_service, "get_public", boom)
    caplog.set_level(logging.INFO)
    token = "ShareTokenSecret_" + "x" * 30
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://testserver") as c:
        r = await c.get(f"{API}/shares/{token}")
    assert r.status_code == 500 and r.json()["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert token not in r.text
    record = next(rec for rec in caplog.records if rec.getMessage() == "unhandled_error")
    assert record.route == "/api/v1/shares/{share_token}"
    server_records = [rec for rec in caplog.records if rec.name != "httpx"]  # httpx = this test's own client
    assert server_records and all(token not in f"{rec.getMessage()} {rec.__dict__}" for rec in server_records)
