from datetime import timedelta

from sqlalchemy import select

from app.core.time import utcnow
from app.db import session as db_session
from app.db.models import Share
from conftest import create_conversation, parse_sse, upload

API = "/api/v1"


async def _chat(client, auth, title, text):
    conv = await create_conversation(client, auth, title)
    await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": text}, headers=auth)
    return conv


async def test_search_titles_and_message_content(client, auth, agent):
    a = await _chat(client, auth, "Quarterly finance", "revenue numbers please")
    b = await _chat(client, auth, "Holiday plans", "Find flights to Lisbon")
    await _chat(client, auth, "Other", "nothing relevant")

    r = await client.get(f"{API}/search/chats", params={"q": "finance"}, headers=auth)
    assert [c["id"] for c in r.json()["items"]] == [a["id"]]

    r = await client.get(f"{API}/search/chats", params={"q": "LISBON"}, headers=auth)
    items = r.json()["items"]
    assert [c["id"] for c in items] == [b["id"]] and "Lisbon" in items[0]["snippet"]


async def test_search_treats_wildcards_and_sql_literally(client, auth, agent):
    await _chat(client, auth, "100% done", "x")
    await _chat(client, auth, "plain", "y")
    assert (await client.get(f"{API}/search/chats", params={"q": "%"}, headers=auth)).json()["total"] == 1
    for payload in ["' OR 1=1 --", "\"; DROP TABLE conversations; --", "_"]:
        r = await client.get(f"{API}/search/chats", params={"q": payload}, headers=auth)
        assert r.status_code == 200 and r.json()["total"] == 0
    assert (await client.get(f"{API}/conversations", headers=auth)).json()["total"] == 2
    assert (await client.get(f"{API}/search/chats", params={"q": ""}, headers=auth)).status_code == 422


async def test_share_lifecycle(client, auth, agent, make_client):
    conv = await _chat(client, auth, "Shared chat", "public question")
    r = await client.post(f"{API}/conversations/{conv['id']}/share", headers=auth)
    assert r.status_code == 201
    share = r.json()
    assert share["url"].endswith(f"/share/{share['token']}") and share["url"].startswith("http://localhost:5173")

    # Messages sent after the link was created are not shared.
    await client.post(f"{API}/conversations/{conv['id']}/messages/stream", json={"content": "private follow-up"}, headers=auth)

    anon = make_client()
    r = await anon.get(f"{API}/shares/{share['token']}")
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "Shared chat"
    assert [m["content"] for m in body["messages"]] == ["public question", "Hello from the agent."]
    # Nothing private or internal leaks.
    assert set(body) == {"title", "shared_at", "messages"}
    assert all(set(m) == {"role", "content", "created_at"} for m in body["messages"])
    assert "@example.com" not in r.text and conv["id"] not in r.text

    status = (await client.get(f"{API}/conversations/{conv['id']}/share", headers=auth)).json()
    assert status["active"] is True and "token" not in status

    assert (await client.delete(f"{API}/conversations/{conv['id']}/share", headers=auth)).status_code == 204
    r = await anon.get(f"{API}/shares/{share['token']}")
    assert r.status_code == 404 and r.json()["error"]["code"] == "SHARE_NOT_FOUND"
    assert (await client.get(f"{API}/conversations/{conv['id']}/share", headers=auth)).json()["active"] is False


async def test_new_share_replaces_old_link(client, auth, agent):
    conv = await _chat(client, auth, "c", "m")
    first = (await client.post(f"{API}/conversations/{conv['id']}/share", headers=auth)).json()["token"]
    second = (await client.post(f"{API}/conversations/{conv['id']}/share", headers=auth)).json()["token"]
    assert (await client.get(f"{API}/shares/{first}")).status_code == 404
    assert (await client.get(f"{API}/shares/{second}")).status_code == 200


async def test_expired_and_invalid_shares_rejected(client, auth, agent):
    conv = await _chat(client, auth, "c", "m")
    token = (await client.post(f"{API}/conversations/{conv['id']}/share", json={"expires_in_days": 1}, headers=auth)).json()["token"]
    assert (await client.get(f"{API}/shares/{token}")).status_code == 200
    async with db_session.session_scope() as db:
        share = await db.scalar(select(Share))
        assert share.share_token_hash != token
        share.expires_at = utcnow() - timedelta(seconds=1)
        await db.commit()
    assert (await client.get(f"{API}/shares/{token}")).status_code == 404
    assert (await client.get(f"{API}/shares/not-a-real-token")).status_code == 404


async def test_settings_persist(client, auth):
    assert (await client.get(f"{API}/settings", headers=auth)).json() == {"theme": "dark"}
    r = await client.patch(f"{API}/settings", json={"theme": "light"}, headers=auth)
    assert r.json() == {"theme": "light"}
    assert (await client.get(f"{API}/settings", headers=auth)).json() == {"theme": "light"}
    assert (await client.patch(f"{API}/settings", json={"theme": "neon"}, headers=auth)).status_code == 422
    assert (await client.patch(f"{API}/settings", json={"evil": 1}, headers=auth)).status_code == 422


async def test_clear_all_data(client, auth, agent):
    await _chat(client, auth, "a", "b")
    await upload(client, auth)
    assert (await client.delete(f"{API}/profile/data", headers=auth)).status_code == 204
    assert (await client.get(f"{API}/conversations", headers=auth)).json()["total"] == 0
    assert (await client.get(f"{API}/documents", headers=auth)).json()["total"] == 0
    assert (await client.get(f"{API}/auth/me", headers=auth)).status_code == 200


async def test_health_endpoints(client):
    assert (await client.get("/health")).json() == {"status": "ok"}
    assert (await client.get("/health/live")).status_code == 200
    r = await client.get("/health/ready")
    assert r.status_code == 200 and r.json()["checks"]["database"] is True
    assert "http" not in r.text
