import io
import uuid
import zipfile
from datetime import timedelta

import jwt
import pytest

from app.core import security
from app.core.config import Settings, settings
from app.core.time import utcnow
from app.integrations.storage import LocalStorage, make_storage_key
from app.services.conversation_service import escape_like
from app.services.document_service import _content_matches, extension_of, sanitize_filename


def test_password_hash_is_argon2id_and_verifies():
    h = security.hash_password("S3cret-pass")
    assert h.startswith("$argon2id$")
    assert "S3cret-pass" not in h
    assert security.verify_password("S3cret-pass", h)
    assert not security.verify_password("wrong", h)
    assert not security.verify_password("S3cret-pass", None)
    assert not security.verify_password("S3cret-pass", "not-a-hash")


def test_access_and_refresh_tokens_are_not_interchangeable():
    uid, sid = uuid.uuid4(), uuid.uuid4()
    access = security.create_access_token(uid, sid)
    refresh = security.create_refresh_token(uid, sid, utcnow() + timedelta(days=1))
    assert security.decode_token(access, security.ACCESS).user_id == uid
    assert security.decode_token(access, security.REFRESH) is None
    assert security.decode_token(refresh, security.ACCESS) is None
    claims = security.decode_token(refresh, security.REFRESH)
    assert claims.session_id == sid and claims.jti


def test_access_token_contains_only_minimal_claims():
    token = security.create_access_token(uuid.uuid4(), uuid.uuid4())
    payload = jwt.decode(token, options={"verify_signature": False})
    assert set(payload) == {"sub", "sid", "typ", "iat", "exp"}


@pytest.mark.parametrize(
    "token",
    [
        "garbage",
        jwt.encode({"sub": str(uuid.uuid4()), "sid": str(uuid.uuid4()), "typ": "access", "iat": 0, "exp": 9999999999}, "a-different-but-sufficiently-long-signing-key", algorithm="HS256"),
        jwt.encode({"sub": str(uuid.uuid4()), "sid": str(uuid.uuid4()), "typ": "access", "iat": 0, "exp": 9999999999}, None, algorithm="none"),
        jwt.encode({"sub": "not-a-uuid", "sid": str(uuid.uuid4()), "typ": "access", "iat": 0, "exp": 9999999999}, settings.JWT_SECRET.get_secret_value(), algorithm="HS256"),
        jwt.encode({"sub": str(uuid.uuid4()), "sid": str(uuid.uuid4()), "typ": "access", "iat": 0, "exp": 1}, settings.JWT_SECRET.get_secret_value(), algorithm="HS256"),
    ],
)
def test_malformed_forged_and_expired_tokens_rejected(token):
    assert security.decode_token(token, security.ACCESS) is None


def test_opaque_token_hash_is_deterministic_and_not_the_token():
    t = security.generate_token()
    assert len(t) >= 40
    assert security.hash_token(t) == security.hash_token(t) != t


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\system32\\evil.docx", "evil.docx"),
        ("C:\\Users\\me\\notes.txt", "notes.txt"),
        ("na\x00me<>.txt", "na_me__.txt"),
        ("  .hidden.md  ", "hidden.md"),
        (None, ""),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_sanitize_filename_truncates_and_keeps_extension():
    name = sanitize_filename("a" * 500 + ".pdf")
    assert len(name) <= 200 and name.endswith(".pdf")
    assert extension_of(name) == "pdf"


def _docx_bytes(valid=True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("word/document.xml" if valid else "other.xml", "<w/>")
    return buf.getvalue()


@pytest.mark.parametrize(
    "ext,content,ok",
    [
        ("pdf", b"%PDF-1.7 ...", True),
        ("pdf", b"MZ\x90\x00 windows exe", False),
        ("png", b"\x89PNG\r\n\x1a\n....", True),
        ("jpg", b"\xff\xd8\xff\xe0....", True),
        ("txt", b"plain text", True),
        ("txt", b"\x7fELF\x00\x00binary", False),
        ("docx", _docx_bytes(True), True),
        ("docx", _docx_bytes(False), False),
        ("docx", b"PK\x03\x04 corrupt", False),
    ],
)
def test_content_sniffing(ext, content, ok):
    assert _content_matches(ext, content[:8192], io.BytesIO(content)) is ok


def test_storage_keys_are_server_generated_and_confined(tmp_path):
    key = make_storage_key(uuid.uuid4(), uuid.uuid4())
    assert ".." not in key
    store = LocalStorage(str(tmp_path))
    with pytest.raises(ValueError):
        store._path("../../outside")


def test_escape_like_neutralises_wildcards():
    assert escape_like("100%_a\\b") == "100\\%\\_a\\\\b"


def test_production_rejects_unsafe_configuration_without_leaking_secrets():
    from app.core.config import ConfigError

    good = {
        "DATABASE_URL": "postgresql+asyncpg://u:dbpassword123@db/app", "ENVIRONMENT": "production", "_env_file": None,
        "JWT_SECRET": "a" * 40, "JWT_REFRESH_SECRET": "b" * 40, "FRONTEND_ORIGIN": "https://app.example.com",
        "INTERNAL_SERVICE_TOKEN": "t" * 40, "AGENT_CORE_URL": "https://agent.internal", "RAG_CORE_URL": "https://rag.internal",
        "RATE_LIMIT_BACKEND": "redis", "REDIS_URL": "redis://redis:6379/0", "EMAIL_PROVIDER": "resend", "EMAIL_API_KEY": "re_" + "k" * 20,
        "EMAIL_FROM": "AdvancedRAG <no-reply@mail.example.com>",
        "STORAGE_BACKEND": "s3", "STORAGE_BUCKET": "b", "STORAGE_ACCESS_KEY": "ak", "STORAGE_SECRET_KEY": "sk",
    }
    ok = Settings(**good).check_safety()
    assert ok.cookie_secure is True and ok.readiness_requires_services is True
    bad_cases = [
        {"JWT_SECRET": "supersecretkey_change_in_production"},
        {"FRONTEND_ORIGIN": "*"},
        {"DATABASE_URL": "sqlite+aiosqlite:///x.db"},
        {"INTERNAL_SERVICE_TOKEN": ""},
        {"AGENT_CORE_URL": "http://localhost:8001"},
        {"RATE_LIMIT_BACKEND": "memory"},
        {"EMAIL_API_KEY": ""},  # resend selected but not configured
        {"EMAIL_FROM": "AdvancedRAG <no-reply@localhost>"},  # Resend can't send from an unverified/local domain
        {"FRONTEND_ORIGIN": "http://app.example.com"},  # reset tokens must not travel over plain http
        {"EMAIL_PROVIDER": "mock"},  # test-only provider
        {"EMAIL_PROVIDER": "file"},  # development-only provider
        {"STORAGE_BACKEND": "local"},
        {"STORAGE_BUCKET": ""},
    ]
    for override in bad_cases:
        with pytest.raises(ConfigError) as exc:
            Settings(**{**good, **override}).check_safety()
        text = str(exc.value)
        assert "dbpassword123" not in text and "a" * 40 not in text and "re_" not in text
    # A mounted persistent volume can be explicitly allowed.
    Settings(**{**good, "STORAGE_BACKEND": "local", "ALLOW_LOCAL_STORAGE_IN_PRODUCTION": True}).check_safety()
    # Email is optional (only password-reset links use it): no provider is a valid production setup.
    Settings(**{**good, "EMAIL_PROVIDER": "", "EMAIL_API_KEY": ""}).check_safety()


def test_environment_must_be_explicit(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    with pytest.raises(ValueError):
        Settings(_env_file=None, DATABASE_URL="sqlite+aiosqlite://", JWT_SECRET="x", JWT_REFRESH_SECRET="y")


def test_agent_citations_are_allowlisted_and_source_ids_sanitised():
    from app.integrations.agent_core import safe_citations
    from app.schemas.message import Citation

    raw = [
        {"source_id": 2, "document_id": None, "filename": "a.pdf", "page": 3, "chunk_id": "d:1", "snippet": "x", "prompt": "secret"},
        {"source_id": "2", "filename": "b.pdf"},
        {"source_id": 0, "filename": "c.pdf"},
        {"source_id": True, "filename": "d.pdf"},
        "not a dict",
    ]
    out = safe_citations(raw)
    assert out[0] == {"source_id": 2, "document_id": None, "filename": "a.pdf", "page": 3, "chunk_id": "d:1", "snippet": "x"}
    assert [c.get("source_id") for c in out] == [2, None, None, None] and len(out) == 4
    assert all(Citation.model_validate(c) for c in out)  # nothing that could break message rendering


async def test_chat_stream_timeout_is_not_inherited_from_a_health_probe_client():
    """Regression (2026-10-08): /ready created the shared Agent Core client with a 15 s timeout and
    chat streams reused it, so a slow follow-up rewrite was cut off after 15 s."""
    import httpx

    from app.core.config import settings
    from app.integrations import http as backend_http
    from app.integrations.agent_core import AgentCoreClient, AgentRequest
    from app.integrations.rag_core import RAGCoreClient

    seen = {}

    def handler(request: httpx.Request):
        seen[request.url.path] = request.extensions["timeout"]
        if request.url.path.endswith("/stream"):
            return httpx.Response(200, text="event: generation.completed\ndata: {}\n\n", headers={"content-type": "text/event-stream"})
        return httpx.Response(204)

    # Health probes use their own pool now…
    await backend_http.ping("agent_core", "http://127.0.0.9:9", "/ready")
    assert "agent_core:health" in backend_http._clients and "agent_core" not in backend_http._clients
    # …and even a pre-existing short-timeout pool cannot shorten real calls.
    for name in ("agent_core", "rag_core"):
        backend_http._clients[name] = httpx.AsyncClient(base_url=f"http://{name}", timeout=15.0, transport=httpx.MockTransport(handler))
    try:
        req = AgentRequest(uuid.uuid4(), uuid.uuid4(), uuid.uuid4(), "hi", [], [])
        async for _ in AgentCoreClient().stream(req):
            pass
        await RAGCoreClient().delete_document(document_id=uuid.uuid4(), user_id=uuid.uuid4())
    finally:
        await backend_http.close_clients()
    assert seen["/api/v1/agent/stream"]["read"] == settings.AGENT_CORE_TIMEOUT_SECONDS
    assert next(v for k, v in seen.items() if k.startswith("/api/v1/rag/documents/"))["read"] == settings.RAG_CORE_TIMEOUT_SECONDS


@pytest.mark.parametrize("hops,legacy,xff,peer,expected", [
    (0, False, "6.6.6.6", "10.0.0.1", "10.0.0.1"),                        # no proxy: header ignored
    (1, False, "203.0.113.7", "10.0.0.1", "203.0.113.7"),                  # Railway edge appended the client
    (1, False, "6.6.6.6, 203.0.113.7", "10.0.0.1", "203.0.113.7"),         # forged left entry ignored
    (2, False, "6.6.6.6, 203.0.113.7, 76.76.21.9", "10.0.0.1", "203.0.113.7"),  # Vercel + Railway
    (2, False, "203.0.113.7", "10.0.0.1", "10.0.0.1"),                     # chain shorter than hops: peer
    (0, True, "6.6.6.6, 203.0.113.7", "10.0.0.1", "203.0.113.7"),          # legacy TRUST_PROXY_HEADERS = 1 hop
])
def test_client_ip_uses_trusted_hops_from_the_right(monkeypatch, hops, legacy, xff, peer, expected):
    from starlette.requests import Request

    from app.middleware import rate_limit

    monkeypatch.setattr(rate_limit.settings, "TRUSTED_PROXY_HOPS", hops)
    monkeypatch.setattr(rate_limit.settings, "TRUST_PROXY_HEADERS", legacy)
    scope = {"type": "http", "headers": [(b"x-forwarded-for", xff.encode())], "client": (peer, 1234)}
    assert rate_limit.client_ip(Request(scope)) == expected
