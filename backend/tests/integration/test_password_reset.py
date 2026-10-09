"""Password reset by email: forgot-password -> hashed single-use token -> EmailService (Resend in
production, MockEmailService here) -> reset-password -> sessions revoked. No test contacts Resend."""
import hashlib
import json
import logging
import pathlib
import re
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from sqlalchemy import select

from app.core.config import settings
from app.core.time import utcnow
from app.db import session as db_session
from app.db.models import AuthToken, User
from app.integrations import email as email_mod
from conftest import PASSWORD, login, register_and_login

API = "/api/v1"

NEW_PASSWORD = "Brand-new-pw1"
GENERIC = "If an account exists for this email, a password reset link has been sent."


async def forgot(client, email, **kw):
    return await client.post(f"{API}/auth/forgot-password", json={"email": email}, **kw)


async def reset(client, token, password=NEW_PASSWORD, **kw):
    return await client.post(f"{API}/auth/reset-password", json={"token": token, "new_password": password}, **kw)


def log_text(caplog) -> str:
    """Everything a log handler could emit: messages, args and every `extra` field."""
    return "\n".join(f"{r.getMessage()} {json.dumps(r.__dict__, default=str)}" for r in caplog.records)


# ------------------------------------------------------------------------- forgot password

async def test_existing_account_gets_a_reset_email_with_an_opaque_link(client, outbox):
    await register_and_login(client, "alice@example.com")
    r = await forgot(client, "alice@example.com")
    assert r.status_code == 200 and r.json() == {"message": GENERIC}

    assert len(outbox.sent) == 1
    mail = outbox.last
    assert mail.recipient == "alice@example.com" and mail.delivered
    assert mail.subject == "Reset your AdvancedRAG password"
    assert mail.expiry_minutes == settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES
    url = urlparse(mail.reset_url)
    assert f"{url.scheme}://{url.netloc}" == settings.frontend_url and url.path == "/reset-password"
    query = parse_qs(url.query)
    assert list(query) == ["token"]  # nothing but the opaque token
    token = query["token"][0]
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", token)  # 32 random bytes
    async with db_session.session_scope() as db:
        user = await db.scalar(select(User).where(User.email == "alice@example.com"))
    for leak in ("alice", "example.com", str(user.id), PASSWORD):
        assert leak not in mail.reset_url


async def test_email_content(client, outbox):
    await register_and_login(client, "content@example.com")
    await forgot(client, "content@example.com")
    mail = outbox.last
    for body in (mail.text, mail.html):
        assert "AdvancedRAG" in body
        assert "Someone requested a password reset for your AdvancedRAG account." in body
        assert f"expires in {settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES} minutes" in body
        assert "If you did not request this, you can safely ignore this email." in body
        assert PASSWORD not in body and "has been changed" not in body
    assert "Reset Password" in mail.html and f'href="{mail.reset_url}"' in mail.html


async def test_unknown_or_inactive_account_gets_the_same_response_and_no_email(client, outbox):
    await register_and_login(client, "known@example.com")
    await register_and_login(client, "inactive@example.com")
    async with db_session.session_scope() as db:
        (await db.scalar(select(User).where(User.email == "inactive@example.com"))).is_active = False
        await db.commit()

    known = await forgot(client, "known@example.com")
    outbox.clear()
    for email in ("nobody@example.com", "inactive@example.com"):
        r = await forgot(client, email)
        assert (r.status_code, r.json()) == (known.status_code, known.json())
        assert set(r.headers) == set(known.headers)
    assert outbox.sent == []


async def test_only_a_hash_of_the_token_is_stored(client, outbox):
    await register_and_login(client, "hash@example.com")
    await forgot(client, "hash@example.com")
    token = outbox.last_token()
    async with db_session.session_scope() as db:
        rows = (await db.scalars(select(AuthToken))).all()
    assert len(rows) == 1
    assert rows[0].token_hash == hashlib.sha256(token.encode()).hexdigest()
    stored = json.dumps({c.name: getattr(rows[0], c.key, None) for c in AuthToken.__table__.columns}, default=str)
    assert token not in stored
    ttl = rows[0].expires_at - rows[0].created_at
    assert abs(ttl - timedelta(minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES)) < timedelta(seconds=1)


async def test_provider_failure_is_safe_and_voids_the_token(client, outbox, caplog):
    await register_and_login(client, "fail@example.com")
    ok = await forgot(client, "nobody@example.com")
    outbox.fail = True
    caplog.set_level(logging.INFO)
    r = await forgot(client, "fail@example.com")
    # Same generic answer (a different one would reveal the account exists).
    assert (r.status_code, r.json()) == (ok.status_code, ok.json())
    assert outbox.last.delivered is False
    token = outbox.last_token()
    # No live token is left behind for an email that never arrived.
    assert (await reset(client, token)).json()["error"]["code"] == "INVALID_TOKEN"
    failure = [rec for rec in caplog.records if rec.getMessage() == "password_reset_email_failed"]
    assert failure and failure[0].request_id == r.headers["x-request-id"] and failure[0].error_type == "EmailDeliveryError"
    assert token not in log_text(caplog)
    outbox.fail = False
    await forgot(client, "fail@example.com")  # a new request works again
    assert (await reset(client, outbox.last_token())).status_code == 200


def test_missing_resend_credentials_fail_clearly(monkeypatch):
    from app.core.config import ConfigError, Settings

    monkeypatch.setattr(settings, "EMAIL_PROVIDER", "resend")
    monkeypatch.setattr(settings, "EMAIL_API_KEY", None)
    with pytest.raises(RuntimeError, match="EMAIL_API_KEY"):
        email_mod.get_email_service()
    prod = {
        "DATABASE_URL": "postgresql+asyncpg://u:p@db.example.com/app?ssl=require", "ENVIRONMENT": "production", "_env_file": None,
        "JWT_SECRET": "a" * 40, "JWT_REFRESH_SECRET": "b" * 40, "FRONTEND_ORIGIN": "https://app.example.com",
        "INTERNAL_SERVICE_TOKEN": "t" * 40, "AGENT_CORE_URL": "http://agent-core.railway.internal:8001",
        "RAG_CORE_URL": "http://rag-core.railway.internal:8002", "RATE_LIMIT_BACKEND": "redis", "REDIS_URL": "redis://r:6379/0",
        "STORAGE_BACKEND": "s3", "STORAGE_BUCKET": "b", "STORAGE_ACCESS_KEY": "ak", "STORAGE_SECRET_KEY": "sk",
        "EMAIL_PROVIDER": "resend", "EMAIL_FROM": "AdvancedRAG <no-reply@mail.example.com>",
    }
    with pytest.raises(ConfigError, match="resend needs EMAIL_API_KEY"):
        Settings(**prod).check_safety()
    Settings(**prod, EMAIL_API_KEY="re_" + "x" * 30).check_safety()  # complete config starts


def test_development_and_tests_need_no_resend_credentials(monkeypatch):
    monkeypatch.setattr(settings, "EMAIL_API_KEY", None)
    monkeypatch.setattr(settings, "EMAIL_PROVIDER", "mock")
    assert isinstance(email_mod.get_email_service(), email_mod.MockEmailService)
    monkeypatch.setattr(settings, "EMAIL_PROVIDER", "")
    assert isinstance(email_mod.get_email_service().provider, (email_mod.FileEmailProvider, email_mod.ConsoleEmailProvider))


# ------------------------------------------------------------------------- Resend provider

async def test_resend_provider_sends_text_and_html_with_the_key_only_in_the_auth_header():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"id": "email-id"})

    key = "re_SENTINEL_" + "k" * 24
    service = email_mod.EmailService(email_mod.ResendEmailProvider(key, "AdvancedRAG <no-reply@mail.example.com>", transport=httpx.MockTransport(handler)))
    await service.send_password_reset_email("bob@example.com", "https://app.example.com/reset-password?token=abc", 30)
    req = seen[0]
    assert str(req.url) == "https://api.resend.com/emails"
    assert req.headers["authorization"] == f"Bearer {key}"
    body = json.loads(req.content)
    assert body["from"] == "AdvancedRAG <no-reply@mail.example.com>" and body["to"] == ["bob@example.com"]
    assert body["subject"] == "Reset your AdvancedRAG password"
    assert "token=abc" in body["text"] and "token=abc" in body["html"]
    assert key not in req.content.decode() and key not in repr(service.provider)


@pytest.mark.parametrize("failure", ["http_error", "network_error"])
async def test_resend_failure_raises_safe_error_without_leaking(failure, caplog):
    key = "re_SENTINEL_" + "k" * 24

    def handler(request):
        if failure == "network_error":
            raise httpx.ConnectError(f"connect failed (auth {request.headers['authorization']})")
        return httpx.Response(422, json={"name": "validation_error", "message": "Invalid `to` field: bob@example.com"})

    provider = email_mod.ResendEmailProvider(key, "AdvancedRAG <no-reply@mail.example.com>", transport=httpx.MockTransport(handler))
    caplog.set_level(logging.INFO)
    with pytest.raises(email_mod.EmailDeliveryError) as exc:
        await email_mod.EmailService(provider).send_password_reset_email("bob@example.com", "https://app.example.com/reset-password?token=SECRETTOKEN", 30)
    assert exc.value.code == "EMAIL_DELIVERY_FAILED" and key not in str(exc.value)
    assert exc.value.__cause__ is None  # the httpx error (with request headers) is not chained
    logs = log_text(caplog)
    assert "email_send_failed" in logs
    for leak in (key, "SECRETTOKEN", "bob@example.com", "reset-password?token"):
        assert leak not in logs
    if failure == "http_error":
        rec = next(r for r in caplog.records if r.getMessage() == "email_send_failed")
        assert rec.status == 422 and rec.error_name == "validation_error"


# ------------------------------------------------------------------------- reset password

async def test_valid_token_changes_password_and_revokes_every_session(client, make_client, outbox):
    other_device = make_client()
    headers = await register_and_login(client, "bob@example.com")
    headers_other = await login(other_device, "bob@example.com")
    await forgot(client, "bob@example.com")
    token = outbox.last_token()

    r = await reset(client, token)
    assert r.status_code == 200 and r.json() == {"message": "Your password has been reset. Please sign in."}
    assert token not in r.text
    # Every existing session and refresh token is revoked.
    for c, h in ((client, headers), (other_device, headers_other)):
        assert (await c.get(f"{API}/auth/me", headers=h)).status_code == 401
        assert (await c.post(f"{API}/auth/refresh")).status_code == 401
    # Old password fails, new password works.
    old = await client.post(f"{API}/auth/login", json={"email": "bob@example.com", "password": PASSWORD})
    assert old.status_code == 401
    new_headers = await login(client, "bob@example.com", NEW_PASSWORD)
    assert (await client.get(f"{API}/auth/me", headers=new_headers)).status_code == 200
    # The token is now unusable.
    again = await reset(client, token, "Another-pw12")
    assert again.status_code == 400 and again.json()["error"]["code"] == "INVALID_TOKEN"
    await login(client, "bob@example.com", NEW_PASSWORD)


async def _invalid_answer(client, token):
    r = await reset(client, token)
    assert r.status_code == 400, (token, r.text)
    err = r.json()["error"]
    return err["code"], err["message"]


async def test_expired_unknown_and_malformed_tokens_get_one_identical_answer(client, outbox):
    await register_and_login(client, "carol@example.com")
    await forgot(client, "carol@example.com")
    expired = outbox.last_token()
    async with db_session.session_scope() as db:
        for t in (await db.scalars(select(AuthToken))).all():
            t.expires_at = utcnow() - timedelta(seconds=1)
        await db.commit()

    expected = ("INVALID_TOKEN", "This link is invalid or has expired.")
    cases = [
        expired,
        "A" * 43,                       # well-formed, never issued
        "",                             # malformed: empty
        "short",                        # malformed: too short
        "x" * 300,                      # malformed: too long
        "abc def ghi jkl mno pqr stu vwx yz0 123",
        "../../etc/passwd/../../../../../../../",
        "<script>alert(1)</script>" + "a" * 20,
        expired + "=",
    ]
    for token in cases:
        assert await _invalid_answer(client, token) == expected
    await login(client, "carol@example.com")  # password unchanged


async def test_latest_reset_request_wins(client, outbox):
    await register_and_login(client, "dave@example.com")
    await forgot(client, "dave@example.com")
    first = outbox.last_token()
    await forgot(client, "dave@example.com")
    second = outbox.last_token()
    assert first != second
    assert (await reset(client, first)).status_code == 400
    assert (await reset(client, second)).status_code == 200


async def test_a_token_only_resets_the_account_it_was_issued_for(client, outbox):
    a_headers = await register_and_login(client, "usera@example.com")
    await register_and_login(client, "userb@example.com")
    await forgot(client, "userb@example.com")
    assert (await reset(client, outbox.last_token())).status_code == 200
    # A is untouched: old password still works, A's session is still valid.
    assert (await client.get(f"{API}/auth/me", headers=a_headers)).status_code == 200
    await login(client, "usera@example.com", PASSWORD)
    assert (await client.post(f"{API}/auth/login", json={"email": "usera@example.com", "password": NEW_PASSWORD})).status_code == 401


async def test_weak_new_password_is_rejected_and_token_stays_usable(client, outbox):
    await register_and_login(client, "erin@example.com")
    await forgot(client, "erin@example.com")
    token = outbox.last_token()
    r = await reset(client, token, "short")
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "short" not in json.dumps(r.json()["error"]["details"]).replace("at least", "")
    assert (await reset(client, token)).status_code == 200


# ------------------------------------------------------------------------- security

async def test_no_token_or_reset_url_is_logged(client, outbox, caplog):
    caplog.set_level(logging.DEBUG)
    await register_and_login(client, "frank@example.com")
    await forgot(client, "frank@example.com")
    token, url = outbox.last_token(), outbox.last.reset_url
    await reset(client, token)
    await reset(client, token)
    logs = log_text(caplog)
    assert token not in logs and url not in logs and "reset-password?token" not in logs
    assert NEW_PASSWORD not in logs


@pytest.mark.parametrize("account_exists", [True, False])
async def test_forgot_password_per_account_limit_is_identical_for_unknown_accounts(client, outbox, monkeypatch, account_exists):
    if account_exists:
        await register_and_login(client, "grace@example.com")
        outbox.clear()
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    # 3 per hour per address; case/whitespace variants share one bucket.
    variants = ("grace@example.com", "GRACE@example.com", " grace@Example.com ", "grace@example.com")
    responses = [await forgot(client, e) for e in variants]
    assert [r.status_code for r in responses] == [200, 200, 200, 429]
    assert responses[3].json()["error"]["code"] == "RATE_LIMITED"
    assert len(outbox.sent) == (3 if account_exists else 0)  # the limited request sent nothing


async def test_forgot_password_per_ip_limit_cannot_be_bypassed(client, outbox, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    # 5 per hour per client IP, across different addresses.
    codes = [(await forgot(client, f"person{i}@example.com")).status_code for i in range(6)]
    assert codes == [200] * 5 + [429]
    # A spoofed X-Forwarded-For is ignored (no trusted proxy configured).
    assert (await forgot(client, "fresh@example.com", headers={"X-Forwarded-For": "203.0.113.99"})).status_code == 429


async def test_reset_password_attempts_are_rate_limited(client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    codes = [(await reset(client, f"{'A' * 40}{i:03d}")).status_code for i in range(11)]
    assert codes[:10] == [400] * 10 and codes[10] == 429
    assert (await reset(client, "B" * 43, headers={"X-Forwarded-For": "198.51.100.1"})).status_code == 429


def test_resend_key_never_reaches_the_frontend():
    frontend = pathlib.Path(__file__).resolve().parents[3] / "frontend"
    if not frontend.is_dir():
        pytest.skip("frontend sources not present")
    files = [p for p in (frontend / "src").rglob("*") if p.is_file()]
    files += [p for p in (frontend / "dist").rglob("*") if p.is_file() and p.suffix in (".js", ".html", ".css", ".json")]
    files += [p for p in frontend.glob(".env*") if p.is_file()] + [frontend / "vercel.json", frontend / "vite.config.ts"]
    for path in files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in ("EMAIL_API_KEY", "RESEND", "api.resend.com", "SMTP_PASSWORD"):
            assert marker not in text, f"{marker} found in {path}"
        # Only VITE_* variables reach the bundle; none may carry a key/secret.
        assert not re.search(r"VITE_[A-Z_]*(KEY|SECRET|TOKEN|PASSWORD)", text), path


# ------------------------------------------------------------------------- production sender

PROD_SENDER = "AdvancedRAG <no-reply@mail.example.com>"


async def test_forgot_password_through_the_resend_provider_with_the_production_sender(client, monkeypatch, caplog):
    """Full flow through the real ResendEmailProvider code; only the HTTP call to Resend is simulated."""
    from app.api.routes import auth as auth_routes

    sent = []
    key = "re_TESTONLY_" + "z" * 24
    monkeypatch.setattr(settings, "EMAIL_FROM", PROD_SENDER)
    monkeypatch.setattr(settings, "FRONTEND_PUBLIC_URL", "https://app.example.com")
    transport = httpx.MockTransport(lambda req: sent.append(req) or httpx.Response(200, json={"id": "e1"}))
    monkeypatch.setattr(auth_routes, "get_email_service",
                        lambda: email_mod.EmailService(email_mod.ResendEmailProvider(key, settings.EMAIL_FROM, transport=transport)))
    caplog.set_level(logging.DEBUG)
    await register_and_login(client, "resend-user@example.com")
    r = await forgot(client, "resend-user@example.com")
    assert r.json() == {"message": GENERIC}

    body = json.loads(sent[0].content)
    assert body["from"] == PROD_SENDER and body["to"] == ["resend-user@example.com"]
    assert body["subject"] == "Reset your AdvancedRAG password"
    url = re.search(r"https://app\.example\.com/reset-password\?token=[A-Za-z0-9_-]{43}", body["text"]).group(0)
    assert f'href="{url}"' in body["html"] and "AdvancedRAG" in body["html"]
    token = parse_qs(urlparse(url).query)["token"][0]
    logs = log_text(caplog)
    for leak in (key, token, url):
        assert leak not in logs
    assert (await reset(client, token)).status_code == 200


def test_production_resend_settings_for_the_verified_domain_are_accepted():
    from app.core.config import Settings

    s = Settings(
        _env_file=None, ENVIRONMENT="production", DATABASE_URL="postgresql+asyncpg://u:p@db.example.com/app?ssl=require",
        JWT_SECRET="a" * 40, JWT_REFRESH_SECRET="b" * 40, FRONTEND_ORIGIN="https://app.example.com",
        FRONTEND_PUBLIC_URL="https://app.example.com", INTERNAL_SERVICE_TOKEN="t" * 40,
        AGENT_CORE_URL="http://agent-core.railway.internal:8001", RAG_CORE_URL="http://rag-core.railway.internal:8002",
        RATE_LIMIT_BACKEND="redis", REDIS_URL="redis://r:6379/0", STORAGE_BACKEND="s3", STORAGE_BUCKET="b",
        STORAGE_ACCESS_KEY="ak", STORAGE_SECRET_KEY="sk",
        EMAIL_PROVIDER="resend", EMAIL_API_KEY="re_TESTONLY_" + "y" * 24, EMAIL_FROM=PROD_SENDER,
    ).check_safety()
    assert s.email_configured and s.frontend_url == "https://app.example.com"


def test_no_real_secret_in_any_env_example():
    root = pathlib.Path(__file__).resolve().parents[3]
    examples = [p for p in root.rglob(".env.example") if "node_modules" not in p.parts and ".venv" not in p.parts]
    assert examples
    for path in examples:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#") or "=" not in line:
                continue
            name, value = (x.strip().strip('"').strip("'") for x in line.split("=", 1))
            assert not re.fullmatch(r"re_[A-Za-z0-9_]{16,}", value), f"{path}: {name} holds a Resend key"
            if re.search(r"(KEY|SECRET|PASSWORD|TOKEN)$", name):
                assert value == "" or value.startswith("<") or "change" in value.lower() or "placeholder" in value.lower(), \
                    f"{path}: {name} must be a placeholder"
