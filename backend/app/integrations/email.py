"""Email delivery (used only for password-reset links).

Auth code depends on `EmailService` only:

    EmailService.send_password_reset_email(recipient, reset_url, expiry_minutes)
        └── EmailProvider.send(to, subject, text, html)   selected by EMAIL_PROVIDER

- resend:  Resend HTTP API (production). Key: EMAIL_API_KEY, sender: EMAIL_FROM
- smtp:    any SMTP server (Gmail/Google Workspace with an app password, Outlook, Brevo, ...)
- file:    development only — writes messages to EMAIL_OUTBOX_PATH so links can be read locally
- console: logs recipient domain + subject only (never bodies: they contain reset tokens)
- mock:    in-memory MockEmailService for automated tests (records, never sends)

Providers raise EmailDeliveryError when delivery fails. Nothing here ever logs an API key,
a message body, a reset URL or a reset token.
"""
import html
import logging
import re
import smtplib
import ssl
import uuid
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import List, Optional, Protocol
from urllib.parse import parse_qs, urlparse

import anyio
import httpx

from app.core.config import settings
from app.core.exceptions import ServiceUnavailable
from app.core.time import utcnow

logger = logging.getLogger("advancerag.email")

RESEND_API_URL = "https://api.resend.com/emails"
PASSWORD_RESET_SUBJECT = "Reset your AdvancedRAG password"


class EmailDeliveryError(ServiceUnavailable):
    code = "EMAIL_DELIVERY_FAILED"
    message = "We couldn't send the email right now. Please try again in a moment."


# ------------------------------------------------------------------------------ content

@dataclass(frozen=True)
class EmailContent:
    subject: str
    text: str
    html: str


def password_reset_content(reset_url: str, expiry_minutes: int) -> EmailContent:
    url = html.escape(reset_url, quote=True)
    text = (
        "Someone requested a password reset for your AdvancedRAG account.\n\n"
        f"Reset your password using the link below. Your reset link expires in {expiry_minutes} minutes "
        "and can be used once.\n\n"
        f"{reset_url}\n\n"
        "If you did not request this, you can safely ignore this email. Your password will not change.\n\n"
        "— AdvancedRAG\n"
    )
    body = f"""<!doctype html>
<html><body style="margin:0;padding:0;background:#f4f4f5;">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#f4f4f5;padding:32px 16px;">
<tr><td align="center">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:480px;background:#ffffff;border-radius:8px;font-family:Arial,Helvetica,sans-serif;color:#18181b;">
<tr><td style="padding:32px 32px 8px;font-size:20px;font-weight:bold;">AdvancedRAG</td></tr>
<tr><td style="padding:8px 32px;font-size:15px;line-height:22px;">Someone requested a password reset for your AdvancedRAG account.</td></tr>
<tr><td style="padding:8px 32px;font-size:15px;line-height:22px;">Reset your password using the button below.</td></tr>
<tr><td style="padding:16px 32px;">
<a href="{url}" style="display:inline-block;background:#18181b;color:#ffffff;text-decoration:none;font-size:15px;font-weight:bold;padding:12px 24px;border-radius:6px;">Reset Password</a>
</td></tr>
<tr><td style="padding:8px 32px;font-size:13px;line-height:20px;color:#52525b;">Your reset link expires in {expiry_minutes} minutes and can be used once.</td></tr>
<tr><td style="padding:8px 32px;font-size:13px;line-height:20px;color:#52525b;">If the button doesn't work, copy this link into your browser:<br><a href="{url}" style="color:#2563eb;word-break:break-all;">{url}</a></td></tr>
<tr><td style="padding:16px 32px 32px;font-size:13px;line-height:20px;color:#52525b;">If you did not request this, you can safely ignore this email. Your password will not change.</td></tr>
</table>
</td></tr>
</table>
</body></html>
"""
    return EmailContent(PASSWORD_RESET_SUBJECT, text, body)


# ------------------------------------------------------------------------------ providers

class EmailProvider(Protocol):
    name: str

    async def send(self, to: str, subject: str, text: str, html: Optional[str] = None) -> None: ...


class ConsoleEmailProvider:
    name = "console"

    async def send(self, to: str, subject: str, text: str, html: Optional[str] = None) -> None:
        logger.info("email_suppressed", extra={"subject": subject, "to_domain": to.split("@")[-1]})


class FileEmailProvider:
    name = "file"

    def __init__(self, directory: str):
        self.directory = Path(directory)

    async def send(self, to: str, subject: str, text: str, html: Optional[str] = None) -> None:
        path = self.directory / f"{utcnow():%Y%m%dT%H%M%S%f}-{uuid.uuid4().hex[:8]}.txt"

        def _write():
            self.directory.mkdir(parents=True, exist_ok=True)
            path.write_text(f"To: {to}\nSubject: {subject}\n\n{text}\n", encoding="utf-8", newline="\n")

        await anyio.to_thread.run_sync(_write)
        logger.info("email_written_to_outbox", extra={"subject": subject})


_ERROR_NAME = re.compile(r"^[a-z_]{1,64}$")


class ResendEmailProvider:
    name = "resend"

    def __init__(self, api_key: str, sender: str, *, transport: Optional[httpx.AsyncBaseTransport] = None):
        self._api_key, self.sender, self._transport = api_key, sender, transport

    def __repr__(self) -> str:  # never include the key
        return "ResendEmailProvider()"

    async def send(self, to: str, subject: str, text: str, html: Optional[str] = None) -> None:
        payload = {"from": self.sender, "to": [to], "subject": subject, "text": text}
        if html:
            payload["html"] = html
        try:
            async with httpx.AsyncClient(timeout=15.0, transport=self._transport) as client:
                response = await client.post(RESEND_API_URL, headers={"Authorization": f"Bearer {self._api_key}"}, json=payload)
        except httpx.HTTPError as exc:
            logger.error("email_send_failed", extra={"provider": self.name, "error_type": type(exc).__name__})
            raise EmailDeliveryError() from None  # no chained exception: it holds the request headers
        if not response.is_success:
            # Only the status and Resend's error *name* (e.g. "validation_error"); its free-text
            # message can echo addresses.
            try:
                error_name = str(response.json().get("name", ""))
            except ValueError:
                error_name = ""
            logger.error("email_send_failed", extra={
                "provider": self.name, "status": response.status_code,
                "error_name": error_name if _ERROR_NAME.match(error_name) else "unknown",
            })
            raise EmailDeliveryError()


class SmtpEmailProvider:
    name = "smtp"

    def __init__(self, *, host: str, port: int, username: Optional[str], password: Optional[str], sender: str,
                 security: str = "starttls", timeout: float = 20.0):
        self.host, self.port, self.username, self._password = host, port, username, password
        self.sender, self.security, self.timeout = sender, security, timeout

    def _send_sync(self, message: EmailMessage) -> None:
        context = ssl.create_default_context()
        if self.security == "ssl":
            server = smtplib.SMTP_SSL(self.host, self.port, timeout=self.timeout, context=context)
        else:
            server = smtplib.SMTP(self.host, self.port, timeout=self.timeout)
        with server:
            if self.security == "starttls":
                server.starttls(context=context)
            if self.username and self._password:
                server.login(self.username, self._password)
            server.send_message(message)

    async def send(self, to: str, subject: str, text: str, html: Optional[str] = None) -> None:
        message = EmailMessage()
        message["From"] = self.sender
        message["To"] = to
        message["Subject"] = subject
        message.set_content(text)
        if html:
            message.add_alternative(html, subtype="html")
        try:
            await anyio.to_thread.run_sync(self._send_sync, message)
        except (smtplib.SMTPException, OSError) as exc:
            # Only the exception type is logged (SMTP errors can echo addresses/credentials).
            logger.error("email_send_failed", extra={"provider": self.name, "error_type": type(exc).__name__})
            raise EmailDeliveryError() from exc


# ------------------------------------------------------------------------------ service

class EmailService:
    """The only email interface the auth code uses."""

    def __init__(self, provider: EmailProvider):
        self.provider = provider

    async def send_password_reset_email(self, recipient: str, reset_url: str, expiry_minutes: int) -> None:
        content = password_reset_content(reset_url, expiry_minutes)
        await self.provider.send(recipient, content.subject, content.text, content.html)


@dataclass
class SentPasswordReset:
    recipient: str
    subject: str
    reset_url: str
    expiry_minutes: int
    text: str
    html: str
    delivered: bool


class MockEmailService(EmailService):
    """Deterministic in-memory email service for automated tests: records every password-reset
    email (recipient, subject, reset URL, expiry, rendered bodies, delivered or not) and never
    contacts a provider. Set `fail = True` to simulate a provider failure."""

    def __init__(self):
        self.sent: List[SentPasswordReset] = []
        self.fail = False

    async def send_password_reset_email(self, recipient: str, reset_url: str, expiry_minutes: int) -> None:
        content = password_reset_content(reset_url, expiry_minutes)
        self.sent.append(SentPasswordReset(recipient, content.subject, reset_url, expiry_minutes, content.text, content.html, not self.fail))
        if self.fail:
            raise EmailDeliveryError()

    def clear(self) -> None:
        self.sent.clear()
        self.fail = False

    @property
    def last(self) -> SentPasswordReset:
        return self.sent[-1]

    def last_token(self) -> str:
        return parse_qs(urlparse(self.last.reset_url).query)["token"][0]


_mock_service = MockEmailService()


def get_email_service() -> EmailService:
    provider = settings.EMAIL_PROVIDER or ("file" if settings.ENVIRONMENT == "development" else "console")
    if provider == "mock":
        return _mock_service
    if provider == "resend":
        if not settings.EMAIL_API_KEY:
            raise RuntimeError("EMAIL_PROVIDER=resend requires EMAIL_API_KEY")
        return EmailService(ResendEmailProvider(settings.EMAIL_API_KEY.get_secret_value(), settings.EMAIL_FROM))
    if provider == "smtp":
        if not settings.SMTP_HOST:
            raise RuntimeError("EMAIL_PROVIDER=smtp requires SMTP_HOST")
        return EmailService(SmtpEmailProvider(
            host=settings.SMTP_HOST,
            port=settings.SMTP_PORT,
            username=settings.SMTP_USERNAME,
            password=settings.SMTP_PASSWORD.get_secret_value() if settings.SMTP_PASSWORD else None,
            sender=settings.EMAIL_FROM,
            security=settings.SMTP_SECURITY,
        ))
    if provider == "file":
        return EmailService(FileEmailProvider(settings.EMAIL_OUTBOX_PATH))
    return EmailService(ConsoleEmailProvider())
