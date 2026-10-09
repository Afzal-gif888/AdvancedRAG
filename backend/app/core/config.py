import os
from functools import cached_property
from typing import List, Literal, Optional
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from advancerag_common.db import tls_problem

INSECURE_SECRET_MARKERS = ("change", "secret", "example", "placeholder")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}



def _env_file():
    """`.env` in the working directory; ADVANCERAG_ENV_FILE overrides it ("none" disables, used by tests)."""
    value = os.environ.get("ADVANCERAG_ENV_FILE", ".env")
    return None if value == "none" else value

class ConfigError(ValueError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_env_file(), env_file_encoding="utf-8", extra="ignore")

    PROJECT_NAME: str = "AdvancedRAG"
    API_V1_STR: str = "/api/v1"
    # Required, no default: a deployment that forgets it must not silently get development behaviour.
    ENVIRONMENT: Literal["development", "staging", "production", "test"]
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = True

    # Security — secrets have no defaults; they must come from the environment.
    JWT_SECRET: SecretStr
    JWT_REFRESH_SECRET: SecretStr
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=15, ge=1, le=120)
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, ge=1, le=90)
    # A just-rotated refresh token is still accepted (access token only, no new cookie) for this
    # long, so concurrent refreshes from tabs of the same browser don't look like token theft.
    REFRESH_REUSE_GRACE_SECONDS: int = Field(default=30, ge=0, le=300)
    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = Field(default=30, ge=5, le=24 * 60)

    # Refresh cookie. COOKIE_SECURE defaults to True everywhere except development/test.
    REFRESH_COOKIE_NAME: str = "refresh_token"
    COOKIE_SECURE: Optional[bool] = None
    COOKIE_SAMESITE: Literal["lax", "strict", "none"] = "lax"
    COOKIE_DOMAIN: Optional[str] = None

    # CORS — comma-separated list of allowed origins.
    FRONTEND_ORIGIN: str = "http://localhost:5173"
    FRONTEND_PUBLIC_URL: Optional[str] = None
    # Client IP behind reverse proxies (used for per-IP rate limits and session records).
    # TRUSTED_PROXY_HOPS = number of proxies YOU control in front of the backend that append to
    # X-Forwarded-For: 0 = use the TCP peer (no proxy), 1 = Railway edge only, 2 = Vercel rewrite
    # + Railway edge. The client address is taken that many entries from the RIGHT, so a forged
    # left-most X-Forwarded-For value is ignored. TRUST_PROXY_HEADERS=true is the legacy form of 1.
    TRUSTED_PROXY_HOPS: int = Field(default=0, ge=0, le=5)
    TRUST_PROXY_HEADERS: bool = False

    @property
    def proxy_hops(self) -> int:
        return self.TRUSTED_PROXY_HOPS or (1 if self.TRUST_PROXY_HEADERS else 0)

    # Database
    DATABASE_URL: SecretStr
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_ECHO: bool = False

    # Internal services. The backend talks only to Agent Core (chat) and RAG Core (ingest/delete);
    # it has no knowledge of the LLM layer or any model-provider keys.
    AGENT_CORE_URL: str = "http://localhost:8001"
    RAG_CORE_URL: str = "http://localhost:8002"
    INTERNAL_SERVICE_TOKEN: Optional[SecretStr] = None
    AGENT_CORE_TIMEOUT_SECONDS: float = 120.0
    # Per HTTP call to RAG Core (start an ingestion, poll its status, delete). Ingestion itself runs
    # in RAG Core's background and is NOT bounded by this; see INGEST_* below.
    RAG_CORE_TIMEOUT_SECONDS: float = 60.0
    CHAT_HISTORY_CONTEXT_MESSAGES: int = Field(default=20, ge=0, le=200)
    # The user's Knowledge Base (all their ready documents) is sent to Agent Core with every turn
    # and searched in every conversation; nothing has to be attached. 500 = Agent Core's limit.
    LIBRARY_DISCOVERY_ENABLED: bool = True
    LIBRARY_DISCOVERY_MAX_DOCUMENTS: int = Field(default=500, ge=1, le=500)
    # /ready fails when Agent Core / RAG Core are down (defaults to on outside development).
    READINESS_REQUIRE_SERVICES: Optional[bool] = None
    # Hosts that put idle services to sleep (Render free): ping Agent Core and RAG Core on start so
    # they wake in parallel (advancerag_common/wake.py). Off by default.
    WAKE_DEPENDENCIES_ON_START: bool = False

    # Storage
    STORAGE_BACKEND: Literal["local", "s3"] = "local"
    LOCAL_STORAGE_PATH: str = "var/storage"
    # Set true only when LOCAL_STORAGE_PATH is a persistent, backed-up volume (e.g. a mounted disk).
    ALLOW_LOCAL_STORAGE_IN_PRODUCTION: bool = False
    STORAGE_ENDPOINT: Optional[str] = None  # S3-compatible endpoint (R2, MinIO, ...); empty = AWS
    STORAGE_REGION: Optional[str] = None
    STORAGE_BUCKET: Optional[str] = None
    STORAGE_ACCESS_KEY: Optional[SecretStr] = None
    STORAGE_SECRET_KEY: Optional[SecretStr] = None

    # Uploads (images are not accepted until RAG Core has OCR/vision support)
    MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024
    ALLOWED_UPLOAD_EXTENSIONS: str = "pdf,txt,md,csv,docx"
    MAX_DOCUMENTS_PER_USER: int = 500
    MAX_STORAGE_BYTES_PER_USER: int = 2 * 1024 * 1024 * 1024
    MAX_JSON_BODY_BYTES: int = 1 * 1024 * 1024

    # Messages
    MAX_MESSAGE_CHARS: int = 32_000
    # Partial assistant output is flushed to the DB this often while streaming (crash safety).
    STREAM_FLUSH_SECONDS: float = 2.0
    # pending/streaming assistant messages older than this are marked failed (INTERRUPTED).
    STALE_GENERATION_SECONDS: int = 600

    # Background jobs
    WORKER_ENABLED: bool = True
    WORKER_CONCURRENCY: int = Field(default=2, ge=1, le=32)
    WORKER_POLL_INTERVAL_SECONDS: float = 5.0
    JOB_MAX_ATTEMPTS: int = 3
    DELETE_JOB_MAX_ATTEMPTS: int = 6
    # A running job whose heartbeat (locked_at) is older than this is treated as crashed. Ingestion
    # jobs heartbeat on every status poll, so a long but alive ingestion is never 'stale'.
    JOB_STALE_AFTER_SECONDS: int = 5 * 60
    # Document ingestion (runs in RAG Core's background; the job only starts it and polls).
    INGEST_POLL_INTERVAL_SECONDS: float = Field(default=2.0, gt=0, le=60)
    INGEST_MAX_SECONDS: int = Field(default=3600, ge=60)  # hard ceiling for one document
    INGEST_STATUS_ERROR_BUDGET_SECONDS: int = Field(default=120, ge=5)  # tolerate RAG Core blips
    INGEST_STALE_CONFIRM_SECONDS: float = Field(default=30.0, ge=0)  # a 'stale' attempt must stay stale this long
    RECONCILE_INTERVAL_SECONDS: int = 600
    # How often running jobs are checked for a stopped heartbeat (cheap; separate from the 10-min
    # housekeeping so a crashed worker's job is recovered within ~JOB_STALE_AFTER + this).
    STALE_SWEEP_INTERVAL_SECONDS: float = Field(default=60.0, gt=0)

    # Rate limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_BACKEND: Literal["memory", "redis"] = "memory"
    REDIS_URL: Optional[SecretStr] = None

    # Email (password-reset links only; sign-up and login are email + password)
    #   resend  — Resend HTTP API (production): EMAIL_API_KEY = the Resend API key, EMAIL_FROM =
    #             "AdvancedRAG <no-reply@your-verified-domain>"
    #   smtp    — any SMTP server (Gmail with an app password, Outlook, Brevo, Mailgun, ...)
    #   file    — development only: messages are written to EMAIL_OUTBOX_PATH
    #   mock    — automated tests only: recorded in memory, never sent
    #   console — nothing is sent (default outside development when unset)
    # Reset links point at FRONTEND_PUBLIC_URL (or the first FRONTEND_ORIGIN).
    EMAIL_PROVIDER: Literal["console", "file", "mock", "resend", "smtp", ""] = ""
    EMAIL_API_KEY: Optional[SecretStr] = None  # backend only — never a VITE_* variable
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[SecretStr] = None
    # "starttls" (port 587), "ssl" (port 465) or "none" (local relays only)
    SMTP_SECURITY: Literal["starttls", "ssl", "none"] = "starttls"
    EMAIL_FROM: str = "AdvancedRAG <no-reply@localhost>"
    EMAIL_OUTBOX_PATH: str = "var/outbox"

    @field_validator(
        "COOKIE_SECURE", "FRONTEND_PUBLIC_URL", "REDIS_URL", "EMAIL_API_KEY", "INTERNAL_SERVICE_TOKEN", "COOKIE_DOMAIN",
        "STORAGE_ENDPOINT", "STORAGE_REGION", "STORAGE_BUCKET", "STORAGE_ACCESS_KEY", "STORAGE_SECRET_KEY",
        "SMTP_HOST", "SMTP_USERNAME", "SMTP_PASSWORD",
        "READINESS_REQUIRE_SERVICES", mode="before",
    )
    @classmethod
    def _blank_to_none(cls, v):
        return None if isinstance(v, str) and v.strip() == "" else v

    def check_safety(self) -> "Settings":
        """Fail fast on unsafe deployments. Deliberately not a pydantic validator: pydantic
        model errors echo every input value, which would print secrets into startup logs."""
        if self.ENVIRONMENT not in ("staging", "production"):
            return self
        problems: List[str] = []
        for name in ("JWT_SECRET", "JWT_REFRESH_SECRET"):
            value = getattr(self, name).get_secret_value()
            if len(value) < 32 or any(m in value.lower() for m in INSECURE_SECRET_MARKERS):
                problems.append(f"{name} must be a strong random value (>= 32 chars)")
        if self.JWT_SECRET.get_secret_value() == self.JWT_REFRESH_SECRET.get_secret_value():
            problems.append("JWT_SECRET and JWT_REFRESH_SECRET must differ")
        if self.database_url.startswith("sqlite"):
            problems.append("DATABASE_URL must be PostgreSQL")
        elif (db_tls := tls_problem(self.database_url)):
            problems.append(db_tls)
        if "*" in self.cors_origins:
            problems.append("wildcard CORS origins are not allowed with credentials")
        if self.COOKIE_SAMESITE == "none" and not self.cookie_secure:
            problems.append("SameSite=None cookies require COOKIE_SECURE=true")
        if not self.internal_token or len(self.internal_token) < 32:
            problems.append("INTERNAL_SERVICE_TOKEN (>= 32 chars) is required")
        if self.ENVIRONMENT == "production":
            for name in ("AGENT_CORE_URL", "RAG_CORE_URL"):
                if (urlparse(getattr(self, name)).hostname or "") in LOCAL_HOSTS:
                    problems.append(f"{name} points to localhost")
        if self.RATE_LIMIT_BACKEND != "redis" or not self.REDIS_URL:
            problems.append("RATE_LIMIT_BACKEND=redis with REDIS_URL is required (in-memory limits are per process)")
        if self.EMAIL_PROVIDER in ("smtp", "resend") and not self.email_configured:
            problems.append("EMAIL_PROVIDER=smtp needs SMTP_HOST/SMTP_USERNAME/SMTP_PASSWORD; resend needs EMAIL_API_KEY")
        if self.EMAIL_PROVIDER in ("file", "mock"):
            problems.append(f"EMAIL_PROVIDER={self.EMAIL_PROVIDER} is for development/tests only; use resend or smtp")
        if self.EMAIL_PROVIDER in ("smtp", "resend"):
            sender_domain = self.EMAIL_FROM.strip().rstrip(">").rpartition("@")[2].lower()
            if not sender_domain or "." not in sender_domain or sender_domain in LOCAL_HOSTS:
                problems.append("EMAIL_FROM must use your verified sending domain, e.g. AdvancedRAG <no-reply@example.com>")
            if urlparse(self.frontend_url).scheme != "https":
                problems.append("password-reset links need an https FRONTEND_PUBLIC_URL (or FRONTEND_ORIGIN)")
        if self.STORAGE_BACKEND == "s3":
            if not (self.STORAGE_BUCKET and self.STORAGE_ACCESS_KEY and self.STORAGE_SECRET_KEY):
                problems.append("STORAGE_BUCKET, STORAGE_ACCESS_KEY and STORAGE_SECRET_KEY are required for s3 storage")
        elif not self.ALLOW_LOCAL_STORAGE_IN_PRODUCTION:
            problems.append("STORAGE_BACKEND=local loses files on redeploy; use s3 or set ALLOW_LOCAL_STORAGE_IN_PRODUCTION=true for a persistent volume")
        if problems:
            raise ConfigError(f"Unsafe {self.ENVIRONMENT} configuration: " + "; ".join(problems))
        return self

    @property
    def email_configured(self) -> bool:
        """A provider that actually delivers mail is fully configured."""
        if self.EMAIL_PROVIDER == "smtp":
            return bool(self.SMTP_HOST and self.SMTP_USERNAME and self.SMTP_PASSWORD)
        if self.EMAIL_PROVIDER == "resend":
            return bool(self.EMAIL_API_KEY)
        return False

    @property
    def database_url(self) -> str:
        return self.DATABASE_URL.get_secret_value()

    @property
    def internal_token(self) -> Optional[str]:
        return self.INTERNAL_SERVICE_TOKEN.get_secret_value() if self.INTERNAL_SERVICE_TOKEN else None

    @property
    def readiness_requires_services(self) -> bool:
        if self.READINESS_REQUIRE_SERVICES is not None:
            return self.READINESS_REQUIRE_SERVICES
        return self.ENVIRONMENT in ("staging", "production")

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip().rstrip("/") for o in self.FRONTEND_ORIGIN.split(",") if o.strip()]

    @property
    def frontend_url(self) -> str:
        return (self.FRONTEND_PUBLIC_URL or self.cors_origins[0]).rstrip("/")

    @property
    def cookie_secure(self) -> bool:
        if self.COOKIE_SECURE is not None:
            return self.COOKIE_SECURE
        return self.ENVIRONMENT not in ("development", "test")

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @cached_property
    def allowed_extensions(self) -> frozenset:
        return frozenset(e.strip().lower().lstrip(".") for e in self.ALLOWED_UPLOAD_EXTENSIONS.split(",") if e.strip())


settings = Settings().check_safety()
