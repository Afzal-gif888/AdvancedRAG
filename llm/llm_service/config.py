import os
import re
from typing import Literal, Optional
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from advancerag_common.service import is_https, is_local_url

# The FreeLLMAPI desktop app's OpenAI-compatible endpoint. Development/test default only.
LOCAL_DEVELOPMENT_URL = "http://127.0.0.1:31415/v1"



def _env_file():
    """`.env` in the working directory; ADVANCERAG_ENV_FILE overrides it ("none" disables, used by tests)."""
    value = os.environ.get("ADVANCERAG_ENV_FILE", ".env")
    return None if value == "none" else value

class ConfigError(ValueError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_env_file(), env_file_encoding="utf-8", extra="ignore")

    # Required, no default: production must never silently run with development settings.
    ENVIRONMENT: Literal["development", "staging", "production", "test"]
    LOG_LEVEL: str = "INFO"
    INTERNAL_SERVICE_TOKEN: Optional[SecretStr] = None

    LLM_PROVIDER: Literal["freellmapi"] = "freellmapi"

    # FreeLLMAPI (OpenAI-compatible gateway). The key stays in this service only.
    # No default outside development/test: production must name its endpoint explicitly and
    # never falls back to the desktop app (see `freellmapi_base_url`).
    FREELLMAPI_BASE_URL: Optional[str] = None
    FREELLMAPI_API_KEY: Optional[SecretStr] = None
    # "auto" (router picks), "auto:<profile>" (e.g. auto:fast) or any model id the gateway lists.
    FREELLMAPI_MODEL: str = Field(default="auto", pattern=r"^[A-Za-z0-9._:/@-]{1,120}$")
    # Ask for usage in the final stream chunk (only if the gateway supports stream_options).
    FREELLMAPI_STREAM_USAGE: bool = False
    # Opt-in: allow plain http ONLY for Railway private-network hosts (*.railway.internal),
    # which never leave Railway's encrypted private network. Every other production URL must be
    # https, and localhost is always refused.
    FREELLMAPI_ALLOW_RAILWAY_PRIVATE_HTTP: bool = False
    # Opt-in: allow plain http ONLY for a single-label host (a service name on a private Docker
    # network, e.g. http://freellmapi:3001/v1 on a single-VM Compose deployment). Such names are
    # not public DNS names; anything with a dot, and localhost, stays refused.
    FREELLMAPI_ALLOW_PRIVATE_NETWORK_HTTP: bool = False

    LLM_TIMEOUT_SECONDS: float = Field(default=90.0, gt=0, le=600)
    LLM_CONNECT_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0, le=60)
    LLM_MAX_RETRIES: int = Field(default=2, ge=0, le=5)
    LLM_RETRY_BACKOFF_SECONDS: float = Field(default=1.0, ge=0, le=10)
    LLM_MAX_BACKOFF_SECONDS: float = Field(default=8.0, ge=0, le=30)
    LLM_MAX_RETRY_AFTER_SECONDS: float = Field(default=10.0, ge=0, le=60)

    @field_validator("INTERNAL_SERVICE_TOKEN", "FREELLMAPI_API_KEY", "FREELLMAPI_BASE_URL", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("FREELLMAPI_BASE_URL")
    @classmethod
    def _strip(cls, v: Optional[str]) -> Optional[str]:
        return v.strip().rstrip("/") if v else v

    @property
    def is_deployed(self) -> bool:
        return self.ENVIRONMENT in ("staging", "production")

    @property
    def freellmapi_base_url(self) -> str:
        """The configured endpoint; the desktop default only in development/test."""
        if self.FREELLMAPI_BASE_URL:
            return self.FREELLMAPI_BASE_URL
        if self.is_deployed:  # unreachable after check_safety(); never fall back to localhost
            raise ConfigError("LLM configuration invalid: FREELLMAPI_BASE_URL is required in production.")
        return LOCAL_DEVELOPMENT_URL

    def check_safety(self) -> "Settings":
        """Fail fast on unsafe configuration. Raised outside pydantic on purpose: a pydantic
        model-level error echoes every input value (including secrets) into logs. Messages
        name the setting and the rule, never a value."""
        problems = []
        url = self.FREELLMAPI_BASE_URL
        if url:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https") or not parsed.hostname:
                problems.append("FREELLMAPI_BASE_URL must be an absolute http(s) URL such as https://host/v1")
        if self.is_deployed:
            if not self.INTERNAL_SERVICE_TOKEN or len(self.INTERNAL_SERVICE_TOKEN.get_secret_value()) < 32:
                problems.append("INTERNAL_SERVICE_TOKEN (>= 32 chars) is required")
            if not self.FREELLMAPI_API_KEY:
                problems.append("FREELLMAPI_API_KEY is required")
            if not url:
                problems.append("FREELLMAPI_BASE_URL is required (an HTTPS endpoint of your hosted FreeLLMAPI)")
            elif is_local_url(url):
                problems.append(
                    f"FREELLMAPI_BASE_URL must be an HTTPS non-local endpoint in {self.ENVIRONMENT}: it points to "
                    "localhost, which a cloud deployment cannot reach (your desktop FreeLLMAPI is for development only)"
                )
            elif not is_https(url) and not self._railway_private_http_allowed(url):
                problems.append(f"FREELLMAPI_BASE_URL must be an HTTPS non-local endpoint in {self.ENVIRONMENT} (plain http is refused)")
        if problems:
            raise ConfigError("LLM configuration invalid: " + "; ".join(problems))
        return self

    def _railway_private_http_allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "http":
            return False
        if self.FREELLMAPI_ALLOW_RAILWAY_PRIVATE_HTTP and host.endswith(".railway.internal"):
            return True
        # A Docker service name: one DNS label, no dot (localhost was already refused above).
        return self.FREELLMAPI_ALLOW_PRIVATE_NETWORK_HTTP and bool(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", host))

    @property
    def internal_token(self) -> Optional[str]:
        return self.INTERNAL_SERVICE_TOKEN.get_secret_value() if self.INTERNAL_SERVICE_TOKEN else None


def load_settings() -> Settings:
    return Settings().check_safety()


settings = load_settings()
