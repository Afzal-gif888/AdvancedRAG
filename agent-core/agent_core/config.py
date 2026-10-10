import os
from typing import Literal, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from advancerag_common.service import is_local_url



def _env_file():
    """`.env` in the working directory; ADVANCERAG_ENV_FILE overrides it ("none" disables, used by tests)."""
    value = os.environ.get("ADVANCERAG_ENV_FILE", ".env")
    return None if value == "none" else value

class ConfigError(ValueError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_env_file(), env_file_encoding="utf-8", extra="ignore")

    ENVIRONMENT: Literal["development", "staging", "production", "test"]
    LOG_LEVEL: str = "INFO"
    INTERNAL_SERVICE_TOKEN: Optional[SecretStr] = None

    RAG_CORE_URL: str = "http://localhost:8002"
    LLM_SERVICE_URL: str = "http://localhost:8003"

    # Model choices are routing hints for the LLM layer (e.g. FreeLLMAPI "auto", "auto:smart",
    # "auto:fast"); empty = the LLM layer's default.
    AGENT_MODEL: Optional[str] = None
    AGENT_FAST_MODEL: Optional[str] = None
    AGENT_TEMPERATURE: float = Field(default=0.3, ge=0, le=2)
    AGENT_MAX_OUTPUT_TOKENS: Optional[int] = Field(default=None, ge=1)
    QUERY_REWRITE_ENABLED: bool = True
    # Frozen RAG contract (docs/rag-architecture.md): requested top-k and the evidence cap mirror
    # RAG Core's RETRIEVAL_TOP_K and MAX_CONTEXT_CHARS. Agent Core never asks for more.
    RETRIEVAL_TOP_K: int = Field(default=6, ge=1, le=50)
    MAX_CONTEXT_CHARS: int = Field(default=12_000, ge=1000, le=12_000)
    # Conversation context sent to the LLM (Backend owns the full history).
    MAX_HISTORY_CHARS: int = 24_000
    MAX_HISTORY_MESSAGES: int = Field(default=12, ge=0, le=100)
    MAX_HISTORY_MESSAGE_CHARS: int = Field(default=4_000, ge=200, le=50_000)

    # Self-correcting RAG (grading.py, structured.py). Every check is one bounded fast-model call;
    # a check that fails or times out falls back safely (see grading.py).
    EVIDENCE_GRADING_ENABLED: bool = True
    ANSWER_VERIFICATION_ENABLED: bool = True
    TABLE_ANALYSIS_ENABLED: bool = True
    MAX_CORRECTION_ATTEMPTS: int = Field(default=1, ge=0, le=2)
    CHECK_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0, le=120)

    # Timeouts (seconds). Read timeouts bound the gap between bytes, so for streams they act as
    # an idle timeout; AGENT_TURN_TIMEOUT_SECONDS bounds a whole turn end to end.
    CONNECT_TIMEOUT_SECONDS: float = Field(default=5.0, gt=0, le=60)
    RAG_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0, le=600)
    LLM_TIMEOUT_SECONDS: float = Field(default=120.0, gt=0, le=600)
    AGENT_TURN_TIMEOUT_SECONDS: float = Field(default=300.0, gt=0, le=1800)
    # Hosts that put idle services to sleep (Render free): ping RAG Core and the LLM service on start
    # so they wake in parallel (advancerag_common/wake.py). Off by default.
    WAKE_DEPENDENCIES_ON_START: bool = False
    # Render only wakes a sleeping service for requests from outside Render: wake through this relay
    # (the Vercel frontend's /wake/<service-name> rewrites). Empty = ping the services directly.
    WAKE_RELAY_URL: Optional[str] = None

    @field_validator("INTERNAL_SERVICE_TOKEN", "AGENT_MODEL", "AGENT_FAST_MODEL", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    def check_safety(self) -> "Settings":
        if self.ENVIRONMENT in ("staging", "production"):
            problems = []
            if not self.INTERNAL_SERVICE_TOKEN or len(self.INTERNAL_SERVICE_TOKEN.get_secret_value()) < 32:
                problems.append("INTERNAL_SERVICE_TOKEN (>= 32 chars) is required")
            for name in ("RAG_CORE_URL", "LLM_SERVICE_URL"):
                if is_local_url(getattr(self, name)) and self.ENVIRONMENT == "production":
                    problems.append(f"{name} points to localhost")
            if problems:
                raise ConfigError("Unsafe Agent Core configuration: " + "; ".join(problems))
        return self

    @property
    def internal_token(self) -> Optional[str]:
        return self.INTERNAL_SERVICE_TOKEN.get_secret_value() if self.INTERNAL_SERVICE_TOKEN else None


def load_settings() -> Settings:
    return Settings().check_safety()


settings = load_settings()
