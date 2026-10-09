from typing import Optional

from llm_service.config import Settings, settings
from llm_service.providers.base import LLMProvider

_provider: Optional[LLMProvider] = None


def build_provider(cfg: Settings) -> LLMProvider:
    if cfg.LLM_PROVIDER == "freellmapi":
        from llm_service.providers.freellmapi import FreeLLMAPIProvider

        return FreeLLMAPIProvider(
            base_url=cfg.freellmapi_base_url,
            api_key=cfg.FREELLMAPI_API_KEY.get_secret_value() if cfg.FREELLMAPI_API_KEY else None,
            default_model=cfg.FREELLMAPI_MODEL,
            timeout=cfg.LLM_TIMEOUT_SECONDS,
            connect_timeout=cfg.LLM_CONNECT_TIMEOUT_SECONDS,
            max_retries=cfg.LLM_MAX_RETRIES,
            backoff=cfg.LLM_RETRY_BACKOFF_SECONDS,
            max_backoff=cfg.LLM_MAX_BACKOFF_SECONDS,
            max_retry_after=cfg.LLM_MAX_RETRY_AFTER_SECONDS,
            stream_usage=cfg.FREELLMAPI_STREAM_USAGE,
        )
    raise ValueError(f"Unknown LLM_PROVIDER {cfg.LLM_PROVIDER!r}")


def get_provider() -> LLMProvider:
    """FastAPI dependency (overridden in tests)."""
    global _provider
    if _provider is None:
        _provider = build_provider(settings)
    return _provider


async def close_provider() -> None:
    global _provider
    if _provider is not None:
        await _provider.aclose()
        _provider = None
