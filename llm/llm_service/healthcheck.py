"""Diagnose the LLM provider connection without revealing secrets.

    python -m llm_service.healthcheck          # config + reachability + auth
    python -m llm_service.healthcheck --chat   # also run a tiny real chat completion
"""
import argparse
import asyncio
import sys
from urllib.parse import urlparse


def _safe_url(url: str) -> str:
    """Scheme, host, port and path only — drops any user:password@ or query string."""
    p = urlparse(url)
    return f"{p.scheme}://{p.hostname or ''}{f':{p.port}' if p.port else ''}{p.path}"


def _line(label: str, value: str) -> None:
    print(f"{label + ':':<18}{value}")


async def _run(chat: bool) -> int:
    try:
        from llm_service.config import load_settings

        cfg = load_settings()
    except Exception as exc:  # validation errors never contain secret values
        print(f"Configuration:    INVALID\n  {exc}")
        return 2

    from llm_service.providers import build_provider

    _line("LLM Provider", "FreeLLMAPI" if cfg.LLM_PROVIDER == "freellmapi" else cfg.LLM_PROVIDER)
    _line("Environment", cfg.ENVIRONMENT)
    _line("Base URL", _safe_url(cfg.freellmapi_base_url))
    _line("Model", cfg.FREELLMAPI_MODEL)
    _line("API key", "configured" if cfg.FREELLMAPI_API_KEY else "MISSING (set FREELLMAPI_API_KEY in llm/.env)")

    provider = build_provider(cfg)
    try:
        status = await provider.health_check(deep=chat)
    finally:
        await provider.aclose()

    _line("Connection", "OK" if status.reachable else "FAILED")
    _line("Authentication", "OK" if status.authenticated else ("FAILED" if status.reachable else "not checked"))
    if status.valid_response is False:
        _line("Models endpoint", "INVALID (not OpenAI-compatible JSON)")
    elif status.models_available is not None:
        _line("Models endpoint", f"OK ({status.models_available} models)")
    if status.model_available is not None:
        _line("Configured model", f"{cfg.FREELLMAPI_MODEL}: {'available' if status.model_available else 'NOT AVAILABLE'}")
    if chat:
        _line("Chat completion", "OK" if status.completion_ok else "FAILED")
    if status.detail:
        _line("Detail", status.detail)
    return 0 if status.ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chat", action="store_true", help="also send a minimal chat completion")
    args = parser.parse_args()
    sys.exit(asyncio.run(_run(args.chat)))


if __name__ == "__main__":
    main()
