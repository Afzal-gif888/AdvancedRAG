"""Provider database URLs (Neon, Railway, RDS…) are accepted as-is and made asyncpg-safe."""
import pytest

from advancerag_common.db import async_engine_args, describe, tls_problem

NEON_POOLED = ("postgresql://alice:s3cr%40t@ep-cool-river-123456-pooler.eu-central-1.aws.neon.tech/neondb"
               "?sslmode=require&channel_binding=require")


def test_neon_pooled_url_is_normalised_for_asyncpg():
    url, args = async_engine_args(NEON_POOLED)
    assert url.startswith("postgresql+asyncpg://alice:s3cr%40t@ep-cool-river-123456-pooler.")  # password kept, still encoded
    assert "sslmode" not in url and "channel_binding" not in url
    assert "prepared_statement_cache_size=0" in url
    assert args["ssl"] == "require" and args["statement_cache_size"] == 0
    assert args["prepared_statement_name_func"]() != args["prepared_statement_name_func"]()  # unique names for PgBouncer


@pytest.mark.parametrize("raw,ssl", [
    ("postgres://u:p@db.example.com:5432/app?sslmode=verify-full", "verify-full"),
    ("postgresql://u:p@ep-x.neon.tech/db?sslmode=require", "require"),
    ("postgresql+asyncpg://u:p@127.0.0.1:5434/advancerag", None),
])
def test_schemes_and_tls_modes(raw, ssl):
    url, args = async_engine_args(raw)
    assert url.startswith("postgresql+asyncpg://") and args.get("ssl") == ssl
    assert "statement_cache_size" not in args  # not a pooler


def test_connect_timeout_and_explicit_pgbouncer_flag():
    url, args = async_engine_args("postgresql://u:p@pooler.example.com/db?connect_timeout=7&pgbouncer=true&sslmode=require")
    assert args["timeout"] == 7.0 and args["statement_cache_size"] == 0 and "pgbouncer" not in url


def test_sqlite_untouched_and_bad_sslmode_rejected():
    assert async_engine_args("sqlite+aiosqlite://") == ("sqlite+aiosqlite://", {})
    with pytest.raises(ValueError):
        async_engine_args("postgresql://u:p@h/db?sslmode=bogus")


@pytest.mark.parametrize("raw,problem", [
    ("postgresql://u:p@ep-x.neon.tech/db", True),                       # public host, no TLS
    ("postgresql://u:p@ep-x.neon.tech/db?sslmode=prefer", True),        # TLS optional is not enough
    ("postgresql://u:p@ep-x.neon.tech/db?sslmode=require", False),
    ("postgresql://u:p@postgres.railway.internal:5432/railway", False),  # Railway private network
    ("postgresql+asyncpg://u:p@postgres:5432/advancerag", False),        # docker service name
    ("postgresql+asyncpg://u:p@127.0.0.1:5434/advancerag", False),
])
def test_remote_databases_require_tls(raw, problem):
    assert bool(tls_problem(raw)) is problem


def test_describe_never_contains_credentials():
    text = describe(NEON_POOLED)
    assert "alice" not in text and "s3cr" not in text and "tls=require" in text and "pooled=True" in text


def test_production_settings_refuse_plaintext_remote_database():
    from app.core.config import ConfigError, Settings

    base = {
        "ENVIRONMENT": "production", "_env_file": None, "JWT_SECRET": "a" * 40, "JWT_REFRESH_SECRET": "b" * 40,
        "FRONTEND_ORIGIN": "https://app.example.com", "INTERNAL_SERVICE_TOKEN": "t" * 40, "AGENT_CORE_URL": "https://agent.internal",
        "RAG_CORE_URL": "https://rag.internal", "RATE_LIMIT_BACKEND": "redis", "REDIS_URL": "redis://redis:6379/0",
        "STORAGE_BACKEND": "s3", "STORAGE_BUCKET": "b", "STORAGE_ACCESS_KEY": "ak", "STORAGE_SECRET_KEY": "sk",
    }
    with pytest.raises(ConfigError) as exc:
        Settings(**base, DATABASE_URL="postgresql://u:dbpass@ep-x.neon.tech/db").check_safety()
    assert "sslmode=require" in str(exc.value) and "dbpass" not in str(exc.value)
    Settings(**base, DATABASE_URL=NEON_POOLED).check_safety()
