"""Database URL handling shared by the backend and RAG Core (and their Alembic migrations).

Managed PostgreSQL (Neon, Railway, Supabase, RDS, ...) hands out libpq-style URLs such as

    postgresql://user:pass@ep-xyz-pooler.eu-central-1.aws.neon.tech/db?sslmode=require&channel_binding=require

The async driver (asyncpg) does not accept `sslmode`/`channel_binding` in the URL, and a
transaction-mode pooler (PgBouncer — Neon's "-pooler" endpoints) breaks asyncpg's cached prepared
statements. `async_engine_args()` turns any such URL into (sqlalchemy_url, connect_args):

- scheme postgres:// | postgresql:// | postgresql+psycopg2:// -> postgresql+asyncpg://
- sslmode=disable|allow|prefer|require|verify-ca|verify-full -> asyncpg `ssl=<mode>` (TLS)
- channel_binding / options / target_session_attrs: dropped (libpq-only)
- connect_timeout=N -> asyncpg `timeout=N`
- pooler (host contains "-pooler." or `pgbouncer=true`): prepared-statement caches disabled and
  statement names made unique, as SQLAlchemy documents for PgBouncer.

SQLite URLs (tests) pass through untouched. Credentials are never logged by this module.
"""
import uuid
from typing import Any, Dict, Tuple, Optional

from sqlalchemy.engine import make_url

_LIBPQ_ONLY = {"channel_binding", "options", "target_session_attrs", "application_name", "gssencmode", "sslcompression"}
_SSL_MODES = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}


def async_engine_args(url: str) -> Tuple[str, Dict[str, Any]]:
    if url.startswith("sqlite"):
        return url, {}
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql":
        return url, {}
    parsed = parsed.set(drivername="postgresql+asyncpg")
    query = dict(parsed.query)
    connect_args: Dict[str, Any] = {}

    sslmode = query.pop("sslmode", None) or query.pop("ssl", None)
    if sslmode:
        if sslmode not in _SSL_MODES:
            raise ValueError("Unsupported sslmode in DATABASE_URL (use require, verify-ca or verify-full)")
        connect_args["ssl"] = sslmode
    if "connect_timeout" in query:
        timeout = query.pop("connect_timeout")
        connect_args["timeout"] = float(timeout[-1] if isinstance(timeout, tuple) else timeout)  # a repeated parameter arrives as a tuple
    pooled = str(query.pop("pgbouncer", "")).lower() in ("1", "true", "yes") or "-pooler." in (parsed.host or "")
    for key in _LIBPQ_ONLY:
        query.pop(key, None)
    if pooled:
        query["prepared_statement_cache_size"] = "0"
        connect_args["statement_cache_size"] = 0
        connect_args["prepared_statement_name_func"] = lambda: f"__asyncpg_{uuid.uuid4().hex}__"
    parsed = parsed.set(query=query)
    return parsed.render_as_string(hide_password=False), connect_args


_PRIVATE_SUFFIXES = (".railway.internal", ".internal", ".local", ".svc", ".cluster.local")
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "postgres", "db"}


def tls_problem(url: str) -> Optional[str]:
    """Production rule: a database reached over a public network must use TLS. Private networks
    (Railway *.railway.internal, Docker service names, localhost) are exempt."""
    if url.startswith("sqlite"):
        return None
    try:
        sa_url, args = async_engine_args(url)
    except ValueError as exc:
        return str(exc)
    host = (make_url(sa_url).host or "").lower()
    private = host in _LOCAL_HOSTS or host.endswith(_PRIVATE_SUFFIXES) or "." not in host
    if not private and args.get("ssl") not in ("require", "verify-ca", "verify-full"):
        return "DATABASE_URL points to a remote host without TLS: add ?sslmode=require (or verify-full)"
    return None


def describe(url: str) -> str:
    """Safe, credential-free description for logs: driver, host, database, TLS, pooler."""
    try:
        sa_url, args = async_engine_args(url)
        u = make_url(sa_url)
        return f"{u.drivername}://{u.host or ''}/{u.database or ''} tls={args.get('ssl', 'off')} pooled={'statement_cache_size' in args}"
    except Exception:
        return "<unparseable DATABASE_URL>"
