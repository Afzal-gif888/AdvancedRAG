import time
from typing import Any, AsyncIterator, Dict, Optional, cast

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from advancerag_common.db import async_engine_args

from app.core.config import settings
from app.core.context import db_time_var

# Created by configure_engine() when this module is imported (below); None only for that instant.
engine: AsyncEngine = cast(Any, None)
SessionLocal: async_sessionmaker[AsyncSession] = cast(Any, None)


def _instrument(sync_engine) -> None:
    """Accumulate per-request DB time/query count for the access log."""

    @event.listens_for(sync_engine, "before_cursor_execute")
    def _before(conn, cursor, statement, parameters, context, executemany):
        conn.info.setdefault("_query_start", []).append(time.perf_counter())

    @event.listens_for(sync_engine, "after_cursor_execute")
    def _after(conn, cursor, statement, parameters, context, executemany):
        started = conn.info["_query_start"].pop()
        acc = db_time_var.get()
        if acc is not None:
            acc[0] += time.perf_counter() - started
            acc[1] += 1


def configure_engine(url: Optional[str] = None, **kwargs) -> AsyncEngine:
    global engine, SessionLocal
    # Accepts provider URLs as-is (Neon/Railway: postgresql://…?sslmode=require, pooled endpoints).
    url, connect_args = async_engine_args(url or settings.database_url)
    options: Dict[str, Any] = {"echo": settings.DATABASE_ECHO, "pool_pre_ping": True}
    if connect_args:
        options["connect_args"] = {**connect_args, **kwargs.pop("connect_args", {})}
    if not url.startswith("sqlite") and "poolclass" not in kwargs:
        options.update(pool_size=settings.DATABASE_POOL_SIZE, max_overflow=settings.DATABASE_MAX_OVERFLOW)
    options.update(kwargs)
    engine = create_async_engine(url, **options)
    if url.startswith("sqlite"):
        @event.listens_for(engine.sync_engine, "connect")
        def _sqlite_fk(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    _instrument(engine.sync_engine)
    SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    return engine


configure_engine()


def session_scope() -> AsyncSession:
    """A new session for work outside a request (streaming bodies, workers)."""
    return SessionLocal()


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
