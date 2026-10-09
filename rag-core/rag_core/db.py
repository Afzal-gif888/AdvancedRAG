import uuid
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, MetaData, String, Text, event
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from advancerag_common.db import async_engine_args


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)
            if dialect.name == "sqlite":
                value = value.replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):
        return value.replace(tzinfo=timezone.utc) if value is not None and value.tzinfo is None else value


JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_N_name)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class IndexedDocument(Base):
    __tablename__ = "rag_documents"

    document_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    filename: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(100))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    # Highest chunk count ever indexed: lets deletes remove every vector id that may exist.
    max_chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[Optional[int]] = mapped_column(Integer)
    embedding_model: Mapped[str] = mapped_column(String(100))
    embedding_dimensions: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow, onupdate=utcnow)


class Chunk(Base):
    __tablename__ = "rag_chunks"
    __table_args__ = (Index("ix_rag_chunks_user_id_document_id", "user_id", "document_id"),)

    # Deterministic "<document_id>:<n>" — also the vector id, so re-ingestion overwrites.
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("rag_documents.document_id", ondelete="CASCADE"))
    user_id: Mapped[uuid.UUID] = mapped_column()
    chunk_index: Mapped[int] = mapped_column(Integer)
    page: Mapped[Optional[int]] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    # Only populated for VECTOR_STORE=local (dev/test).
    embedding: Mapped[Optional[list]] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class TableData(Base):
    """The complete parsed CSV of a document, for exact whole-dataset analysis (tables.py).
    Written in the same transaction as the document's chunks; removed with the document."""

    __tablename__ = "rag_tables"

    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("rag_documents.document_id", ondelete="CASCADE"), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    columns: Mapped[list] = mapped_column(JSONType)
    rows: Mapped[list] = mapped_column(JSONType)
    row_count: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)


class IngestionAttempt(Base):
    """One row per document: the current/last ingestion attempt. Doubles as the per-document lock
    (one active attempt per document, enforced by compare-and-set on attempt_id) and records the
    vector extent written so far, so any partial Pinecone data can always be found and removed."""

    __tablename__ = "rag_ingestions"

    document_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(index=True)
    attempt_id: Mapped[uuid.UUID] = mapped_column()
    status: Mapped[str] = mapped_column(String(20))  # running | succeeded | failed | cancelled
    stage: Mapped[Optional[str]] = mapped_column(String(40))
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    # Highest vector index count this document may have in the vector store (all attempts).
    vector_extent: Mapped[int] = mapped_column(Integer, default=0)
    error_code: Mapped[Optional[str]] = mapped_column(String(64))
    error_message: Mapped[Optional[str]] = mapped_column(String(300))
    retryable: Mapped[bool] = mapped_column(default=False)
    timings: Mapped[Optional[dict]] = mapped_column(JSONType)
    heartbeat_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, default=utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime)


def chunk_id(document_id: uuid.UUID, index: int) -> str:
    return f"{document_id}:{index}"


engine: Optional[AsyncEngine] = None
SessionLocal: Optional[async_sessionmaker[AsyncSession]] = None


def configure_engine(url: str, **kwargs) -> AsyncEngine:
    global engine, SessionLocal
    # Accepts provider URLs as-is (Neon/Railway: postgresql://…?sslmode=require, pooled endpoints).
    url, connect_args = async_engine_args(url)
    options = {"pool_pre_ping": True}
    if connect_args:
        options["connect_args"] = {**connect_args, **kwargs.pop("connect_args", {})}
    if not url.startswith("sqlite") and "poolclass" not in kwargs:
        options.update(pool_size=5, max_overflow=10)
    options.update(kwargs)
    engine = create_async_engine(url, **options)
    if url.startswith("sqlite"):
        @event.listens_for(engine.sync_engine, "connect")
        def _fk(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    return engine


async def get_db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session
