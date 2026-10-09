import os
from typing import Literal, Optional

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from advancerag_common.db import tls_problem



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

    # RAG Core's own database (chunk text, keyword index, ingestion bookkeeping).
    DATABASE_URL: SecretStr

    # Embeddings run locally (fastembed / ONNX, CPU) — no external embedding API or key.
    # Vectors are tied to the model: changing EMBEDDING_MODEL requires a Pinecone index with the
    # new dimension and re-ingesting every document.
    EMBEDDING_PROVIDER: Literal["local", "fake"] = "local"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    # Optional guard: if set, must equal the model's real output dimension (384 for bge-small).
    EMBEDDING_DIMENSIONS: Optional[int] = Field(default=None, ge=16, le=4096)
    EMBEDDING_BATCH_SIZE: int = Field(default=32, ge=1, le=256)
    EMBEDDING_THREADS: Optional[int] = Field(default=None, ge=1, le=64)
    # Where the model files are downloaded on first use (~70 MB for bge-small).
    EMBEDDING_CACHE_DIR: str = "var/models"

    # Vector store. "local" keeps vectors in RAG Core's DB (brute-force cosine) — dev/test only.
    VECTOR_STORE: Literal["pinecone", "local"] = "pinecone"
    PINECONE_API_KEY: Optional[SecretStr] = None
    PINECONE_INDEX: Optional[str] = None
    PINECONE_INDEX_HOST: Optional[str] = None  # optional; resolved from the control plane otherwise
    PINECONE_API_URL: str = "https://api.pinecone.io"

    RERANKER: Literal["pinecone", "none"] = "pinecone"
    RERANK_MODEL: str = "bge-reranker-v2-m3"

    CHUNK_SIZE: int = Field(default=1200, ge=200, le=8000)
    CHUNK_OVERLAP: int = Field(default=200, ge=0, le=2000)
    MAX_FILE_BYTES: int = 30 * 1024 * 1024
    MAX_CHUNKS_PER_DOCUMENT: int = 5000

    RETRIEVAL_CANDIDATES: int = Field(default=20, ge=1, le=100)
    RETRIEVAL_TOP_K: int = Field(default=6, ge=1, le=50)
    RRF_K: int = 60
    MAX_CONTEXT_CHARS: int = 12_000

    HTTP_TIMEOUT_SECONDS: float = 30.0

    # Ingestion lifecycle (orchestration only; nothing here changes chunking, embeddings or retrieval).
    # Ingestion runs as a background task per document, guarded by a per-document attempt record.
    INGEST_CONCURRENCY: int = Field(default=2, ge=1, le=8)          # ingestions running at once in this process
    INGEST_HEARTBEAT_SECONDS: float = Field(default=10.0, gt=0, le=60)
    INGEST_STALE_SECONDS: float = Field(default=120.0, gt=0, le=3600)  # no heartbeat this long -> attempt is dead
    INGEST_EMBED_SLICE: int = Field(default=256, ge=32, le=4096)     # progress/cancel checkpoint granularity
    INGEST_UPSERT_SLICE: int = Field(default=500, ge=100, le=5000)
    # Before an ingestion counts as successful, its vectors must be readable back from the store.
    INGEST_VERIFY_SECONDS: float = Field(default=20.0, ge=0, le=120)

    # CSV tables kept for exact whole-dataset analysis (tables.py). Larger files are still indexed
    # for search; only exact statistics are unavailable for them.
    TABLE_MAX_ROWS: int = Field(default=200_000, ge=100, le=2_000_000)
    TABLE_MAX_COLUMNS: int = Field(default=100, ge=2, le=1000)

    @field_validator("INTERNAL_SERVICE_TOKEN", "PINECONE_API_KEY", "PINECONE_INDEX", "PINECONE_INDEX_HOST",
                     "EMBEDDING_DIMENSIONS", "EMBEDDING_THREADS", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    def check_safety(self) -> "Settings":
        """Raised outside pydantic so secrets never appear in the error text."""
        problems = []
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            problems.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.ENVIRONMENT in ("staging", "production"):
            if not self.INTERNAL_SERVICE_TOKEN or len(self.INTERNAL_SERVICE_TOKEN.get_secret_value()) < 32:
                problems.append("INTERNAL_SERVICE_TOKEN (>= 32 chars) is required")
            if self.EMBEDDING_PROVIDER == "fake":
                problems.append("EMBEDDING_PROVIDER=fake is for tests only")
            if self.VECTOR_STORE == "local":
                problems.append("VECTOR_STORE=local is for development only")
            if self.database_url.startswith("sqlite"):
                problems.append("DATABASE_URL must be PostgreSQL")
            elif (db_tls := tls_problem(self.database_url)):
                problems.append(db_tls)
            if self.VECTOR_STORE == "pinecone" and (not self.PINECONE_API_KEY or not self.PINECONE_INDEX):
                problems.append("PINECONE_API_KEY and PINECONE_INDEX are required")
        if problems:
            raise ConfigError("Unsafe RAG Core configuration: " + "; ".join(problems))
        return self

    @property
    def database_url(self) -> str:
        return self.DATABASE_URL.get_secret_value()

    @property
    def internal_token(self) -> Optional[str]:
        return self.INTERNAL_SERVICE_TOKEN.get_secret_value() if self.INTERNAL_SERVICE_TOKEN else None

    @property
    def vector_namespace_prefix(self) -> str:
        return "user-"


def load_settings() -> Settings:
    return Settings().check_safety()


settings = load_settings()
