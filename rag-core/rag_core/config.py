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
    # Texts per model call. Measured (docs/rag-architecture.md, "Ingestion speed"): 4 is the fastest on CPU,
    # 26% faster than 32 on 12 cores and up to 4x faster on one thread; the vectors are identical.
    EMBEDDING_BATCH_SIZE: int = Field(default=4, ge=1, le=256)
    EMBEDDING_THREADS: Optional[int] = Field(default=None, ge=1, le=64)
    # Where the model files are downloaded on first use (~70 MB for bge-small).
    EMBEDDING_CACHE_DIR: str = "var/models"

    # Vector store. "local" keeps vectors in RAG Core's DB (brute-force cosine) — dev/test only.
    VECTOR_STORE: Literal["pinecone", "local"] = "pinecone"
    PINECONE_API_KEY: Optional[SecretStr] = None
    PINECONE_INDEX: Optional[str] = None
    PINECONE_INDEX_HOST: Optional[str] = None  # optional; resolved from the control plane otherwise
    PINECONE_API_URL: str = "https://api.pinecone.io"
    # Ingestion writes: 100-vector upsert requests sent this many at a time (one 500-vector slice =
    # 5 requests, so 5 is one round trip per slice instead of five), and each request/fetch/delete
    # is tried at most PINECONE_MAX_ATTEMPTS times (429/5xx/network errors only, exponential backoff).
    # A transient Pinecone error must not throw away the embedding work of a whole document.
    PINECONE_UPSERT_CONCURRENCY: int = Field(default=5, ge=1, le=8)
    PINECONE_MAX_ATTEMPTS: int = Field(default=3, ge=1, le=5)

    # The metric the Pinecone index must use. Verified against the index description (the code used to
    # check only the dimension).
    PINECONE_METRIC: str = "cosine"

    RERANKER: Literal["pinecone", "none"] = "pinecone"
    RERANK_MODEL: str = "bge-reranker-v2-m3"

    CHUNK_SIZE: int = Field(default=1200, ge=200, le=8000)
    CHUNK_OVERLAP: int = Field(default=200, ge=0, le=2000)
    MAX_FILE_BYTES: int = 30 * 1024 * 1024
    # Measured, not assumed: embedding dominates ingestion time (docs/rag-architecture.md "Ingestion
    # speed"), so raise this only after measuring memory, embedding throughput and Pinecone writes
    # on the target hardware (docs/verification/PERFORMANCE.md).
    MAX_CHUNKS_PER_DOCUMENT: int = 5000

    # ---- chunking strategy. "recursive" is the frozen baseline; the others are opt-in and only
    # affect documents ingested afterwards (a re-ingest is needed to apply them to old documents).
    CHUNKING_STRATEGY: Literal["recursive", "structure", "semantic"] = "recursive"
    # Embed "<section path>\n<chunk text>" instead of the bare chunk text (structure/semantic only).
    EMBED_SECTION_CONTEXT: bool = True
    SEMANTIC_BREAKPOINT_PERCENTILE: int = Field(default=90, ge=50, le=99)
    SEMANTIC_MIN_CHUNK_CHARS: int = Field(default=300, ge=50, le=4000)
    # Above this many sentences semantic chunking is skipped (recorded in the extraction report) and
    # the structure/recursive chunker is used, so a huge document cannot trigger a huge embedding bill.
    SEMANTIC_MAX_SENTENCES: int = Field(default=4000, ge=10, le=100_000)

    # ---- text extraction hygiene
    TEXT_CLEANUP: bool = True  # ligatures, soft hyphens, line-end hyphenation, repeated headers/footers
    UNWRAP_LINES: bool = True  # join hard-wrapped lines of PDF/OCR paragraphs (never lists, tables, code)
    MAX_PDF_PAGES: int = Field(default=2000, ge=1, le=100_000)
    # Table structure in text-layer PDFs and OCR output (rag_core/pdftables.py). off = unchanged plain text (the frozen
    # behaviour); lines = bordered tables only (cell boundaries come from the ruling lines); auto = bordered, then
    # borderless tables inferred from text alignment (heuristic). Never changes embeddings, only the text that is chunked.
    PDF_PARSE_TIMEOUT_SECONDS: float = Field(default=900.0, gt=0, le=14_400)  # native text extraction, whole document; later pages are reported unread
    PDF_TABLES: Literal["off", "lines", "auto"] = "off"
    PDF_TABLE_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0, le=1800)  # per document; pages after it keep plain text
    # DOCX/ZIP bomb guard: refuse a package whose members expand beyond these limits.
    MAX_ZIP_UNCOMPRESSED_BYTES: int = 300 * 1024 * 1024
    MAX_ZIP_RATIO: int = Field(default=200, ge=2, le=100_000)

    # ---- OCR (scanned PDFs and PNG/JPEG images). Off by default: it needs CPU and RAM that small
    # instances (Render free: 0.1 CPU / 512 MB) do not have. Enable per deployment after measuring.
    OCR_ENABLED: bool = False
    OCR_RENDER_DPI: int = Field(default=150, ge=72, le=300)
    OCR_MAX_PAGES_PER_DOCUMENT: int = Field(default=200, ge=1, le=5000)
    OCR_MAX_PIXELS: int = Field(default=12_000_000, ge=100_000, le=200_000_000)  # per rendered page
    OCR_PAGE_TIMEOUT_SECONDS: float = Field(default=120.0, gt=0, le=1800)
    OCR_DOCUMENT_TIMEOUT_SECONDS: float = Field(default=1800.0, gt=0, le=14_400)
    OCR_MIN_NATIVE_CHARS: int = Field(default=20, ge=1, le=1000)  # fewer native characters => OCR candidate
    OCR_MIN_CONFIDENCE: float = Field(default=0.5, ge=0, le=1)  # engine-reported; below => a warning
    IMAGE_MAX_PIXELS: int = Field(default=40_000_000, ge=100_000, le=500_000_000)
    # Where uploads are spooled while they are being processed (default: the system temp directory).
    SPOOL_DIR: Optional[str] = None

    RETRIEVAL_CANDIDATES: int = Field(default=20, ge=1, le=100)
    RETRIEVAL_TOP_K: int = Field(default=6, ge=1, le=50)
    RRF_K: int = 60
    # How the PostgreSQL keyword candidates are ranked. idf (default since the 2026-10-23 evaluation): the OR candidates re-scored with
    # BM25 using statistics of the user's own chunks. cover_density: the original ts_rank_cd, which ignores term rarity (on evaluation
    # dataset v1.2 it found NONE of the exact-identifier chunks in its top 20 and made the hybrid worse than dense search alone;
    # docs/verification/RETRIEVAL_EVALUATION.md). Roll back with KEYWORD_RANKING=cover_density. tf: ts_rank. and_then_or: AND first.
    KEYWORD_RANKING: Literal["cover_density", "idf", "tf", "and_then_or"] = "idf"
    RRF_KEYWORD_WEIGHT: float = Field(default=1.0, gt=0, le=10)  # weight of the keyword ranking in fusion; 1.0 = plain RRF
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
                     "EMBEDDING_DIMENSIONS", "EMBEDDING_THREADS", "SPOOL_DIR", mode="before")
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
