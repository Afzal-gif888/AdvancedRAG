"""Guards the frozen RAG baseline (docs/rag-architecture.md). If one of these fails, a frozen value
changed: follow "Changing a frozen value" in that document (re-index + benchmark), then update
this test and the document together."""
from pathlib import Path

import pytest

FROZEN = {
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    "CHUNK_SIZE": 1200,
    "CHUNK_OVERLAP": 200,
    "RETRIEVAL_CANDIDATES": 20,
    "RETRIEVAL_TOP_K": 6,
    "RRF_K": 60,
    "RERANKER": "pinecone",
    "RERANK_MODEL": "bge-reranker-v2-m3",
    "MAX_CONTEXT_CHARS": 12_000,
    "MAX_CHUNKS_PER_DOCUMENT": 5000,
    "VECTOR_STORE": "pinecone",
    "EMBEDDING_PROVIDER": "local",
    # Added with the extraction work: the defaults keep the baseline behaviour (opt-in features are off).
    "CHUNKING_STRATEGY": "recursive",
    "OCR_ENABLED": False,
    "PINECONE_METRIC": "cosine",
}


@pytest.fixture
def defaults(monkeypatch):
    """Settings built from code defaults only (the test suite overrides several via env)."""
    from rag_core.config import Settings

    for name in FROZEN:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("EMBEDDING_DIMENSIONS", raising=False)
    return Settings(_env_file=None, ENVIRONMENT="test", DATABASE_URL="sqlite+aiosqlite://")


@pytest.mark.parametrize("name,value", sorted(FROZEN.items()))
def test_frozen_default(defaults, name, value):
    assert getattr(defaults, name) == value, f"{name} is frozen (docs/rag-architecture.md)"


def test_embedding_dimension_is_384_and_matches_the_index_name(defaults):
    from rag_core.embeddings import model_dimensions

    assert model_dimensions(defaults.EMBEDDING_MODEL) == 384  # catalog lookup, no download
    assert defaults.EMBEDDING_DIMENSIONS is None  # derived from the model, never hand-set


def test_docker_image_bakes_the_frozen_model():
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")
    assert f"ARG EMBEDDING_MODEL={FROZEN['EMBEDDING_MODEL']}" in dockerfile


def test_namespace_and_vector_id_scheme():
    import uuid

    from rag_core.db import chunk_id
    from rag_core.vectorstore import namespace_for

    u, d = uuid.UUID(int=1), uuid.UUID(int=2)
    assert namespace_for(u) == f"user-{u}"
    assert chunk_id(d, 7) == f"{d}:7"
