"""Embedding providers — all local, no external embedding API.

- local: fastembed (ONNX, CPU) running a sentence-embedding model inside RAG Core.
         Default BAAI/bge-small-en-v1.5 -> 384-dimensional, unit-normalised vectors.
         Documents use `passage_embed`, queries use `query_embed` (bge's query instruction),
         so ingestion and retrieval always share the same model and vector space.
- fake:  deterministic hashing vectors for automated tests only (refused in production).

The vector dimension is a property of the model, not a free setting: it is read from the
model catalog, checked against EMBEDDING_DIMENSIONS when that is set, and verified on every
batch. The Pinecone index must have exactly this dimension.
"""
import hashlib
import logging
import math
import threading
from abc import ABC, abstractmethod
from typing import List, Optional

import anyio

from advancerag_common.observability import timed
from rag_core import errors
from rag_core.config import ConfigError, Settings

log = logging.getLogger("advancerag.rag.embeddings")

DOCUMENT = "document"
QUERY = "query"


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class EmbeddingProvider(ABC):
    model: str
    dimensions: int
    # Longest input (in model tokens) embedded without truncation; None = no limit to enforce.
    max_tokens: Optional[int] = None

    @abstractmethod
    async def embed(self, texts: List[str], task: str) -> List[List[float]]: ...

    def is_ready(self) -> bool:
        return True

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    async def warm_up(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def model_dimensions(model: str) -> Optional[int]:
    """Output dimension from fastembed's model catalog (no download)."""
    from fastembed import TextEmbedding

    for spec in TextEmbedding.list_supported_models():
        if spec.get("model", "").lower() == model.lower():
            return int(spec["dim"])
    return None


class LocalEmbeddings(EmbeddingProvider):
    def __init__(self, *, model: str, cache_dir: Optional[str], batch_size: int = 32, threads: Optional[int] = None,
                 expected_dimensions: Optional[int] = None):
        dims = model_dimensions(model)
        if dims is None:
            raise ConfigError(f"EMBEDDING_MODEL {model!r} is not a fastembed text-embedding model")
        if expected_dimensions and expected_dimensions != dims:
            raise ConfigError(
                f"EMBEDDING_DIMENSIONS={expected_dimensions} but {model} produces {dims}-dimensional vectors; "
                "remove EMBEDDING_DIMENSIONS or set it to the model's dimension"
            )
        self.model, self.dimensions = model, dims
        self.max_tokens = 512  # BERT-family limit; replaced by the tokenizer's own value on load
        self._cache_dir, self._batch_size, self._threads = cache_dir, batch_size, threads
        self._engine = None
        self._counter = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._engine is None:
                from fastembed import TextEmbedding
                from tokenizers import Tokenizer

                log.info("embedding_model_loading", extra={"model": self.model})
                engine = TextEmbedding(self.model, cache_dir=self._cache_dir, threads=self._threads)
                tokenizer = getattr(getattr(engine, "model", None), "tokenizer", None)
                if tokenizer is not None:
                    self.max_tokens = int((tokenizer.truncation or {}).get("max_length") or self.max_tokens)
                    # Separate, non-truncating copy so counting never affects embedding calls.
                    self._counter = Tokenizer.from_str(tokenizer.to_str())
                    self._counter.no_truncation()
                    self._counter.no_padding()
                self._engine = engine
                log.info("embedding_model_loaded", extra={"model": self.model, "dimensions": self.dimensions, "max_tokens": self.max_tokens})
        return self._engine

    def count_tokens(self, text: str) -> int:
        """Model tokens including special tokens; falls back to a conservative estimate."""
        self._load()
        if self._counter is None:
            return len(text) // 2
        return len(self._counter.encode(text).ids)

    def is_ready(self) -> bool:
        return self._engine is not None

    async def warm_up(self) -> None:
        """Download (first run only) and load the model, off the event loop."""
        await anyio.to_thread.run_sync(self._load)

    def _run(self, texts: List[str], task: str) -> List[List[float]]:
        engine = self._load()
        fn = engine.query_embed if task == QUERY else engine.passage_embed
        # Batches are padded to their longest text, so similar lengths per batch means less wasted
        # compute: 22-24% faster, measured. Each text's vector does not depend on its batch (verified
        # bit-identical), and results are returned in the caller's order.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        vectors = [v.tolist() for v in fn([texts[i] for i in order], batch_size=self._batch_size)]
        if len(vectors) != len(texts) or any(len(v) != self.dimensions for v in vectors):
            raise errors.ServiceError(errors.EMBEDDING_REJECTED, "Embedding output does not match the model dimension.", 502)
        result: List[Optional[List[float]]] = [None] * len(texts)
        for position, i in enumerate(order):
            result[i] = _normalize(vectors[position])
        return result  # type: ignore[return-value]

    async def embed(self, texts: List[str], task: str) -> List[List[float]]:
        if not texts:
            return []
        async with timed("embedding", provider="local", task=task):
            try:
                return await anyio.to_thread.run_sync(self._run, texts, task)
            except errors.ServiceError:
                raise
            except Exception as exc:  # model download/load failures, ONNX runtime errors
                log.exception("embedding_failed")
                raise errors.embedding_unavailable() from exc


class FakeEmbeddings(EmbeddingProvider):
    """Deterministic bag-of-words hashing vectors: similar text -> similar vectors. Lets the
    pipeline run in tests without downloading a model. Refused in production."""

    def __init__(self, dimensions: int = 384):
        self.model, self.dimensions = "fake-hash", dimensions

    async def embed(self, texts: List[str], task: str) -> List[List[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dimensions
            for word in text.lower().split():
                word = "".join(ch for ch in word if ch.isalnum())
                if word:
                    vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimensions] += 1.0
            vectors.append(_normalize(vec))
        return vectors


def build_embeddings(cfg: Settings) -> EmbeddingProvider:
    if cfg.EMBEDDING_PROVIDER == "fake":
        return FakeEmbeddings(cfg.EMBEDDING_DIMENSIONS or 384)
    return LocalEmbeddings(
        model=cfg.EMBEDDING_MODEL,
        cache_dir=cfg.EMBEDDING_CACHE_DIR,
        batch_size=cfg.EMBEDDING_BATCH_SIZE,
        threads=cfg.EMBEDDING_THREADS,
        expected_dimensions=cfg.EMBEDDING_DIMENSIONS,
    )
