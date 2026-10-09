"""RAG Core tests. Uses TEST_DATABASE_URL (PostgreSQL) when set, otherwise in-memory SQLite.
On PostgreSQL the schema comes from the real Alembic migration and the keyword search uses
the real full-text index."""
import os

TEST_DB = os.environ.get("TEST_DATABASE_URL", "sqlite+aiosqlite://")
os.environ.update(
    {
        "ADVANCERAG_ENV_FILE": "none",  # never read a developer .env in tests
        "ENVIRONMENT": "test",
        "DATABASE_URL": TEST_DB,
        "EMBEDDING_PROVIDER": "fake",
        "VECTOR_STORE": "local",
        "RERANKER": "none",
        "EMBEDDING_DIMENSIONS": "64",
        "INTERNAL_SERVICE_TOKEN": "",
        "LOG_LEVEL": "WARNING",
        "CHUNK_SIZE": "300",
        "CHUNK_OVERLAP": "50",
    }
)

import httpx  # noqa: E402
import pytest  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.pool import NullPool, StaticPool  # noqa: E402

from rag_core import db as rag_db  # noqa: E402

IS_PG = TEST_DB.startswith("postgresql")
MODEL_CACHE = __import__("pathlib").Path(__file__).resolve().parents[1] / "var" / "models"


@pytest.fixture(scope="session")
def migrated():
    if IS_PG:
        from alembic import command
        from alembic.config import Config

        cfg = Config("alembic.ini")
        cfg.attributes["database_url"] = TEST_DB
        cfg.attributes["skip_logging_config"] = True
        command.downgrade(cfg, "base")
        command.upgrade(cfg, "head")
    return True


@pytest.fixture(autouse=True)
async def database(migrated):
    if IS_PG:
        engine = rag_db.configure_engine(TEST_DB, poolclass=NullPool)
        async with engine.begin() as conn:
            await conn.execute(text("TRUNCATE rag_chunks, rag_documents, rag_ingestions CASCADE"))
    else:
        engine = rag_db.configure_engine(TEST_DB, poolclass=StaticPool, connect_args={"check_same_thread": False})
        async with engine.begin() as conn:
            await conn.run_sync(rag_db.Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def services():
    from rag_core.config import settings
    from rag_core.embeddings import FakeEmbeddings
    from rag_core.pipeline import Services
    from rag_core.rerank import NoopReranker
    from rag_core.vectorstore import LocalVectorStore

    return Services(settings, FakeEmbeddings(64), LocalVectorStore(), NoopReranker())


@pytest.fixture
async def client(services):
    from rag_core import main

    main.app.dependency_overrides[main.get_services] = lambda: services
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://rag") as c:
        yield c
    main.app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def pinecone_url():
    from mock_providers import ServerThread, free_port, pinecone

    port = free_port()
    with ServerThread(pinecone, port):
        yield f"http://127.0.0.1:{port}"


@pytest.fixture(scope="session")
def local_embeddings():
    """The real local model (downloaded once into rag-core/var/models, then cached)."""
    from rag_core.embeddings import LocalEmbeddings

    return LocalEmbeddings(model="BAAI/bge-small-en-v1.5", cache_dir=str(MODEL_CACHE))


@pytest.fixture
def real_provider_services(pinecone_url, local_embeddings):
    """Real local embeddings + the Pinecone adapter wired to the mock Pinecone server."""
    from mock_providers import PINECONE_KEY, state
    from rag_core.config import settings
    from rag_core.pipeline import Services
    from rag_core.rerank import PineconeReranker
    from rag_core.vectorstore import PineconeStore

    state.reset()
    return Services(
        settings,
        local_embeddings,
        PineconeStore(api_key=PINECONE_KEY, index="advancerag-test", dimensions=local_embeddings.dimensions, api_url=pinecone_url),
        PineconeReranker(api_key=PINECONE_KEY, model="bge-reranker-v2-m3", api_url=pinecone_url),
    )


def make_pdf(pages):
    """Minimal valid PDF with one text line per page (no external libraries)."""
    objs = []
    kids = []
    font_id = 3 + 2 * len(pages)
    for i, line in enumerate(pages):
        page_id, content_id = 3 + 2 * i, 4 + 2 * i
        kids.append(f"{page_id} 0 R")
        stream = f"BT /F1 12 Tf 72 712 Td ({line}) Tj ET".encode()
        objs.append((page_id, f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>".encode()))
        objs.append((content_id, b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"))
    objs = [(1, b"<< /Type /Catalog /Pages 2 0 R >>"), (2, f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode())] + objs
    objs.append((font_id, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"))
    objs.sort()
    out = bytearray(b"%PDF-1.4\n")
    offsets = {}
    for oid, body in objs:
        offsets[oid] = len(out)
        out += f"{oid} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for oid, _ in objs:
        out += f"{offsets[oid]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return bytes(out)
