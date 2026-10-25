"""Migration 0005 is additive: existing rows keep their meaning, and the downgrade removes only what it added."""
import uuid

import pytest
from sqlalchemy import create_engine, inspect, text

from alembic import command
from alembic.config import Config


def _cfg(url: str) -> Config:
    cfg = Config("alembic.ini")
    cfg.attributes["database_url"] = url
    cfg.attributes["skip_logging_config"] = True
    return cfg


def test_upgrade_preserves_existing_documents_and_downgrade_is_clean(tmp_path):
    db = tmp_path / "mig.db"
    async_url = f"sqlite+aiosqlite:///{db}"
    sync = create_engine(f"sqlite:///{db}")
    cfg = _cfg(async_url)
    command.upgrade(cfg, "0004")

    user, doc = str(uuid.uuid4()), str(uuid.uuid4())
    with sync.begin() as c:
        cols = {r[1] for r in c.execute(text("PRAGMA table_info(users)"))}
        values = {"id": user, "email": "old@example.com", "full_name": "Old User", "password_hash": "x",
                  "created_at": "2026-01-01 00:00:00", "updated_at": "2026-01-01 00:00:00", "is_active": 1, "settings": "{}"}
        use = {k: v for k, v in values.items() if k in cols}
        c.execute(text(f"INSERT INTO users ({', '.join(use)}) VALUES ({', '.join(':' + k for k in use)})"), use)
        c.execute(text(
            "INSERT INTO documents (id, user_id, filename, original_filename, mime_type, size_bytes, checksum_sha256, storage_key, status, created_at, updated_at) "
            "VALUES (:id, :u, 'old.pdf', 'old.pdf', 'application/pdf', 10, 'abc', 'documents/u/d', 'ready', '2026-01-01 00:00:00', '2026-01-01 00:00:00')"),
            {"id": doc, "u": user})

    command.upgrade(cfg, "head")
    with sync.connect() as c:
        row = c.execute(text("SELECT status, storage_backend, version, logical_id, extraction, replaces_id FROM documents WHERE id = :id"), {"id": doc}).one()
        tables = set(inspect(c).get_table_names())
    assert tuple(row) == ("ready", "local", 1, None, None, None), "old rows keep their meaning: local storage, version 1, no report"
    assert {"stored_objects", "stored_object_parts"} <= tables

    command.downgrade(cfg, "0004")
    with sync.connect() as c:
        row = c.execute(text("SELECT status, filename FROM documents WHERE id = :id"), {"id": doc}).one()
        tables = set(inspect(c).get_table_names())
        cols = {col["name"] for col in inspect(c).get_columns("documents")}
    assert tuple(row) == ("ready", "old.pdf")
    assert "stored_objects" not in tables and not ({"storage_backend", "extraction", "version", "logical_id", "replaces_id", "superseded_at"} & cols)
    sync.dispose()
