"""The Alembic migration must produce exactly the schema the models describe."""
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine

import app.db.models  # noqa: F401
from app.db.base import Base


import pytest

from conftest import IS_PG


@pytest.mark.skipif(IS_PG, reason="on PostgreSQL the whole suite already runs on the migrated schema (see pg_schema)")
def test_migrations_match_models(tmp_path):
    db = tmp_path / "migrations.db"
    cfg = Config("alembic.ini")
    cfg.attributes["database_url"] = f"sqlite+aiosqlite:///{db}"
    cfg.attributes["skip_logging_config"] = True
    command.upgrade(cfg, "head")

    engine = create_engine(f"sqlite:///{db}")
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}), Base.metadata)
    engine.dispose()
    assert diff == []

    command.downgrade(cfg, "base")
