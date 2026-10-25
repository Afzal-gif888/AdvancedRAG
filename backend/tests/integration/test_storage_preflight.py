"""scripts/check_storage.py: the operator's write/read/verify/delete round trip for the selected backend."""
import importlib.util
import io
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.core.config import settings
from app.db import session as db_session
from app.db.models import StoredObject
from app.integrations import storage as storage_mod

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check_storage.py"


def load_script():
    spec = importlib.util.spec_from_file_location("check_storage", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fresh(monkeypatch):
    storage_mod._storage = None
    storage_mod._by_backend.clear()
    yield
    storage_mod._storage = None
    storage_mod._by_backend.clear()


async def test_local_backend_round_trip_leaves_nothing_behind(monkeypatch, tmp_path, capsys, fresh):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "local")
    monkeypatch.setattr(settings, "LOCAL_STORAGE_PATH", str(tmp_path))
    assert await load_script().main() == 0
    out = capsys.readouterr().out
    assert "write/read/verify: OK" in out and "cleanup: preflight object removed" in out
    assert not [p for p in tmp_path.rglob("*") if p.is_file()]


async def test_db_backend_round_trip_leaves_no_rows(client, auth, monkeypatch, capsys, fresh):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "db")
    monkeypatch.setattr(settings, "STORAGE_DB_PART_BYTES", 64 * 1024)   # forces several parts
    assert await load_script().main() == 0
    assert "write/read/verify: OK" in capsys.readouterr().out
    async with db_session.session_scope() as db:
        assert await db.scalar(select(func.count()).select_from(StoredObject)) == 0


async def test_db_backend_without_any_user_says_so_instead_of_failing(client, monkeypatch, capsys, fresh):
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "db")
    assert await load_script().main() == 2
    assert "no user exists yet" in capsys.readouterr().out


async def test_s3_failure_is_reported_without_leaking_credentials(monkeypatch, capsys, fresh):
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "s3")
    monkeypatch.setattr(settings, "STORAGE_BUCKET", "no-such-bucket")
    monkeypatch.setattr(settings, "STORAGE_ACCESS_KEY", type(settings.JWT_SECRET)("AKIA-SECRET-ACCESS-KEY"))
    monkeypatch.setattr(settings, "STORAGE_SECRET_KEY", type(settings.JWT_SECRET)("super-secret-value-123"))
    monkeypatch.setattr(settings, "STORAGE_REGION", "us-east-1")
    with moto.mock_aws():
        assert await load_script().main() == 1          # the bucket does not exist
    out = capsys.readouterr().out
    assert "FAILED" in out and "AKIA-SECRET-ACCESS-KEY" not in out and "super-secret-value-123" not in out


async def test_s3_round_trip_against_a_mocked_bucket(monkeypatch, capsys, fresh):
    boto3 = pytest.importorskip("boto3")
    moto = pytest.importorskip("moto")
    monkeypatch.setattr(settings, "STORAGE_BACKEND", "s3")
    monkeypatch.setattr(settings, "STORAGE_BUCKET", "adv-preflight")
    monkeypatch.setattr(settings, "STORAGE_ACCESS_KEY", type(settings.JWT_SECRET)("x"))
    monkeypatch.setattr(settings, "STORAGE_SECRET_KEY", type(settings.JWT_SECRET)("y"))
    monkeypatch.setattr(settings, "STORAGE_REGION", "us-east-1")
    with moto.mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="adv-preflight")
        assert await load_script().main() == 0
        listing = boto3.client("s3", region_name="us-east-1").list_objects_v2(Bucket="adv-preflight")
        assert listing.get("KeyCount", 0) == 0
    assert "write/read/verify: OK" in capsys.readouterr().out
