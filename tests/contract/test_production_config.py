"""render.yaml (and the prepared durable-storage overlays) must pass the services' own production fail-fast rules
with dummies for dashboard secrets, and the checker must actually fail on a broken blueprint."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "verify_production_config.py"
OVERLAYS = ROOT / "infrastructure" / "durable-storage"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=300)


def test_the_blueprint_is_accepted_by_every_service():
    r = run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("[OK]") == 4
    assert "STORAGE_BACKEND=local" in r.stdout, "the known durability risk must stay visible until it is switched"


@pytest.mark.parametrize("name", ["backend-db.overlay.yaml", "backend-s3.overlay.yaml"])
def test_the_prepared_durable_storage_overlays_pass_and_clear_the_warning(name):
    r = run("--overlay", str(OVERLAYS / name))
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("[OK]") == 4 and "STORAGE_BACKEND=local" not in r.stdout


def test_s3_without_its_credentials_is_rejected(tmp_path):
    overlay = tmp_path / "o.yaml"
    overlay.write_text("service: advancerag-backend\nset:\n  STORAGE_BACKEND: s3\nremove:\n  - ALLOW_LOCAL_STORAGE_IN_PRODUCTION\n", encoding="utf8")
    r = run("--overlay", str(overlay))
    assert r.returncode == 1 and "STORAGE_BUCKET" in r.stdout


def test_the_checker_rejects_a_typo_and_an_unsafe_value(tmp_path):
    text = (ROOT / "render.yaml").read_text(encoding="utf8")
    bad = text.replace("key: STORAGE_BACKEND", "key: STORAGE_BAKEND", 1).replace("value: https://app.afzaltech.in", 'value: "*"', 1)
    assert bad != text
    p = tmp_path / "render.yaml"
    p.write_text(bad, encoding="utf8")
    r = run(str(p))
    assert r.returncode == 1 and "STORAGE_BAKEND" in r.stdout and "[FAIL]" in r.stdout
