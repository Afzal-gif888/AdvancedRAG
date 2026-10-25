"""Storage preflight for operators: write, read back, verify and delete one tiny object in the backend the current
environment selects (STORAGE_BACKEND=local|s3|db), under a key that cannot collide with a document (`preflight/...`, or `documents/<an existing user>/preflight-...` for db). It never lists, reads, overwrites
or deletes any other object, never prints a secret, and only touches the database for `db` (one small row set that it
removes again).

Run it in the same environment as the backend (so it sees the same variables), BEFORE switching production:

    cd backend && python ../scripts/check_storage.py

Exit status 0 only if the round trip returned identical bytes. This proves credentials, bucket/database reachability and
permissions; it does not prove capacity, backups, or that the Blueprint has been updated.
"""
import asyncio
import hashlib
import io
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "backend"), str(ROOT / "shared")]


async def main() -> int:
    from app.core.config import settings
    from app.integrations.storage import build_storage

    backend = settings.STORAGE_BACKEND
    storage = build_storage(backend)
    key = f"preflight/{uuid.uuid4()}"
    if backend == "db":
        # The database backend ties every object to its owner (a foreign key to users), so the probe borrows the id of
        # an existing user (a read-only lookup) and uses a key no document can have.
        from sqlalchemy import select

        from app.db.models import User
        from app.db.session import session_scope

        async with session_scope() as db:
            owner = await db.scalar(select(User.id).limit(1))
        if owner is None:
            print("no user exists yet, so the database backend cannot be probed; register one account and run this again")
            return 2
        key = f"documents/{owner}/preflight-{uuid.uuid4()}"
    payload = os.urandom(300_000)  # more than one part for the db backend only if STORAGE_DB_PART_BYTES is small
    digest = hashlib.sha256(payload).hexdigest()
    print(f"backend={backend} environment={settings.ENVIRONMENT}")
    try:
        if backend == "db":
            from app.db.session import session_scope

            async with session_scope() as db:
                await storage.put(key, io.BytesIO(payload), db=db)
                await db.commit()
        else:
            await storage.put(key, io.BytesIO(payload))
        back = await storage.read_bytes(key)
        ok = hashlib.sha256(back).hexdigest() == digest and await storage.exists(key)
        print("write/read/verify:", "OK" if ok else "MISMATCH")
        return 0 if ok else 1
    except Exception as exc:  # report the class and message; credentials are SecretStr and never part of these
        print(f"FAILED: {type(exc).__name__}: {str(exc)[:300]}")
        return 1
    finally:
        try:
            await storage.delete(key)
            print("cleanup: preflight object removed")
        except Exception as exc:
            print(f"cleanup failed ({type(exc).__name__}); remove key {key} manually")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
