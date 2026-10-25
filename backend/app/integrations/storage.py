"""Object storage abstraction. Keys are always server-generated; local paths are confined
to the storage root so a key can never escape it.
"""
import hashlib
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, BinaryIO, Optional, Protocol

import anyio

from app.core.config import settings

CHUNK = 64 * 1024


class Storage(Protocol):
    # `db` is the caller's open transaction: the database backend writes into it, so the object is committed
    # atomically with the document row and its job. Other backends ignore it.
    async def put(self, key: str, fileobj: BinaryIO, db=None) -> None: ...
    async def read_bytes(self, key: str) -> bytes: ...
    def stream(self, key: str) -> AsyncIterator[bytes]: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...


def make_storage_key(user_id: uuid.UUID, document_id: uuid.UUID) -> str:
    # No client-controlled component at all.
    return f"documents/{user_id}/{document_id}"


class LocalStorage:
    def __init__(self, root: str):
        self.root = Path(root).resolve()

    def _path(self, key: str) -> Path:
        # Confinement is checked lexically (no filesystem access): `..` and absolute keys are normalised away and then rejected.
        # Path.resolve() was used before; on Windows it can return the extended-length `\\?\C:\...` form while another request
        # is creating the directory, which made a valid key fail under concurrent first uploads (found by scripts/load_test.py).
        # Keys are server-generated and nothing can create links inside the storage root, so symlink resolution adds nothing.
        base = os.path.normcase(os.path.normpath(str(self.root)))
        full = os.path.normpath(os.path.join(str(self.root), key))
        norm = os.path.normcase(full)
        if not norm.startswith(base + os.sep):  # must name something inside the root, never the root itself
            raise ValueError("Invalid storage key")
        return Path(full)

    async def put(self, key: str, fileobj: BinaryIO, db=None) -> None:
        path = self._path(key)

        def _write():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".part")
            fileobj.seek(0)
            with open(tmp, "wb") as out:
                shutil.copyfileobj(fileobj, out, CHUNK)
            os.replace(tmp, path)

        await anyio.to_thread.run_sync(_write)

    async def read_bytes(self, key: str) -> bytes:
        path = self._path(key)
        return await anyio.to_thread.run_sync(path.read_bytes)

    async def stream(self, key: str) -> AsyncIterator[bytes]:
        path = self._path(key)
        f = await anyio.to_thread.run_sync(open, path, "rb")
        try:
            while True:
                chunk = await anyio.to_thread.run_sync(f.read, CHUNK)
                if not chunk:
                    break
                yield chunk
        finally:
            await anyio.to_thread.run_sync(f.close)

    async def delete(self, key: str) -> None:
        path = self._path(key)
        await anyio.to_thread.run_sync(lambda: path.unlink(missing_ok=True))

    async def exists(self, key: str) -> bool:
        path = self._path(key)
        return await anyio.to_thread.run_sync(path.is_file)

    async def healthy(self) -> bool:
        def _probe():
            self.root.mkdir(parents=True, exist_ok=True)
            probe = self.root / ".ready-probe"
            probe.write_bytes(b"ok")
            probe.unlink()
            return True

        try:
            return await anyio.to_thread.run_sync(_probe)
        except OSError:
            return False


class S3Storage:
    """S3-compatible object storage (AWS S3, Cloudflare R2, MinIO, ...). boto3 is synchronous,
    so every call runs in a worker thread."""

    def __init__(self, *, bucket: str, access_key: str, secret_key: str, endpoint: Optional[str] = None, region: Optional[str] = None):
        import boto3
        from botocore.config import Config

        self.bucket = bucket
        self._s3 = boto3.client(
            "s3",
            endpoint_url=endpoint or None,
            # R2 and MinIO accept us-east-1 (R2 also "auto"); AWS needs the real region.
            region_name=region or ("us-east-1" if endpoint else None),
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(retries={"max_attempts": 3, "mode": "standard"}, signature_version="s3v4"),
        )

    async def put(self, key: str, fileobj: BinaryIO, db=None) -> None:
        def _put():
            fileobj.seek(0)
            self._s3.upload_fileobj(fileobj, self.bucket, key)

        await anyio.to_thread.run_sync(_put)

    async def read_bytes(self, key: str) -> bytes:
        return await anyio.to_thread.run_sync(lambda: self._s3.get_object(Bucket=self.bucket, Key=key)["Body"].read())

    async def stream(self, key: str) -> AsyncIterator[bytes]:
        body = await anyio.to_thread.run_sync(lambda: self._s3.get_object(Bucket=self.bucket, Key=key)["Body"])
        try:
            while True:
                chunk = await anyio.to_thread.run_sync(body.read, CHUNK)
                if not chunk:
                    break
                yield chunk
        finally:
            await anyio.to_thread.run_sync(body.close)

    async def delete(self, key: str) -> None:
        await anyio.to_thread.run_sync(lambda: self._s3.delete_object(Bucket=self.bucket, Key=key))

    async def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError

        def _head():
            try:
                self._s3.head_object(Bucket=self.bucket, Key=key)
                return True
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                    return False
                raise

        return await anyio.to_thread.run_sync(_head)

    async def healthy(self) -> bool:
        try:
            await anyio.to_thread.run_sync(lambda: self._s3.head_bucket(Bucket=self.bucket))
            return True
        except Exception:
            return False


class DatabaseStorage:
    """Objects as 1 MiB parts in PostgreSQL (see app/db/models/stored_object.py)."""

    @staticmethod
    def _owner(key: str) -> uuid.UUID:
        return uuid.UUID(key.split("/")[1])  # documents/<user_id>/<document_id>

    async def put(self, key: str, fileobj: BinaryIO, db=None) -> None:
        from sqlalchemy import delete, insert, update

        from app.db.models import StoredObject, StoredObjectPart
        from app.db.session import session_scope

        own = db is None
        session = session_scope() if own else db
        try:
            await session.execute(delete(StoredObjectPart).where(StoredObjectPart.key == key))
            await session.execute(delete(StoredObject).where(StoredObject.key == key))
            digest, size, idx = hashlib.sha256(), 0, 0
            fileobj.seek(0)
            await session.execute(insert(StoredObject).values(key=key, user_id=self._owner(key), size_bytes=0, sha256="", part_count=0))
            while True:
                chunk = await anyio.to_thread.run_sync(fileobj.read, settings.STORAGE_DB_PART_BYTES)
                if not chunk:
                    break
                await session.execute(insert(StoredObjectPart).values(key=key, idx=idx, data=chunk))
                digest.update(chunk)
                size += len(chunk)
                idx += 1
            await session.execute(update(StoredObject).where(StoredObject.key == key).values(size_bytes=size, sha256=digest.hexdigest(), part_count=idx))
            if own:
                await session.commit()
        except BaseException:
            if own:
                await session.rollback()
            raise
        finally:
            if own:
                await session.close()

    async def _meta(self, session, key: str):
        from sqlalchemy import select

        from app.db.models import StoredObject

        return (await session.execute(select(StoredObject.part_count, StoredObject.size_bytes).where(StoredObject.key == key))).first()

    async def read_bytes(self, key: str) -> bytes:
        return b"".join([chunk async for chunk in self.stream(key)])

    async def stream(self, key: str) -> AsyncIterator[bytes]:
        from sqlalchemy import select

        from app.db.models import StoredObjectPart
        from app.db.session import session_scope

        async with session_scope() as session:
            meta = await self._meta(session, key)
            if meta is None:
                raise FileNotFoundError(key)
            for idx in range(meta[0]):
                data = await session.scalar(select(StoredObjectPart.data).where(StoredObjectPart.key == key, StoredObjectPart.idx == idx))
                if data is None:
                    raise FileNotFoundError(f"{key} part {idx}")
                yield bytes(data)

    async def delete(self, key: str) -> None:
        from sqlalchemy import delete

        from app.db.models import StoredObject, StoredObjectPart
        from app.db.session import session_scope

        async with session_scope() as session:
            await session.execute(delete(StoredObjectPart).where(StoredObjectPart.key == key))  # explicit: SQLite FK cascades are optional
            await session.execute(delete(StoredObject).where(StoredObject.key == key))
            await session.commit()

    async def exists(self, key: str) -> bool:
        from app.db.session import session_scope

        async with session_scope() as session:
            return await self._meta(session, key) is not None

    async def healthy(self) -> bool:
        from sqlalchemy import text

        from app.db.session import session_scope

        try:
            async with session_scope() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False


_storage = None
_by_backend: dict = {}


def build_storage(backend: Optional[str] = None) -> Storage:
    backend = backend or settings.STORAGE_BACKEND
    if backend == "db":
        return DatabaseStorage()
    if backend == "s3":
        return S3Storage(
            bucket=settings.STORAGE_BUCKET,
            access_key=settings.STORAGE_ACCESS_KEY.get_secret_value(),
            secret_key=settings.STORAGE_SECRET_KEY.get_secret_value(),
            endpoint=settings.STORAGE_ENDPOINT,
            region=settings.STORAGE_REGION,
        )
    return LocalStorage(settings.LOCAL_STORAGE_PATH)


def get_storage() -> Storage:
    """The backend new uploads are written to."""
    global _storage
    if _storage is None:
        _storage = build_storage()
    return _storage


def get_storage_for(backend: Optional[str]) -> Storage:
    """The backend a stored document lives in (documents remember theirs; null/legacy = local). A storage
    installed with `set_storage` (tests) always wins."""
    if _storage is not None and getattr(_storage, "_forced", False):
        return _storage
    name = backend or "local"
    if name == settings.STORAGE_BACKEND:
        return get_storage()
    if name not in _by_backend:
        _by_backend[name] = build_storage(name)
    return _by_backend[name]


def set_storage(storage: Storage) -> None:
    global _storage
    _storage = storage
    try:
        storage._forced = True  # type: ignore[attr-defined]
    except Exception:
        pass
    _by_backend.clear()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@asynccontextmanager
async def local_copy(storage: Storage, key: str) -> AsyncIterator[Path]:
    """A path on local disk holding the object, without loading it into memory. Local storage is used in
    place (nothing is copied or deleted); other backends are streamed to a temporary file that is removed
    afterwards."""
    if isinstance(storage, LocalStorage):
        path = storage._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        yield path
        return
    fd, name = tempfile.mkstemp(prefix="obj-", dir=settings.TEMP_DIR or None)
    try:
        with os.fdopen(fd, "wb") as out:
            async for chunk in storage.stream(key):
                await anyio.to_thread.run_sync(out.write, chunk)
        yield Path(name)
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass
