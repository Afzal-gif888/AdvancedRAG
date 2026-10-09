"""Object storage abstraction. Keys are always server-generated; local paths are confined
to the storage root so a key can never escape it.
"""
import os
import shutil
import uuid
from pathlib import Path
from typing import AsyncIterator, BinaryIO, Optional, Protocol

import anyio

from app.core.config import settings

CHUNK = 64 * 1024


class Storage(Protocol):
    async def put(self, key: str, fileobj: BinaryIO) -> None: ...
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
        path = (self.root / key).resolve()
        if self.root != path and self.root not in path.parents:
            raise ValueError("Invalid storage key")
        return path

    async def put(self, key: str, fileobj: BinaryIO) -> None:
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

    async def put(self, key: str, fileobj: BinaryIO) -> None:
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


_storage = None


def build_storage() -> Storage:
    if settings.STORAGE_BACKEND == "s3":
        return S3Storage(
            bucket=settings.STORAGE_BUCKET,
            access_key=settings.STORAGE_ACCESS_KEY.get_secret_value(),
            secret_key=settings.STORAGE_SECRET_KEY.get_secret_value(),
            endpoint=settings.STORAGE_ENDPOINT,
            region=settings.STORAGE_REGION,
        )
    return LocalStorage(settings.LOCAL_STORAGE_PATH)


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = build_storage()
    return _storage


def set_storage(storage: Storage) -> None:
    global _storage
    _storage = storage
