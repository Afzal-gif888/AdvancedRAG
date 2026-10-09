"""Tests against real infrastructure. Skipped unless the services are provided:

    TEST_REDIS_URL=redis://127.0.0.1:6380/15
    TEST_S3_ENDPOINT=http://127.0.0.1:8333  TEST_S3_ACCESS_KEY=any  TEST_S3_SECRET_KEY=any

(`docker compose --profile storage up -d` starts Redis and an S3-compatible server locally.)
"""
import os
import uuid

import pytest

from app.core.config import settings
from conftest import PASSWORD, register_and_login, upload

REDIS_URL = os.environ.get("TEST_REDIS_URL")
S3 = {k: os.environ.get(f"TEST_S3_{k}") for k in ("ENDPOINT", "ACCESS_KEY", "SECRET_KEY")}
API = "/api/v1"


# ---------------------------------------------------------------- Redis rate limiting

@pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")
async def test_redis_counters_are_shared_between_instances():
    from app.middleware.rate_limit import RedisBackend

    a, b = RedisBackend(REDIS_URL), RedisBackend(REDIS_URL)  # two API processes
    key = f"test:{uuid.uuid4()}"
    counts = [(await a.hit(key, 60))[0], (await b.hit(key, 60))[0], (await a.hit(key, 60))[0]]
    assert counts == [1, 2, 3]
    _, ttl = await b.hit(key, 60)
    assert 0 < ttl <= 60
    assert await a.ping()
    await a.aclose()
    await b.aclose()


@pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL not set")
async def test_login_rate_limit_enforced_through_redis(client, monkeypatch):
    from app.middleware import rate_limit

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    backend = rate_limit.RedisBackend(REDIS_URL)
    await backend._redis.flushdb()
    rate_limit.set_backend(backend)
    email = f"redis-{uuid.uuid4().hex[:6]}@example.com"
    await client.post(f"{API}/auth/register", json={"email": email, "password": PASSWORD, "full_name": "Test User"})
    codes = [(await client.post(f"{API}/auth/login", json={"email": email, "password": "Wrong12345"})).status_code for _ in range(11)]
    assert codes[-1] == 429 and codes[:10] == [401] * 10
    await backend.aclose()


async def test_unreachable_redis_fails_closed_in_production(client, monkeypatch):
    from app.middleware import rate_limit

    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    rate_limit.set_backend(rate_limit.RedisBackend("redis://127.0.0.1:1/0"))  # nothing listens on port 1
    r = await client.post(f"{API}/auth/login", json={"email": "x@example.com", "password": "Wrong12345"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "RATE_LIMITER_UNAVAILABLE"


# ---------------------------------------------------------------- S3-compatible storage

@pytest.fixture
def s3_storage():
    if not all(S3.values()):
        pytest.skip("TEST_S3_* not set")
    import boto3

    from app.integrations.storage import S3Storage

    bucket = f"advancerag-test-{uuid.uuid4().hex[:8]}"
    raw = boto3.client("s3", endpoint_url=S3["ENDPOINT"], aws_access_key_id=S3["ACCESS_KEY"], aws_secret_access_key=S3["SECRET_KEY"], region_name="us-east-1")
    raw.create_bucket(Bucket=bucket)
    storage = S3Storage(bucket=bucket, access_key=S3["ACCESS_KEY"], secret_key=S3["SECRET_KEY"], endpoint=S3["ENDPOINT"])
    yield storage
    for obj in raw.list_objects_v2(Bucket=bucket).get("Contents", []):
        raw.delete_object(Bucket=bucket, Key=obj["Key"])
    raw.delete_bucket(Bucket=bucket)


async def test_s3_storage_roundtrip(s3_storage):
    import io

    key = f"documents/{uuid.uuid4()}/{uuid.uuid4()}"
    data = b"x" * (200 * 1024 + 7)
    await s3_storage.put(key, io.BytesIO(data))
    assert await s3_storage.exists(key)
    assert await s3_storage.read_bytes(key) == data
    assert b"".join([c async for c in s3_storage.stream(key)]) == data
    assert await s3_storage.healthy()
    await s3_storage.delete(key)
    assert not await s3_storage.exists(key)


async def test_upload_download_delete_through_api_with_s3(client, s3_storage, rag):
    from app.integrations.storage import set_storage
    from app.workers.runner import process_one

    set_storage(s3_storage)
    headers = await register_and_login(client)
    doc = (await upload(client, headers, "s3.txt", b"stored in object storage")).json()
    r = await client.get(f"{API}/documents/{doc['id']}/content", headers=headers)
    assert r.status_code == 200 and r.content == b"stored in object storage"
    await process_one(rag=rag, storage=s3_storage)
    assert rag.ingested[0]["content"] == b"stored in object storage"
    await client.delete(f"{API}/documents/{doc['id']}", headers=headers)
    await process_one(rag=rag, storage=s3_storage)
    assert (await client.get(f"/ready")).json()["checks"]["storage"] is True
