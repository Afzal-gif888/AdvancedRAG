"""Production rehearsal: build the four service images, run them with ENVIRONMENT=production
behind private service names (infrastructure/production-rehearsal/compose.yml), verify the whole
user path plus the production controls, then tear everything down (containers, volumes, the
rehearsal user's Pinecone vectors).

    python scripts/rehearse_production.py [--keep]

Secrets are generated per run and passed to docker compose through the environment only; they
are never printed or written to disk. PINECONE_API_KEY/PINECONE_INDEX are read from rag-core/.env.
The only stand-in is FreeLLMAPI (the mock gateway on a Railway-style private hostname).
"""
import argparse
import json
import os
import pathlib
import secrets
import subprocess
import sys
import time

import httpx

ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPOSE = ["docker", "compose", "-f", str(ROOT / "infrastructure" / "production-rehearsal" / "compose.yml")]
BASE = "http://127.0.0.1:8200"
API = BASE + "/api/v1"
FACT = "The Orion Z9 telescope mount carries a warranty of seven years and operates between -20 C and 45 C."

results: list = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def dotenv(path: pathlib.Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def compose(env: dict, *args: str, check_rc: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(COMPOSE + list(args), env=env, capture_output=capture, text=True, encoding="utf-8", errors="replace")
    if check_rc and r.returncode != 0:
        raise SystemExit(f"docker compose {' '.join(args[:2])} failed:\n{(r.stderr or '')[-2000:]}")
    return r


def py_in(env: dict, service: str, code: str) -> str:
    return compose(env, "exec", "-T", service, "python", "-c", code).stdout.strip()


def sse(client: httpx.Client, url: str, headers: dict, body: dict) -> list:
    events, name = [], None
    with client.stream("POST", url, headers=headers, json=body, timeout=180) as r:
        assert r.status_code == 200, (r.status_code, r.read()[:300])
        for line in r.iter_lines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                events.append((name, json.loads(line[5:].strip())))
    return events


def document_vectors(env: dict, user_id: str, document_id: str, max_chunks: int = 50) -> int:
    """How many of this document's vector ids exist in its owner's namespace (Pinecone fetch by id).
    Index-wide statistics lag behind writes, so they cannot prove that a delete removed anything."""
    return int(py_in(env, "rag-core", (
        "import os, httpx; h = {'Api-Key': os.environ['PINECONE_API_KEY'], 'X-Pinecone-API-Version': '2025-04'}; "
        "host = httpx.get('https://api.pinecone.io/indexes/' + os.environ['PINECONE_INDEX'], headers=h, timeout=20).json()['host']; "
        f"ids = [f'{document_id}:{{i}}' for i in range({max_chunks})]; "
        f"r = httpx.get(f'https://{{host}}/vectors/fetch', headers=h, params={{'namespace': 'user-{user_id}', 'ids': ids}}, timeout=20); "
        "print(len(r.json().get('vectors') or {}))")))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="leave the stack running afterwards")
    args = parser.parse_args()

    rag_env = dotenv(ROOT / "rag-core" / ".env")
    if rag_env.get("PINECONE_INDEX") != "advancerag-bge384":
        raise SystemExit("refusing: rag-core/.env PINECONE_INDEX is not advancerag-bge384")
    gen = {k: secrets.token_urlsafe(48) for k in ("INTERNAL_SERVICE_TOKEN", "JWT_SECRET", "JWT_REFRESH_SECRET")}
    gen |= {k: secrets.token_hex(16) for k in ("POSTGRES_PASSWORD", "REDIS_PASSWORD", "FREELLMAPI_API_KEY", "S3_ACCESS_KEY", "S3_SECRET_KEY")}
    env = os.environ | gen | {"PINECONE_API_KEY": rag_env["PINECONE_API_KEY"], "PINECONE_INDEX": rag_env["PINECONE_INDEX"]}
    secret_values = list(gen.values()) + [rag_env["PINECONE_API_KEY"]]

    print("building and starting the production stack (first build takes a few minutes)...", flush=True)
    compose(env, "up", "-d", "--build", capture=True)
    try:
        return run(env, secret_values)
    finally:
        if args.keep:
            print("stack left running (--keep); remove with: docker compose -p advrag-rehearsal down -v")
        else:
            compose(env, "down", "-v", "--remove-orphans", check_rc=False)
            print("stack removed (containers, network, volumes)")


def run(env: dict, secret_values: list) -> int:
    # Bucket provisioning is a one-time deploy step (R2/S3 console in production).
    for _ in range(30):
        r = compose(env, "exec", "-T", "backend", "python", "-c", (
            "import os, boto3; s3 = boto3.client('s3', endpoint_url=os.environ['STORAGE_ENDPOINT'], region_name='us-east-1', "
            "aws_access_key_id=os.environ['STORAGE_ACCESS_KEY'], aws_secret_access_key=os.environ['STORAGE_SECRET_KEY']); "
            "s3.create_bucket(Bucket=os.environ['STORAGE_BUCKET'])"), check_rc=False)
        if r.returncode == 0 or "BucketAlready" in (r.stderr or ""):
            break
        time.sleep(2)

    ready = None
    for _ in range(90):  # rag-core loads the embedding model on start
        try:
            ready = httpx.get(BASE + "/ready", timeout=20)
            if ready.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(3)
    if not check("backend /ready (database, S3 storage, agent-core, rag-core → llm → FreeLLMAPI)",
                 ready is not None and ready.status_code == 200, ready.text if ready is not None else "no response"):
        print(compose(env, "logs", "--tail", "40", check_rc=False).stdout[-4000:])
        return 1
    check("readiness requires dependent services in production", ready.json().get("services_required") is True)

    published = compose(env, "ps", "--format", "json").stdout
    ports = [p for line in published.splitlines() if line.strip() for p in (json.loads(line).get("Publishers") or []) if p.get("PublishedPort")]
    check("only the backend is published to the host", [p["PublishedPort"] for p in ports] == [8200], str([p["PublishedPort"] for p in ports]))

    for svc, port in (("agent-core", 8001), ("rag-core", 8002), ("llm", 8003)):
        # Railway's private network can be IPv6-only: the services listen on HOST=:: (dual-stack).
        got = py_in(env, "backend", (
            "import socket, httpx; "
            f"a = socket.getaddrinfo('{svc}', {port}, socket.AF_INET6, socket.SOCK_STREAM)[0][4][0]; "
            f"print(a, httpx.get(f'http://[{{a}}]:{port}/health', timeout=10).status_code)"))
        check(f"{svc} reachable over IPv6 from the backend", got.endswith(" 200"), got)

    for svc, url in (("agent-core", "http://agent-core:8001/api/v1/agent/invoke"), ("rag-core", "http://rag-core:8002/api/v1/rag/retrieve"), ("llm", "http://llm:8003/v1/generate")):
        code = py_in(env, "backend", f"import httpx; print(httpx.post('{url}', json={{}}, timeout=10).status_code)")
        check(f"{svc} rejects a call without INTERNAL_SERVICE_TOKEN (inside the private network)", code == "401", code)

    c = httpx.Client(timeout=60, headers={"X-Forwarded-For": "203.0.113.7"})
    email = f"rehearsal-{secrets.token_hex(4)}@example.com"
    password = "Rehearsal-" + secrets.token_urlsafe(12)
    r = c.post(API + "/auth/register", json={"email": email, "password": password, "full_name": "Rehearsal"})
    check("register", r.status_code == 201, str(r.status_code))
    cookie = r.headers.get("set-cookie", "")
    check("refresh cookie is HttpOnly + Secure in production", "httponly" in cookie.lower() and "secure" in cookie.lower())
    r = c.post(API + "/auth/login", json={"email": email, "password": password})
    check("password login", r.status_code == 200, str(r.status_code))
    auth = {"Authorization": f"Bearer {r.json()['access_token']}"}
    user_id = r.json()["user"]["id"]

    r = c.post(API + "/documents", headers=auth, files={"file": ("orion-z9.txt", (FACT + "\n") * 3, "text/plain")})
    check("upload to S3 storage", r.status_code == 201, str(r.status_code))
    doc = r.json()["id"]
    status = None
    for _ in range(60):
        status = c.get(f"{API}/documents/{doc}", headers=auth).json()["status"]
        if status in ("ready", "failed"):
            break
        time.sleep(2)
    check("ingestion reaches ready (rag-core → Pinecone advancerag-bge384)", status == "ready", str(status))
    indexed = document_vectors(env, user_id, doc)
    check("document's vectors are in Pinecone (fetched by id in the user's namespace)", indexed > 0, f"{indexed} vectors")
    stored = py_in(env, "backend", (
        "import os, boto3; s3 = boto3.client('s3', endpoint_url=os.environ['STORAGE_ENDPOINT'], region_name='us-east-1', "
        "aws_access_key_id=os.environ['STORAGE_ACCESS_KEY'], aws_secret_access_key=os.environ['STORAGE_SECRET_KEY']); "
        "print(s3.list_objects_v2(Bucket=os.environ['STORAGE_BUCKET']).get('KeyCount', 0))"))
    check("uploaded file is in the S3 bucket", stored not in ("", "0"), stored)

    conv = c.post(API + "/conversations", headers=auth, json={"title": "rehearsal"}).json()["id"]
    events = sse(c, f"{API}/conversations/{conv}/messages/stream", auth,
                 {"content": "How long is the warranty on the Orion Z9?", "document_ids": [doc]})
    names = [n for n, _ in events]
    done = next((d for n, d in events if n == "message.completed"), {})
    msg = done.get("assistant_message") or {}
    text = "".join(d.get("delta", d.get("content", "")) for n, d in events if n == "content.delta")
    check("SSE stream: retrieval → generation → completed", all(e in names for e in ("retrieval.completed", "content.delta", "message.completed")) and "error" not in names, ",".join(dict.fromkeys(names)))
    check("grounded answer quotes the document", "seven years" in text, text[:120])
    cites = msg.get("citations") or []
    check("citation returned with filename and source_id", any(ci.get("filename") == "orion-z9.txt" and ci.get("source_id") == 1 for ci in cites), json.dumps(cites)[:200])

    other = httpx.Client(timeout=30)
    o = other.post(API + "/auth/register", json={"email": f"x-{secrets.token_hex(4)}@example.com", "password": password, "full_name": "Other"}).json()
    oh = {"Authorization": f"Bearer {o['access_token']}"}
    check("cross-user document access is 404", other.get(f"{API}/documents/{doc}", headers=oh).status_code == 404)
    check("cross-user conversation access is 404", other.get(f"{API}/conversations/{conv}", headers=oh).status_code == 404)

    keys = compose(env, "exec", "-T", "redis", "redis-cli", "-a", env["REDIS_PASSWORD"], "--no-auth-warning", "--scan").stdout.split()
    check("rate-limit counters live in Redis", len(keys) > 0, f"{len(keys)} keys")
    bad = httpx.Client(timeout=30, headers={"X-Forwarded-For": "198.51.100.9"})
    codes = [bad.post(API + "/auth/login", json={"email": email, "password": "wrong-password-1A"}).status_code for _ in range(12)]
    check("login brute force is throttled (429) via Redis", 429 in codes, str(codes))

    r = c.delete(f"{API}/documents/{doc}", headers=auth)
    check("document delete", r.status_code == 204, str(r.status_code))
    c.delete(f"{API}/conversations/{conv}", headers=auth)
    left = indexed
    for _ in range(15):  # Pinecone is eventually consistent
        left = document_vectors(env, user_id, doc)
        if left == 0:
            break
        time.sleep(4)
    check("Pinecone vectors removed on delete (fetched by id)", indexed > 0 and left == 0, f"before={indexed} after={left}")

    logs = compose(env, "logs", "--no-color", check_rc=False).stdout
    leaked = sum(1 for s in secret_values if s in logs)
    check("no secret value appears in any service log", leaked == 0, f"{leaked} leaked")
    check("logs are structured JSON with request ids", '"request_id"' in logs)

    r = compose(env, "run", "--rm", "--no-deps", "-e", "FREELLMAPI_BASE_URL=http://127.0.0.1:31415/v1", "-e", "FREELLMAPI_ALLOW_RAILWAY_PRIVATE_HTTP=false", "llm", check_rc=False)
    check("llm refuses to start with a localhost FreeLLMAPI URL in production", r.returncode != 0)
    r = compose(env, "run", "--rm", "--no-deps", "-e", "JWT_SECRET=short", "backend", "python", "-c", "import app.main", check_rc=False)
    check("backend refuses to start with a weak JWT secret in production", r.returncode != 0)

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
