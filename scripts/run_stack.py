"""Start the whole AdvancedRAG stack as local processes.

    python scripts/run_stack.py               # development: each service reads its own .env
    python scripts/run_stack.py --frontend    # ... and the Vite dev server
    python scripts/run_stack.py --mode e2e    # isolated: temp SQLite DBs, mock FreeLLMAPI, real local
                                              # embeddings, local vector store (Playwright suite)

Development mode expects PostgreSQL (docker compose up -d) and FreeLLMAPI on your desktop.
Ctrl+C stops everything.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
# e2e mode writes emails (sign-up/login codes) here so the Playwright tests can read them.
E2E_OUTBOX = ROOT / ".e2e-outbox"

SERVICES = [
    # name, directory, asgi app, dev port, e2e port
    ("llm", "llm", "llm_service.main:app", 8003, 8103),
    ("rag-core", "rag-core", "rag_core.main:app", 8002, 8102),
    ("agent-core", "agent-core", "agent_core.main:app", 8001, 8101),
    ("backend", "backend", "app.main:app", 8000, 8100),
]


def wait_http(url: str, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status < 500:
                    return
        except Exception:
            time.sleep(0.3)
    raise RuntimeError(f"timed out waiting for {url}")


def migrate(directory: str, env: dict) -> None:
    subprocess.run([PY, "-m", "alembic", "upgrade", "head"], cwd=ROOT / directory, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "e2e"], default="dev")
    parser.add_argument("--frontend", action="store_true")
    parser.add_argument("--frontend-port", type=int, default=None)
    args = parser.parse_args()
    e2e = args.mode == "e2e"

    procs = []
    workdir = Path(tempfile.mkdtemp(prefix="advancerag-e2e-")) if e2e else None
    if e2e:
        shutil.rmtree(E2E_OUTBOX, ignore_errors=True)

    def stop(*_):
        for p in reversed(procs):
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        sys.exit(0)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, stop)

    ports = {name: (e2e_port if e2e else dev_port) for name, _, _, dev_port, e2e_port in SERVICES}
    frontend_port = args.frontend_port or (5174 if e2e else 5173)
    base = {**os.environ, "PYTHONUNBUFFERED": "1"}
    if e2e:
        token = "e2e-internal-service-token-0123456789abcdefghijkl"
        mock_port = 3901
        # ADVANCERAG_ENV_FILE=none: the e2e stack never reads a developer's .env, so real keys
        # (Resend, Pinecone, FreeLLMAPI) can't be used by tests; everything it needs is set here.
        common = {"ENVIRONMENT": "development", "INTERNAL_SERVICE_TOKEN": token, "LOG_LEVEL": "WARNING", "ADVANCERAG_ENV_FILE": "none"}
        overrides = {
            "llm": {**common, "FREELLMAPI_BASE_URL": f"http://127.0.0.1:{mock_port}/v1", "FREELLMAPI_API_KEY": "e2e-mock-key"},
            "rag-core": {**common, "DATABASE_URL": f"sqlite+aiosqlite:///{workdir / 'rag.db'}", "EMBEDDING_PROVIDER": "local",
                         # Real local model (shared download cache); Pinecone swapped for the dev-only local store.
                         "EMBEDDING_CACHE_DIR": str(ROOT / "rag-core" / "var" / "models"),
                         "VECTOR_STORE": "local", "RERANKER": "none"},
            "agent-core": {**common, "RAG_CORE_URL": f"http://127.0.0.1:{ports['rag-core']}", "LLM_SERVICE_URL": f"http://127.0.0.1:{ports['llm']}"},
            "backend": {**common, "DATABASE_URL": f"sqlite+aiosqlite:///{workdir / 'backend.db'}",
                        "JWT_SECRET": "e2e-access-secret-0123456789abcdefghijklmnop", "JWT_REFRESH_SECRET": "e2e-refresh-secret-0123456789abcdefghijklmn",
                        "AGENT_CORE_URL": f"http://127.0.0.1:{ports['agent-core']}", "RAG_CORE_URL": f"http://127.0.0.1:{ports['rag-core']}",
                        "FRONTEND_ORIGIN": f"http://localhost:{frontend_port}", "LOCAL_STORAGE_PATH": str(workdir / "storage"),
                        "EMAIL_PROVIDER": "file", "EMAIL_OUTBOX_PATH": str(E2E_OUTBOX), "RATE_LIMIT_ENABLED": "false", "WORKER_POLL_INTERVAL_SECONDS": "0.5",
                        "LOG_JSON": "false"},
        }
        procs.append(subprocess.Popen([PY, str(ROOT / "scripts" / "mock_freellmapi.py"), "--port", str(mock_port), "--key", "e2e-mock-key"], env=base))
        wait_http(f"http://127.0.0.1:{mock_port}/docs")
        migrate("rag-core", {**base, **overrides["rag-core"]})
        migrate("backend", {**base, **overrides["backend"]})
    else:
        overrides = {name: {} for name, *_ in SERVICES}
        for directory in ("rag-core", "backend"):
            migrate(directory, base)

    for name, directory, app, *_ in SERVICES:
        env = {**base, **overrides[name]}
        cmd = [PY, "-m", "uvicorn", app, "--host", "127.0.0.1", "--port", str(ports[name])]
        procs.append(subprocess.Popen(cmd, cwd=ROOT / directory, env=env))
        wait_http(f"http://127.0.0.1:{ports[name]}/health")
        print(f"[run_stack] {name} ready on :{ports[name]}", flush=True)

    if args.frontend or e2e:
        npx = "npx.cmd" if os.name == "nt" else "npx"
        fenv = {**base, "VITE_API_PROXY_TARGET": f"http://127.0.0.1:{ports['backend']}"}
        procs.append(subprocess.Popen([npx, "vite", "--port", str(frontend_port), "--strictPort"], cwd=ROOT / "frontend", env=fenv))
        wait_http(f"http://localhost:{frontend_port}/")
        print(f"[run_stack] frontend ready on http://localhost:{frontend_port}", flush=True)

    print("[run_stack] all services running — Ctrl+C to stop", flush=True)
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(1)
        print("[run_stack] a service exited; shutting down", flush=True)
    finally:
        stop()


if __name__ == "__main__":
    main()
