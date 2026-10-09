# AdvancedRAG

Chat with your documents. A React frontend, a FastAPI backend, and three internal services:

- **Agent Core** orchestrates each chat turn.
- **RAG Core** handles retrieval.
- **The LLM layer** talks to models, through **FreeLLMAPI** (an OpenAI-compatible gateway).

```text
Frontend (React/TS)          UI, streaming display. No secrets.
   │  HTTPS + SSE (/api/v1)
Backend (FastAPI)            auth, users, conversations, messages, documents, shares, jobs, SSE
   │  internal HTTP + Bearer INTERNAL_SERVICE_TOKEN
Agent Core                   decides retrieval, rewrites follow-ups, builds grounded prompt, citations
   ├──► RAG Core             parse → chunk → local embeddings (fastembed bge-small, 384-d) → Pinecone
   │                         + PostgreSQL full-text
   │                         → hybrid retrieval → RRF → Pinecone rerank → context
   └──► LLM layer            provider abstraction (LLMProvider) → FreeLLMAPIProvider
          │  OpenAI-compatible HTTP
        FreeLLMAPI           routes "auto" / "auto:fast" / "auto:smart" to an actual model
```

| Component | Owns | Holds secrets for |
|---|---|---|
| `frontend/` | UI only | none (everything in `VITE_*` is public) |
| `backend/` | users, sessions, chats, documents metadata/files, shares, background jobs | JWT secrets, DB, Redis, storage, email |
| `agent-core/` | orchestration decisions, prompts, citations | none |
| `rag-core/` | parsing, chunking, local embeddings, vectors, keyword index, reranking, context | Pinecone |
| `llm/` | model calls, streaming, retries, timeouts | **FreeLLMAPI key** |
| `shared/` | request IDs, redacting logs, metrics, internal auth helpers | — |

Observability works the same way in every service:

- `X-Request-ID` is propagated through every hop.
- Logs are structured JSON and redact secrets.
- Each service records latency for every dependency call.
- `/metrics` (Prometheus format) is protected by the internal token.

The interfaces are documented in [backend/API_CONTRACT.md](backend/API_CONTRACT.md).

---

## Local setup

**Prerequisites:**

- Python 3.11+ (3.13 tested)
- Node 20+
- Docker Desktop
- Google Chrome (for the end-to-end tests)
- The FreeLLMAPI desktop app

### 1. Infrastructure (PostgreSQL + Redis)

```bash
cp .env.example .env            # set POSTGRES_PASSWORD (any random string)
docker compose up -d            # postgres on 127.0.0.1:5434, redis on 127.0.0.1:6380
```

The first start creates these databases:

- `advancerag` (backend)
- `advancerag_rag` (RAG Core)
- `advancerag_test` and `advancerag_rag_test` (test suites)

If the ports clash with something else on your machine, change `POSTGRES_PORT` / `REDIS_PORT` in `.env`.

### 2. Python environment (one venv for all services)

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements-dev.txt        # Windows
# .venv/bin/pip install -r requirements-dev.txt          # Linux/macOS
```

### 3. Configure each service

Copy each `.env.example` to `.env` and fill it in:

| File | What to set |
|---|---|
| `backend/.env` | `DATABASE_URL` (password from `.env`, host `127.0.0.1`, port `5434`), `JWT_SECRET`, `JWT_REFRESH_SECRET`, `INTERNAL_SERVICE_TOKEN` |
| `agent-core/.env` | the same `INTERNAL_SERVICE_TOKEN` |
| `rag-core/.env` | `DATABASE_URL` (…`/advancerag_rag`), `INTERNAL_SERVICE_TOKEN`, `PINECONE_API_KEY`, `PINECONE_INDEX` |
| `llm/.env` | `INTERNAL_SERVICE_TOKEN`, `FREELLMAPI_API_KEY` |
| `frontend/.env` *(optional)* | nothing needed for local development |

Generate secrets with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

**Embeddings are local, so no embedding API key is needed.** RAG Core runs `BAAI/bge-small-en-v1.5` through fastembed (ONNX, on the CPU). It downloads the model once (about 70 MB) into `rag-core/var/models` on first start. Documents and queries are embedded with the same model, into **384-dimensional** vectors.

**Pinecone:** create a *serverless* index with **dimension 384** and **metric cosine**. Put its name in `PINECONE_INDEX` and your key in `PINECONE_API_KEY`.

- The dimension must match the embedding model. RAG Core checks this: `/ready` reports `DIMENSION_MISMATCH: index has X, embedding model produces 384`, and writes to a mismatched index are refused.
- An existing 768-dimension index cannot be reused with this model; create a new 384-dimension one.
- If you ever change `EMBEDDING_MODEL`, create a new index with the new model's dimension and re-ingest your documents.

### 4. FreeLLMAPI (desktop)

1. Start the **FreeLLMAPI** desktop app and, in **its dashboard, add at least one upstream provider key** (e.g. OpenRouter, Groq, Google AI Studio). Without one, FreeLLMAPI answers every request with `503 no_providers_configured`. Its OpenAI-compatible API listens locally (for example `http://127.0.0.1:31415/v1`; the port depends on the app version), so set `FREELLMAPI_BASE_URL` to match.
2. Copy its **unified API key** from the app into `llm/.env` as `FREELLMAPI_API_KEY`. Only the LLM service ever sees this key.
3. Check the connection. This never prints the key:

```bash
cd llm
../.venv/Scripts/python -m llm_service.healthcheck          # config, reachability, auth
../.venv/Scripts/python -m llm_service.healthcheck --chat   # also a tiny real completion
```

To choose a model, set `FREELLMAPI_MODEL` (in `llm/.env`) and `AGENT_FAST_MODEL` (in `agent-core/.env`). Both accept any id that `GET /v1/models` lists (for example `auto`); leave `AGENT_FAST_MODEL` empty to use the default. No code changes are needed.

### 5. Accounts and email

Sign-up asks for a name, an email and a password, and signs the user in straight away. Login is email and password. Neither sends any email.

Email is used only for **forgot-password** links, and it is optional. With `EMAIL_PROVIDER` empty, development writes reset emails to `backend/var/outbox/` (open the newest file to get the link). For real delivery, set Resend (production) or SMTP in `backend/.env` (see `backend/.env.example` and [docs/production-deployment.md](docs/production-deployment.md#5a-password-reset-email-resend)). Passwords cannot be changed from the profile, so without an email provider, a user who forgets their password cannot reset it.

### 6. Run everything

```bash
.venv/Scripts/python scripts/run_stack.py --frontend
```

This applies migrations and starts each part on its own port:

| Service | Port |
|---|---|
| llm | :8003 |
| rag-core | :8002 |
| agent-core | :8001 |
| backend | :8000 |
| Vite | :5173 |

Then open http://localhost:5173. Press Ctrl+C to stop everything.

<details><summary>Or run the services one by one</summary>

```bash
# each from its own directory, so it reads its own .env
cd llm        && ../.venv/Scripts/python -m uvicorn llm_service.main:app --port 8003
cd rag-core   && ../.venv/Scripts/python -m alembic upgrade head && ../.venv/Scripts/python -m uvicorn rag_core.main:app --port 8002
cd agent-core && ../.venv/Scripts/python -m uvicorn agent_core.main:app --port 8001
cd backend    && ../.venv/Scripts/python -m alembic upgrade head && ../.venv/Scripts/python -m uvicorn app.main:app --port 8000
cd frontend   && npm install && npm run dev
```
</details>

**Readiness:** `GET http://localhost:8000/ready` reports the database, storage, Agent Core and RAG Core. Agent Core's `/ready` in turn checks RAG Core and the LLM layer, and the LLM layer's `/ready` checks FreeLLMAPI reachability and authentication.

---

## How it works

### Document ingestion

1. **Upload.** The browser uploads the file to `POST /api/v1/documents`.
2. **Validation.** The backend checks:
   - the extension allowlist;
   - that the file's bytes actually match the extension (the browser's MIME type is ignored);
   - size and quota limits.

   It also cleans up the filename.
3. **Storage.** The backend stores the file under a key it generates itself (local disk, or S3 in production) and creates a `background_jobs` row.
4. **Worker.** The in-process worker claims the job (compare-and-set, so the same job never runs twice) and sends the file to RAG Core at `POST /api/v1/rag/ingest`.
5. **RAG Core** processes the file:
   - parses PDF, DOCX, TXT, MD or CSV, keeping page numbers;
   - splits it into overlapping chunks;
   - embeds the chunks locally (bge-small, 384 dimensions);
   - upserts the vectors into Pinecone, in a separate namespace per user;
   - stores the chunk text in PostgreSQL with a full-text index.

   Re-ingesting the same document overwrites its previous data.
6. **Status.** The backend marks the document `ready`. The UI shows Uploading → Queued → Processing → Ready / Failed.

Failures are bounded:

- Transient errors are retried with backoff.
- Permanent errors (empty or scanned PDFs, unsupported content) fail immediately with a clear message.
- A reconciler recovers jobs left behind by a crashed worker, never retrying more than `max_attempts`.
- Deletes also go through RAG Core (vectors) and storage (file), with bounded re-drives, so nothing is orphaned.

### Chat generation

1. **Request.** The browser sends `POST /api/v1/conversations/{id}/messages/stream` (SSE).
2. **Backend.** It saves the user message and an assistant placeholder, then calls Agent Core with the recent history and the conversation's *ready* documents.
3. **Agent Core decides what to do:**
   - With documents attached (and a message that isn't small talk), it rewrites a follow-up question into a standalone search query using `auto:fast`. It then asks RAG Core for passages: semantic search plus keyword search, fused with RRF and reranked by Pinecone.
   - It builds a grounded prompt with numbered sources and streams the answer from the LLM layer, which streams from FreeLLMAPI.
   - Agent Core turns `[n]` markers into citations, and only for sources that actually exist.
4. **Streaming back.** The backend forwards only safe events (`agent.started`, `retrieval.*`, `generation.started`, `content.delta`, `message.completed` / `error`) and saves partial output every few seconds. The final message is saved with its citations and safe metadata (model, token usage, latencies).
5. **Failures.** Nothing is ever fabricated:
   - If RAG Core is unavailable, no answer is generated.
   - If the LLM layer fails, the message is saved as `failed` with a safe error code, keeping any partial text, and the user can retry.
   - If the browser disconnects, the message is saved as `cancelled`.
   - If the process crashes mid-stream, the reconciler marks the message `failed` (`INTERRUPTED`) and it can be retried.

---

## Tests

```bash
# per service (each from its directory)
cd backend    && ../.venv/Scripts/python -m pytest      # API, auth, isolation, jobs, SSE, security
cd llm        && ../.venv/Scripts/python -m pytest      # FreeLLMAPI adapter vs a mock OpenAI-compatible HTTP server
cd rag-core   && ../.venv/Scripts/python -m pytest      # parsing, chunking, the real local model, retrieval, Pinecone adapter vs mock Pinecone
cd agent-core && ../.venv/Scripts/python -m pytest      # orchestration decisions and failure handling
.venv/Scripts/python -m pytest tests/contract           # real backend -> agent-core -> rag-core/llm -> mock FreeLLMAPI

# against real PostgreSQL / Redis / S3 (docker compose up -d; --profile storage for S3)
TEST_DATABASE_URL=postgresql+asyncpg://advancerag:<pw>@127.0.0.1:5434/advancerag_test \
TEST_REDIS_URL=redis://127.0.0.1:6380/15 \
TEST_S3_ENDPOINT=http://127.0.0.1:8333 TEST_S3_ACCESS_KEY=any TEST_S3_SECRET_KEY=any \
  ../.venv/Scripts/python -m pytest                     # from backend/
TEST_DATABASE_URL=postgresql+asyncpg://advancerag:<pw>@127.0.0.1:5434/advancerag_rag_test \
  ../.venv/Scripts/python -m pytest                     # from rag-core/

# browser end-to-end (starts its own isolated stack with a mock FreeLLMAPI; no keys needed)
cd frontend && npm run test:e2e
```

No automated test needs a real FreeLLMAPI or Pinecone key. The first run downloads the local embedding model.

---

## Deployment notes and limitations

> **FreeLLMAPI on your desktop is for local development only.** A cloud deployment of the LLM service (Railway, Fly, any server) cannot reach `127.0.0.1:31415` on your machine. In production, `FREELLMAPI_BASE_URL` must point to a **secure (https), remotely reachable** FreeLLMAPI instance that you host (FreeLLMAPI ships a Docker server image). The LLM service refuses to start in staging or production with a missing, localhost or plain-http URL. Do not expose your desktop FreeLLMAPI to the internet. See **[docs/production-freellmapi.md](docs/production-freellmapi.md)**.

Production (`ENVIRONMENT=production`) **fails fast** unless all of the following are true:

- strong, distinct JWT secrets;
- a PostgreSQL `DATABASE_URL`;
- no wildcard CORS;
- `INTERNAL_SERVICE_TOKEN` of at least 32 characters, in every service;
- Redis rate limiting (`RATE_LIMIT_BACKEND=redis` with `REDIS_URL`);
- if an email provider is chosen (`resend` or `smtp`, used only for password-reset links), it must be fully configured, send from a real domain and link to an https frontend; the test/dev providers (`mock`, `file`) are refused;
- persistent storage (`STORAGE_BACKEND=s3`, or an explicit `ALLOW_LOCAL_STORAGE_IN_PRODUCTION=true` for a mounted volume);
- for RAG Core: Pinecone key and index (the fake embeddings and local vector store are refused);
- for the LLM layer: a FreeLLMAPI key plus an https, non-localhost URL.

Other deployment notes:

- Run `alembic upgrade head` for the backend and for RAG Core as a release step.
- The backend Dockerfile migrates on start, which is only safe for a single replica.
- Keep Agent Core, RAG Core and the LLM layer on a private network. They also require the internal token.
- Build images from the repository root, for example `docker build -f backend/Dockerfile .`. `docker compose --profile app up --build` runs everything in containers (the desktop FreeLLMAPI is reached via `host.docker.internal:31415`, which works only if the app accepts non-loopback connections).
- Images cannot be indexed yet. RAG Core has no OCR or vision step, so uploads are limited to PDF, TXT, MD, CSV and DOCX. Scanned PDFs fail with a clear message.
