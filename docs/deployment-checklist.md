# Production deployment checklist

Step-by-step order for deploying the whole application, derived from the repository as it is
(Dockerfiles, `*/railway.json`, `frontend/vercel.json`, each service's settings and startup
checks). Background and rationale: [production-deployment.md](production-deployment.md) and
[production-freellmapi.md](production-freellmapi.md). No secret values appear here. Generate
secrets with `python scripts/generate_secrets.py` and paste them only into the platform's
variable settings.

Status legend: ☐ to do · ✅ verified (with evidence) · ⚠ needs your action.

## 0. Accounts and access (your action)

| # | Account | Needed for | Status |
|---|---|---|---|
| 0.1 | **GitHub** repository (private), with this repo pushed | Railway and Vercel deploy from it | ⚠ the local repo has **no commits and no remote** yet |
| 0.2 | **Railway** project | 5 services + Redis + FreeLLMAPI | ⚠ |
| 0.3 | **Neon** project | 2 PostgreSQL databases | ⚠ |
| 0.4 | **Cloudflare R2** bucket + API token scoped to it (or another S3-compatible store) | uploaded files | ⚠ |
| 0.5 | **Pinecone** (existing) | index `advancerag-bge384` | ✅ reachable, verified in live runs and the rehearsal |
| 0.6 | **LLM provider keys** you use in FreeLLMAPI today (re-entered in the hosted dashboard) | answers | ⚠ |
| 0.7 | **Vercel** project | web frontend | ⚠ |
| 0.8 | **Resend** + a domain you own, verified for sending (optional) | password-reset email | ⚠ optional: without it, set `EMAIL_PROVIDER=` (empty); password reset then sends nothing |

The CLIs (`railway`, `vercel`, `neonctl`, `wrangler`) are not installed on the development
machine, and no account is linked. Each step below is done in the platform's web dashboard,
unless you install and log in to a CLI yourself.

## 1. Secrets to create (names only)

| Secret | Where | How |
|---|---|---|
| `JWT_SECRET`, `JWT_REFRESH_SECRET` (different values, ≥ 32 chars) | backend | `scripts/generate_secrets.py` |
| `INTERNAL_SERVICE_TOKEN` (≥ 32 chars, **same value** in 4 services) | backend, agent-core, rag-core, llm | `scripts/generate_secrets.py` |
| `ENCRYPTION_KEY` (64 hex chars) | freellmapi | `python -c "import secrets; print(secrets.token_hex(32))"`. **Never change it** later: it encrypts the stored provider keys |
| FreeLLMAPI admin email and password (≥ 8 chars, strong) | freellmapi (`FREEAPI_CONFIG_JSON`) | password manager |
| `FREELLMAPI_API_KEY` (the unified key) | llm | copied from the hosted FreeLLMAPI dashboard, Keys page |
| Neon connection strings (pooled, `sslmode=require`) | backend, rag-core | Neon dashboard |
| `STORAGE_ACCESS_KEY`, `STORAGE_SECRET_KEY` | backend | R2 API token |
| `PINECONE_API_KEY` | rag-core | Pinecone console (existing key) |
| `EMAIL_API_KEY` | backend (optional) | Resend |

## 2. Deployment order

Every service refuses to start with an unsafe configuration (§5). Railway's health check is
`/ready`, which checks real dependencies, so deploy **in this order** and validate each step
before the next. A dependent service cannot become healthy before its dependencies are.

```
Neon ─┐                      ┌─ rag-core ──┐
R2 ───┤                      │             ├─ agent-core ─┐
Redis ┤   freellmapi ─ llm ──┴─────────────┘              ├─ backend (public) ─ Vercel (frontend)
      └───────────────────────────────────────────────────┘
```

| Step | Service | Validate before continuing |
|---|---|---|
| 2.1 | Neon: databases `advancerag` and `advancerag_rag` | both pooled connection strings work (`sslmode=require`) |
| 2.2 | R2: private bucket + scoped token | — (checked by backend `/ready` in 2.8) |
| 2.3 | Railway: **Redis** plugin | — (checked by backend startup in 2.8) |
| 2.4 | Railway: **freellmapi** (image) | `GET /health` → 200 · dashboard login works · provider keys added · unified key copied · **public domain removed** afterwards |
| 2.5 | Railway: **llm** | `/ready` → `reachable`, `authenticated`, model `auto` listed; then `python -m llm_service.healthcheck --chat` from its shell → a real completion |
| 2.6 | Railway: **rag-core** | `/ready` → database, `vector_store` (index + 384 dims), `embedding_model_loaded`; migrations ran (logs: `alembic upgrade head`) |
| 2.7 | Railway: **agent-core** | `/ready` → `rag_core: true, llm: true` |
| 2.8 | Railway: **backend** (public domain) | `/ready` → database, storage, agent_core, rag_core all `true` |
| 2.9 | Vercel: **frontend** | site loads over HTTPS; `/api/*` reaches the backend |
| 2.10 | backend: set `FRONTEND_ORIGIN` / `FRONTEND_PUBLIC_URL` to the Vercel URL, redeploy | — |
| 2.11 | **End to end** | `python scripts/smoke_production.py --url https://<app>.vercel.app --backend-url https://<backend>.up.railway.app` → all checks pass, **with your PC switched off** (or FreeLLMAPI desktop closed) |

## 3. Railway services

All services: **source** = the GitHub repo; **root directory** = repository root (empty: the
Dockerfiles copy `shared/`); **builder** = Dockerfile (from the config file); **config as code**
= the path in the table. Railway injects `PORT`; set it explicitly so the private URLs are stable.
Every service listens on `HOST` (set `::` on Railway). The start command (in each Dockerfile; leave
Railway's start command empty) binds one dual-stack socket, so IPv6 (the private network) and IPv4
(health checks, the public edge) both work. Plain `uvicorn --host ::` would accept IPv6 only,
because asyncio sets `IPV6_V6ONLY`. That was found and fixed in the production rehearsal, which now
checks IPv6 between services and IPv4 to the backend (29/29).

| Service name | Config file / image | Dockerfile | Start command (from the Dockerfile) | `PORT` | Public domain | Health check | Volume |
|---|---|---|---|---|---|---|---|
| `freellmapi` | image `ghcr.io/tashfeenahmed/freellmapi:latest` | — | image default (`node server/dist/index.js`) | 3001 | **no** (temporarily yes, for the dashboard) | `/health` (unauthenticated, 200) | **`/app/server/data`** (required) |
| `llm` | `/llm/railway.json` | `llm/Dockerfile` | `python -m advancerag_common.serve llm_service.main:app 8003` | 8003 | no | `/ready`, 60 s | — |
| `rag-core` | `/rag-core/railway.json` | `rag-core/Dockerfile` (bakes BGE-small into the image) | `alembic upgrade head`, then `python -m advancerag_common.serve rag_core.main:app 8002` | 8002 | no | `/ready`, 300 s | — |
| `agent-core` | `/agent-core/railway.json` | `agent-core/Dockerfile` | `python -m advancerag_common.serve agent_core.main:app 8001` | 8001 | no | `/ready`, 60 s | — |
| `backend` | `/backend/railway.json` | `backend/Dockerfile` | `alembic upgrade head`, then `python -m advancerag_common.serve app.main:app 8000` | 8000 | **yes** | `/ready`, 120 s | — |
| `Redis` | Railway Redis plugin | — | — | — | no | — | managed |

Run one replica of each service. Migrations run on start (`RUN_MIGRATIONS=1`). With several
replicas, set `RUN_MIGRATIONS=0` on all but one.

## 4. Environment variables

`ENVIRONMENT=production`, `HOST=::` and `INTERNAL_SERVICE_TOKEN` go on **llm, rag-core,
agent-core and backend**. Everything not listed keeps its code default. RAG parameters are frozen:
do not set them.

**freellmapi**
| Variable | Value |
|---|---|
| `ENCRYPTION_KEY` | secret (64 hex) |
| `PORT` | `3001` |
| `FREEAPI_CONFIG_JSON` | `{"admin":{"email":"<you>","password":"<strong>"}}`. Set **before the first start**, so nobody else can claim the dashboard |

**llm**
| Variable | Value |
|---|---|
| `PORT` | `8003` |
| `FREELLMAPI_BASE_URL` | `http://freellmapi.railway.internal:3001/v1` |
| `FREELLMAPI_ALLOW_RAILWAY_PRIVATE_HTTP` | `true` (allows plain http only for `*.railway.internal`) |
| `FREELLMAPI_API_KEY` | secret (unified key) |
| `FREELLMAPI_MODEL` | `auto` |

**rag-core**
| Variable | Value |
|---|---|
| `PORT` | `8002` |
| `DATABASE_URL` | secret: Neon pooled URL for `advancerag_rag`, `sslmode=require` |
| `PINECONE_API_KEY` | secret |
| `PINECONE_INDEX` | `advancerag-bge384` |
| (do not set) `EMBEDDING_MODEL` / `EMBEDDING_DIMENSIONS` | defaults `BAAI/bge-small-en-v1.5` / 384 (frozen; the model is baked into the image) |

**agent-core**
| Variable | Value |
|---|---|
| `PORT` | `8001` |
| `RAG_CORE_URL` | `http://rag-core.railway.internal:8002` |
| `LLM_SERVICE_URL` | `http://llm.railway.internal:8003` |

**backend**
| Variable | Value |
|---|---|
| `PORT` | `8000` |
| `DATABASE_URL` | secret: Neon pooled URL for `advancerag`, `sslmode=require` |
| `JWT_SECRET`, `JWT_REFRESH_SECRET` | secrets |
| `AGENT_CORE_URL` | `http://agent-core.railway.internal:8001` |
| `RAG_CORE_URL` | `http://rag-core.railway.internal:8002` |
| `RATE_LIMIT_BACKEND` | `redis` |
| `REDIS_URL` | `${{Redis.REDIS_URL}}` (Railway reference) |
| `STORAGE_BACKEND` | `s3` |
| `STORAGE_ENDPOINT` | `https://<account-id>.r2.cloudflarestorage.com` |
| `STORAGE_REGION` | `auto` |
| `STORAGE_BUCKET` | bucket name |
| `STORAGE_ACCESS_KEY`, `STORAGE_SECRET_KEY` | secrets |
| `FRONTEND_ORIGIN`, `FRONTEND_PUBLIC_URL` | `https://<app>.vercel.app` (step 2.10) |
| `TRUSTED_PROXY_HOPS` | `2` (proxy mode: browser → Vercel → Railway edge → backend) |
| `EMAIL_PROVIDER` | empty (no reset email), or `resend` with `EMAIL_API_KEY` and `EMAIL_FROM=AdvancedRAG <no-reply@your-verified-domain>` |

## 5. Startup refusals (fix the variable, never relax the check)

| Service | Refuses to start when |
|---|---|
| backend | a JWT secret is weak / missing / equal to the other; `DATABASE_URL` is SQLite, or a public host without TLS; `INTERNAL_SERVICE_TOKEN` < 32 chars; agent/rag URL is localhost; no Redis rate limiting; `STORAGE_BACKEND=local` without the override; `EMAIL_PROVIDER` is `file`/`mock`, or resend/smtp is incomplete, or has a non-https reset URL / an unverified sender domain; wildcard CORS |
| rag-core | token < 32 chars; fake embeddings or local vectors; SQLite or a database without TLS; Pinecone key or index missing |
| agent-core | token < 32 chars; RAG/LLM URL is localhost |
| llm | FreeLLMAPI URL missing, localhost, or plain http (except `*.railway.internal` with the opt-in); key missing; token < 32 chars |

## 6. Vercel (frontend)

| Setting | Value |
|---|---|
| Root directory | `frontend` |
| Framework preset | Vite |
| Build command | `npm run build` |
| Output directory | `dist` |
| Install command | `npm ci` |
| Node.js | 22.x |
| Environment variable | `VITE_API_BASE_URL` **empty** (proxy mode: the browser calls `/api/v1` on the Vercel origin) |

Before the first Vercel deploy, once the backend has its domain (step 2.8):

```
python scripts/configure_vercel.py --mode proxy --backend-url https://<backend>.up.railway.app
```

This adds the `/api/*` → backend rewrite to `frontend/vercel.json` (commit it). The refresh cookie
then stays same-site, and the mobile app (later) must use this same origin
([mobile-readiness-report.md](mobile-readiness-report.md) G3).

To verify after deploy (not verifiable before): that SSE streaming passes through Vercel's
rewrite unbuffered, and that 25 MB uploads pass the rewrite. `smoke_production.py` covers
streaming. Test a large upload manually.

## 7. Done when

- [ ] Every Railway service is healthy on `/ready`, and its dependencies show `true`.
- [ ] `python -m llm_service.healthcheck --chat` succeeds on the production llm service.
- [ ] `smoke_production.py` passes against the Vercel URL **with your PC off**.
- [ ] A manual browser check: sign up, upload a PDF, ask with citations, CSV table, delete.

Only then start the React Native / Expo app.
