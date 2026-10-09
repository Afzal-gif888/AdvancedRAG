# Free deployment: Vercel + Render + Neon + Pinecone + Railway (FreeLLMAPI)

| Component | Platform | Plan | Status (2026-10-10) |
|---|---|---|---|
| React + Vite frontend | Vercel, project `advancerag-web` | Hobby (free) | **live**: https://advancerag-web.vercel.app (custom domain `app.afzaltech.in` pending DNS verification) |
| PostgreSQL `advancerag`, `advancerag_rag` | Neon project `AdvancedRAG` (aws-us-east-2, PostgreSQL 18), branch `production` | Free | **created and migrated** (2026-10-10): `advancerag` at 0004 (owner role `advancerag_backend_app`), `advancerag_rag` at 0003 (owner role `advancerag_rag_app`); TLS 1.3 enforced |
| Vectors | Pinecone `advancerag-bge384` (384-dim, cosine, aws us-east-1) | existing | verified; reused unchanged |
| backend, agent-core, rag-core, llm | Render (virginia), created from the `render.yaml` values | Free web services | **live**: https://advancerag-backend.onrender.com, https://advancerag-agent-core.onrender.com, https://advancerag-rag-core.onrender.com, https://advancerag-llm.onrender.com (all `/ready` 200; smoke test 18/18) |
| Redis (rate limits) | Render Key Value `advancerag-redis` | Free | **available** |
| FreeLLMAPI | Railway project `advancerag`, service `freellmapi` | existing | running; https://freellmapi-production-a959.up.railway.app (unified key set on `advancerag-llm`; real completions verified) |

Railway's own Redis keeps running, unused: Render's free services cannot reach it privately, and making
it public is not worth the risk. Render's free Key Value is private and in the same region.

## What the free tiers mean in practice

Measured with the production RAG Core image limited to Render's free instance (512 MB RAM, 0.1 CPU),
with real Pinecone and embeddings:

| | Default settings | Settings in `render.yaml` |
|---|---|---|
| 1,000-row CSV | **killed (out of memory)** | indexed in 141 s |
| 30-page PDF | — | indexed in 306 s |
| 2-page PDF | 36 s | 7 s |
| retrieval (with reranking) | — | 5.6 s |
| peak RAM | 512 MiB (killed) | 324 MiB |
| start until the model is loaded | 135 s | up to 345 s |

The settings changed are memory and speed knobs only (`EMBEDDING_BATCH_SIZE=4`, `EMBEDDING_THREADS=1`,
`INGEST_CONCURRENCY=1`, `INGEST_EMBED_SLICE=32`, `MALLOC_ARENA_MAX=2`). The frozen retrieval
parameters (model, chunking, top-k, RRF, reranker, context) are unchanged.

Limits to expect:

- **Sleep.** Each service sleeps after 15 min without traffic. The first visit after a pause wakes the
  backend (about 1 min); it then wakes Agent Core and RAG Core in parallel (`WAKE_DEPENDENCIES_ON_START`),
  and RAG Core needs a few more minutes to load the model. The first question can take several minutes
  or time out; ask again once it is awake. The timeouts in `render.yaml` are raised for this.
- **750 free instance-hours per month for the whole workspace.** Four services awake together use four
  hours per hour, so about 187 hours of active use per month. If the pool runs out, Render suspends
  every free service until the next month.
- **Slow indexing.** 0.1 CPU is about 30× slower than a laptop. Keep the browser tab open while a large
  document indexes: the page's status polling keeps the services awake.
- **Uploaded originals are not durable.** There is no persistent disk on free instances. Indexing works
  (it runs right after the upload), but downloading or reprocessing an original after the service
  restarts or sleeps fails. For durable files, switch the backend to `STORAGE_BACKEND=s3` with Cloudflare
  R2 (free 10 GB; Cloudflare asks for a payment method to enable R2).
- **Build minutes and bandwidth** count against Render's monthly free allowance. Without a payment
  method on file, Render disables builds or suspends services instead of charging.
- **FreeLLMAPI on Railway's free plan** stops when the Railway free credit is used up.

## Frontend to backend

The Vercel project proxies `/api/*` to the backend (`frontend/vercel.json`, written by
`python scripts/configure_vercel.py --mode proxy --backend-url https://advancerag-backend.onrender.com`).
The browser only talks to `advancerag-web.vercel.app`, so the refresh cookie is first-party. Requests pass
two proxies (Vercel, then Render), hence `TRUSTED_PROXY_HOPS=2` on the backend. Once `api.afzaltech.in`
works, switch to direct mode with `--mode direct --backend-url https://api.afzaltech.in` and set
`TRUSTED_PROXY_HOPS=1`.

## Setup steps

1. **Neon** (done). The databases exist and are migrated; each has its own owner role. The **pooled**
   connection strings (`sslmode=require`) for step 3 are in `%USERPROFILE%dvancerag-secrets
eon.txt`
   (local only). Migrations also run on service start and are idempotent.
2. **FreeLLMAPI provider keys.** Open https://freellmapi-production-a959.up.railway.app and sign in with
   the details in `%USERPROFILE%\advancerag-secrets\freellmapi.txt`. Add your LLM provider keys, then
   copy the unified key from the Keys page.
3. **Render.** Sign up with the GitHub account that owns the repository (Afzal-gif888) so Render can read
   it. Then New → Blueprint → select `AdvancedRAG`. Render reads `render.yaml` and asks for:

   | Service | Variable | Value |
   |---|---|---|
   | advancerag-rag-core | `DATABASE_URL` | Neon pooled URL for `advancerag_rag` |
   | advancerag-rag-core | `PINECONE_API_KEY` | your existing Pinecone key |
   | advancerag-llm | `FREELLMAPI_BASE_URL` | `https://freellmapi-production-a959.up.railway.app/v1` |
   | advancerag-llm | `FREELLMAPI_API_KEY` | the unified key from step 2 |
   | advancerag-agent-core | `RAG_CORE_URL`, `LLM_SERVICE_URL` | `https://advancerag-rag-core.onrender.com`, `https://advancerag-llm.onrender.com` |
   | advancerag-backend | `DATABASE_URL` | Neon pooled URL for `advancerag` |
   | advancerag-backend | `AGENT_CORE_URL`, `RAG_CORE_URL` | `https://advancerag-agent-core.onrender.com`, `https://advancerag-rag-core.onrender.com` |

   If Render gives a service a different `.onrender.com` name (because one was taken), use the real one.
   `INTERNAL_SERVICE_TOKEN` and the JWT secrets are generated by Render.
4. **DNS for afzaltech.in** (at your DNS provider):
   - `api` → CNAME to `advancerag-backend.onrender.com`. Render shows the exact target and verifies it
     under the backend's Custom Domains.
   - `app` → add `app.afzaltech.in` to the Vercel project `advancerag-web` (Settings → Domains), then
     create the record Vercel shows.
5. **Verify** with your PC switched off:
   ```
   python scripts/smoke_production.py --url https://app.afzaltech.in --backend-url https://api.afzaltech.in
   ```
   Also check `TRUSTED_PROXY_HOPS` in the backend logs: rate-limit keys must be the real client IP
   behind Render's proxy.

## Alternatives kept in the repository

- `infrastructure/production-vm/`: everything on one VM with Docker Compose (validated locally; needs
  a VM with at least 2 GB RAM + swap).
- `docs/deployment-checklist.md`: Railway layout (needs a paid Railway plan for more than three services).
