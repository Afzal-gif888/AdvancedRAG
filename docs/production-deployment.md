# Production deployment runbook

Target topology: **Vercel** (frontend) → **Railway** (backend, agent-core, rag-core, llm, Redis)
→ **Neon** (PostgreSQL ×2), **S3/R2** (uploads), **Pinecone** (`advancerag-bge384`), **hosted
FreeLLMAPI** (see [production-freellmapi.md](production-freellmapi.md)).

Only the backend gets a public Railway domain. agent-core, rag-core and llm are reachable only
over Railway private networking (`*.railway.internal`), and they also require the shared
`INTERNAL_SERVICE_TOKEN` on every endpoint except `/health` and `/ready`. The test
`tests/contract/test_service_auth.py` fails if a route is added without that guard.

Step-by-step order, exact service settings and every variable: [deployment-checklist.md](deployment-checklist.md).

Before deploying, rehearse locally with every production control turned on:

```
python scripts/rehearse_production.py
```

This runs the four images with `ENVIRONMENT=production`, Redis, S3 (SeaweedFS), private service
names and real Pinecone. It checks the full user path and the controls, then tears down.

## 1. Secrets

```
python scripts/generate_secrets.py     # prints JWT_SECRET, JWT_REFRESH_SECRET, INTERNAL_SERVICE_TOKEN
```

Rules:
- Paste the values into Railway Variables only. They never go into git or Vercel; the frontend
  has no secrets.
- Rotating `JWT_*` signs everyone out.
- Rotate `INTERNAL_SERVICE_TOKEN` in all four services at the same moment; requests fail with
  401 in between.

## 2. Databases (Neon)

1. Create one Neon project with two databases: `advancerag` (backend) and `advancerag_rag`
   (rag-core).
2. Use the **pooled** connection string (host contains `-pooler.`) with `?sslmode=require`.
   You can paste it exactly as Neon shows it:
   - `postgresql://` is converted to `postgresql+asyncpg://`.
   - libpq-only options such as `channel_binding` are dropped.
   - On the pooler, prepared-statement caching is disabled (`shared/advancerag_common/db.py`).
3. In production, both services refuse to start if a public database host has no
   `sslmode=require`, `verify-ca` or `verify-full`. Private Railway Postgres
   (`*.railway.internal`) is exempt.
4. Migrations run on container start (`alembic upgrade head`, `RUN_MIGRATIONS=1` by default). If
   a service ever runs more than one replica, set `RUN_MIGRATIONS=0` on all but one.

Verified locally: the backend suite (146 tests) passed over TLS PostgreSQL 16 both directly and
through PgBouncer 1.26 in transaction mode, and rag-core (45 tests) passed through PgBouncer.

## 3. Railway

Create one project with five services. Each service's settings point to its config file, and
the build context is the **repository root**:

| Service | Config as code | Public domain | Healthcheck |
|---|---|---|---|
| backend | `backend/railway.json` | yes | `/ready` |
| agent-core | `agent-core/railway.json` | no | `/ready` |
| rag-core | `rag-core/railway.json` | no | `/ready` (embedding model loads on start, 300 s timeout) |
| llm | `llm/railway.json` | no | `/ready` (lists FreeLLMAPI `/models`, no chat call) |
| Redis | Railway Redis plugin | no | — |

All services listen on `$PORT`. Uvicorn runs as PID 1 and gets 30 s for graceful shutdown.

### Environment matrix

Every internal service sets `ENVIRONMENT=production`, `HOST=::` (listen dual-stack; Railway's
private network can be IPv6-only) and `INTERNAL_SERVICE_TOKEN` (the same value everywhere). In
addition:

| Service | Variable | Value |
|---|---|---|
| backend | `DATABASE_URL` | Neon pooled URL for `advancerag`, `sslmode=require` |
| | `JWT_SECRET`, `JWT_REFRESH_SECRET` | from `generate_secrets.py` |
| | `FRONTEND_ORIGIN` | `https://<your-app>.vercel.app` (or your domain) |
| | `FRONTEND_PUBLIC_URL` | same; used in password-reset links |
| | `AGENT_CORE_URL` | `http://agent-core.railway.internal:<PORT>` |
| | `RAG_CORE_URL` | `http://rag-core.railway.internal:<PORT>` |
| | `TRUSTED_PROXY_HOPS` | `2` in proxy mode (Vercel rewrite → Railway edge); `1` in direct mode |
| | `RATE_LIMIT_BACKEND`, `REDIS_URL` | `redis`, and `${{Redis.REDIS_URL}}` |
| | `STORAGE_BACKEND=s3`, `STORAGE_ENDPOINT`, `STORAGE_REGION`, `STORAGE_BUCKET`, `STORAGE_ACCESS_KEY`, `STORAGE_SECRET_KEY` | R2: endpoint `https://<account>.r2.cloudflarestorage.com`, region `auto` |
| | `COOKIE_SAMESITE`, `COOKIE_DOMAIN` | see §4 |
| | `EMAIL_PROVIDER=resend`, `EMAIL_API_KEY`, `EMAIL_FROM` | password-reset email (see §5a); optional, but without it users cannot reset a forgotten password |
| agent-core | `RAG_CORE_URL` | `http://rag-core.railway.internal:<PORT>` |
| | `LLM_SERVICE_URL` | `http://llm.railway.internal:<PORT>` |
| rag-core | `DATABASE_URL` | Neon pooled URL for `advancerag_rag` |
| | `PINECONE_API_KEY` | your key |
| | `PINECONE_INDEX` | `advancerag-bge384` (frozen) |
| llm | `FREELLMAPI_BASE_URL`, `FREELLMAPI_API_KEY` | hosted FreeLLMAPI ([production-freellmapi.md](production-freellmapi.md)) |
| | `FREELLMAPI_ALLOW_RAILWAY_PRIVATE_HTTP=true` | only if FreeLLMAPI runs as a private Railway service |

Notes:
- Give each internal service a fixed `PORT` variable (for example 8001, 8002, 8003) so the
  private URLs are stable.
- Leave the RAG parameters (chunking, top-k, reranker, context limits) at their defaults. They
  are frozen (see [rag-architecture.md](rag-architecture.md)).
- No Gemini variables exist; the only LLM path is llm → FreeLLMAPI.

Each service refuses to start in production when:
- a database host is public and has no TLS;
- a JWT secret is weak or missing;
- the internal token is missing;
- the FreeLLMAPI URL is localhost or plain HTTP (except with the private-Railway flag);
- storage is local without the explicit override.

Fix the variable rather than relaxing the check.

## 4. Vercel and cookies

The refresh token is an HttpOnly, Secure cookie on path `/api/v1/auth`. The browser must see it
as same-site, so pick one of these modes:

**Proxy mode (no custom domain needed; recommended to start).** Vercel forwards `/api/*` to
Railway, so the browser only ever talks to the Vercel origin:

```
python scripts/configure_vercel.py --mode proxy --backend-url https://<backend>.up.railway.app
```

Leave `VITE_API_BASE_URL` empty in Vercel and set `TRUSTED_PROXY_HOPS=2` on the backend.

**Direct mode (custom domain).** Use, for example, `app.example.com` (Vercel) and
`api.example.com` (Railway custom domain):

```
python scripts/configure_vercel.py --mode direct --backend-url https://api.example.com
```

Then:
- set `VITE_API_BASE_URL=https://api.example.com` in Vercel;
- set `TRUSTED_PROXY_HOPS=1` and `FRONTEND_ORIGIN=https://app.example.com` on the backend.

Commit the generated `frontend/vercel.json`. In the Vercel project settings, set the root
directory to `frontend`, framework Vite, build `npm run build`, output `dist`. The file sets the
SPA fallback, the CSP and the security headers. A local check of the production build under
that CSP found 0 violations.

Do not deploy a direct-mode frontend on `*.vercel.app` against a `*.up.railway.app` backend.
Those are different sites, so the refresh cookie would be third-party and blocked by browsers.

## 5. Storage (S3 / Cloudflare R2)

1. Create the bucket once, keep it private, and leave versioning optional.
2. Use credentials scoped to that bucket only.
3. The backend's `/ready` reports `storage: false` until the bucket exists and the credentials
   work.

## 5a. Password-reset email (Resend)

Email is used only for forgot-password links. The flow:
1. `POST /api/v1/auth/forgot-password` stores a SHA-256 hash of a 256-bit random token (never the
   token itself) with a 30-minute expiry. Requesting a new link voids the older ones.
2. After the response, the backend emails `https://<frontend>/reset-password?token=<token>`
   through Resend.
3. `POST /api/v1/auth/reset-password` uses the token once, sets the new password and revokes
   every session and refresh token.

The forgot-password answer is identical whether or not the account exists.

**Setup**
1. Create a Resend account and add your sending domain (Resend → Domains). A subdomain such as
   `mail.example.com` is fine.
2. Add the DNS records Resend shows (SPF/MX and DKIM TXT), then wait until the domain is
   **Verified**.
   - Without a verified domain, Resend only delivers to your own account address, so real users
     get nothing.
3. Create an API key with **Sending access** only, restricted to that domain.
4. On the Railway **backend** service only, set:

| Variable | Value |
|---|---|
| `EMAIL_PROVIDER` | `resend` |
| `EMAIL_API_KEY` | the Resend key (`re_…`) |
| `EMAIL_FROM` | `AdvancedRAG <no-reply@mail.example.com>`, on the verified domain |
| `FRONTEND_PUBLIC_URL` | the exact public frontend origin, e.g. `https://app.example.com` (defaults to the first `FRONTEND_ORIGIN`) |
| `PASSWORD_RESET_TOKEN_EXPIRE_MINUTES` | optional, default `30` |

**Secret handling**
- The Resend key lives only in the backend's Railway variables. It is never a `VITE_*` variable
  and never in Vercel or the frontend; a test scans the frontend sources and bundle for this.
- The key is not logged and does not appear in errors or API responses.
- Logs never contain the reset token, the reset URL or the email body. A delivery failure is
  logged as `email_send_failed` / `password_reset_email_failed`, with only the request id, the
  HTTP status and Resend's error name (e.g. `validation_error`).
- Rotate the key by creating a new one in Resend, updating `EMAIL_API_KEY`, redeploying, then
  deleting the old key.

**Startup checks (staging/production).** The backend refuses to start when:
- `EMAIL_PROVIDER=resend` has no `EMAIL_API_KEY`;
- `EMAIL_FROM` is a localhost/placeholder domain;
- the frontend URL is not https;
- the provider is `mock` or `file`.

With no provider at all, it starts but logs `password_reset_email_disabled`.

**Rate limits** (Redis-backed, shared across replicas):
- `forgot-password`: 5 per hour per client IP, and 3 per hour per email address. The address is
  normalized, so case and spacing variants share one limit. Unknown addresses are limited exactly
  like real ones.
- `reset-password`: 10 attempts per hour per client IP.
- The client IP is taken from `X-Forwarded-For` only through the configured
  `TRUSTED_PROXY_HOPS`, so a spoofed header doesn't reset the limit.

**Delivery failures.** The email is sent after the response, so response timing doesn't reveal
whether the account exists. As a consequence:
- The user always sees the generic message.
- If Resend rejects or times out, the token is voided immediately (no usable token for an email
  that never arrived) and the failure is logged with the request id.
- The user can simply request again.
- Watch for `password_reset_email_failed` in the logs.

**Free tier.** Resend's free plan has daily and monthly sending caps (check resend.com/pricing for
current numbers). With 3 emails per address and 5 per IP per hour, abuse is bounded. Upgrade if
real usage approaches the cap; Resend returns 429 when it is exceeded, which shows up as
`email_send_failed status=429`.

**Smoke test (staging or production, after every email config change)**
1. Create a test account in the deployed app.
2. Log out, then choose **Forgot password?** and enter the address.
3. Confirm the email arrives from `EMAIL_FROM`, not in spam. Check the Resend dashboard → Emails
   if it doesn't.
4. Open the link; it must go to `https://<frontend>/reset-password?token=…`.
5. Set a new password.
6. Log in with the new password.
7. Confirm the old password fails.
8. Open the same link again and submit; it must show "This reset link is invalid or has expired."
9. In the Railway logs, search for the token value and `reset-password?token`: no matches. Search
   for `re_`: no key.

**Troubleshooting**

| Symptom | Likely cause |
|---|---|
| Backend won't start: `resend needs EMAIL_API_KEY` / `EMAIL_FROM must use your verified sending domain` / `need an https FRONTEND_PUBLIC_URL` | fix that variable |
| Log `email_send_failed status=403` or `error_name=validation_error` | domain not verified, or `EMAIL_FROM` not on the verified domain |
| `status=401` | wrong or revoked API key |
| `status=429` | Resend rate or plan limit |
| Email arrives but the link opens the wrong site | `FRONTEND_PUBLIC_URL` / `FRONTEND_ORIGIN` |
| Users see "Too many requests" | the per-address/IP limits above; wait an hour |
| No email and no error log | `EMAIL_PROVIDER` empty (look for `password_reset_email_disabled` at startup) or the address has no account |

Real Resend delivery is **not verified** until the smoke test above passes against your domain.

## 6. Monitoring

| Signal | Where | Use |
|---|---|---|
| `GET /health` | backend public domain | uptime monitor (liveness, no dependencies) |
| `GET /ready` | backend public domain | dependency monitor: `database`, `storage`, `agent_core`, `rag_core` booleans only; 503 if any fails |
| `GET /metrics` | each internal service, with `Authorization: Bearer $INTERNAL_SERVICE_TOKEN` | request, latency and error counters; never expose publicly |
| JSON logs | Railway log explorer | search by `request_id` (propagated backend → agent-core → rag-core/llm). No secrets or document text are logged |

What to alert on:
- `/ready` failing for more than 2 minutes;
- a sustained rise in `LLM_RATE_LIMITED` or `LLM_UNAVAILABLE`. That is FreeLLMAPI capacity, not
  an AdvancedRAG fault;
- documents stuck in `processing` for longer than the largest expected document (about 20 min
  on a small CPU for about 5,000 chunks).
  - RAG Core's `GET /api/v1/rag/ingestions/{id}` shows the stage and heartbeat age.
  - Search the logs for `ingestion_succeeded`, `ingestion_failed`, `ingestion_cancelled` and
    `stale_jobs_recovered`.

Ingestion settings (defaults are fine; frozen RAG parameters are unaffected):

| Service | Variable | Default | Meaning |
|---|---|---|---|
| backend | `INGEST_POLL_INTERVAL_SECONDS` | 2 | how often the worker polls RAG Core |
| backend | `INGEST_MAX_SECONDS` | 3600 | ceiling per document, counted in active (awake) time |
| backend | `JOB_STALE_AFTER_SECONDS` / `STALE_SWEEP_INTERVAL_SECONDS` | 300 / 60 | lease length / how often dead leases are recovered |
| backend | `RAG_CORE_TIMEOUT_SECONDS` | 60 | per HTTP call to RAG Core (start, status, delete), not the ingestion |
| rag-core | `INGEST_CONCURRENCY` | 2 | ingestions running at once per RAG Core process |
| rag-core | `INGEST_HEARTBEAT_SECONDS` / `INGEST_STALE_SECONDS` | 10 / 120 | attempt heartbeat / when a silent attempt counts as dead |

RAG Core's migration `0002` (table `rag_ingestions`) runs automatically on container start.

## 7. Post-deploy smoke test

1. `/ready` returns 200 with all four checks true.
2. Register and log in, then upload a small `.txt` file and wait for `ready`.
3. Ask a question about the file. The answer should stream and show a numbered citation chip.
4. Delete the document. Its Pinecone vectors are removed.
5. Log out and log back in. The history persists.
