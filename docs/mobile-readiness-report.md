# Mobile app — Phase 1: API and deployment-readiness report

Date: 2026-10-09. Scope: is the existing AdvancedRAG backend ready to serve an Android app?
No AI logic was rewritten for this audit. Everything marked **verified** was executed; the few
statements based only on reading code say so.

## 1. Verdict

| Area | Status |
|---|---|
| Backend API for a mobile client (auth, chats, messages, Knowledge Base, CSV analysis, follow-ups, share, isolation) | **Verified** on the real stack: 49/49 functional checks (plus 1 corrected check, §3) |
| Production configuration of all services (production mode, private networking, TLS rules, Redis, S3, Pinecone) | **Verified** in the local production rehearsal: 26/26 |
| LLM in production | **Blocker.** The only configured LLM is FreeLLMAPI on `http://127.0.0.1:31415/v1` (your PC). A production backend cannot reach it, and the LLM service correctly refuses to start with it in production. No hosted FreeLLMAPI exists yet. |
| Mobile-specific integration | **3 gaps to decide before building the app** (§4: refresh token is cookie-only, entry point / proxy hops, streaming client choice) |
| Production accounts (Railway, Neon, R2, Resend, hosted FreeLLMAPI) | **Not verified**: none were provided or deployed during this audit |

**Do not release an APK** until §4 G1 (hosted LLM) is done and the post-deploy checks in §7 pass
against the real production URL.

## 2. Repository and services

| Service | Port (dev) | Role | Talks to |
|---|---|---|---|
| backend (FastAPI) | 8000 | the only public API: auth, chats, documents, shares, SSE | PostgreSQL `advancerag`, storage (local / S3), agent-core, rag-core, Redis (prod) |
| agent-core | 8001 | per-turn orchestration: classification, retrieval, grading, verification, follow-ups | rag-core, llm |
| rag-core | 8002 | ingestion (parse, chunk, BGE embeddings), retrieval, CSV tables | PostgreSQL `advancerag_rag`, Pinecone `advancerag-bge384` |
| llm | 8003 | the only holder of the LLM key | FreeLLMAPI (`/v1/chat/completions`, `/v1/models`) |
| frontend (Vite/React) | 5173 | web app | backend only |

Only the backend is public. The other three are private and require `INTERNAL_SERVICE_TOKEN`
(verified in the rehearsal: each returns 401 without it, from inside the private network).

## 3. What was verified, and how

**Live API audit** (scratch script, not committed): the new code on alternate ports, called over
HTTP like a native client with a cookie jar. Real: PostgreSQL 16 (test databases),
BAAI/bge-small-en-v1.5 (384 dims), Pinecone `advancerag-bge384` and its reranker, FreeLLMAPI
(desktop). Synthetic users only; everything was deleted afterwards (0 rows, 0 vectors left).

| Area | Checks | Result | Measured |
|---|---|---|---|
| Health | `/ready` of all four services, `/health` | pass | backend /ready 1.4 s |
| Auth | register, login, wrong password (401), `/auth/me`, refresh with rotation, refresh-token reuse, forgot-password (generic answer), unauthenticated call (401), logout, logout-all | pass | register 99 ms, login 82 ms, refresh 13 ms |
| Profile / settings | get + patch both; clear all data | pass | |
| Conversations | create, list, get, rename, delete, search | pass | 13–35 ms |
| Messages | SSE stream, non-streaming send, history, paging (`order=desc`), retry of a completed answer (409 as designed) | pass | general answer: first token 0.9 s; KB answer: first token 4.0 s, done 4.9 s |
| Knowledge Base documents | multipart upload, poll to ready, list, download, unsupported type (415), failed ingestion reported, reprocess, delete | pass | 3-page PDF ready in 5.1 s; upload returns in 30 ms |
| Ingestion / retrieval | chunks in DB = vectors in Pinecone (fetched by id); KB question in a new chat with citations (pages 2 and 3); absent fact → exact abstain answer | pass | |
| Delete | after `DELETE /documents/{id}`: 0 chunks, 0 Pinecone vectors | pass | |
| CSV analysis | 5,000-row CSV → stored table; exact distribution over all rows with `:table` citation; follow-up "give above in table format" → exact table, no retrieval | pass | CSV ready in 29.6 s; analysis 1.8 s; follow-up 31 ms |
| Share | create link, public read without auth, status, revoke (link then 404) | pass | |
| Isolation | another user: conversation, messages, document, download, delete, post → all 404 | pass | |

One check of mine was wrong at first: replaying an old refresh token immediately was accepted.
That is by design: a 30-second grace window (`REFRESH_REUSE_GRACE_SECONDS`) for two requests
refreshing at once. It returns only an access token and never a new refresh token. Verified live
separately: a replay after 31 s returns 401 `REFRESH_TOKEN_REUSED` and revokes the whole session.

**Production rehearsal** (`python scripts/rehearse_production.py`): the four Docker images with
`ENVIRONMENT=production`, private service names, Redis, S3-compatible storage (SeaweedFS) and
**real Pinecone**; FreeLLMAPI is the only stand-in (mock gateway). **26/26 passed**, including:
- the refresh cookie is HttpOnly + Secure;
- the S3 upload, ingestion to Pinecone and the SSE answer with a citation;
- Redis-backed rate limits (login brute force → 429);
- no secret appears in any log;
- the LLM service refuses a localhost FreeLLMAPI URL in production;
- the backend refuses a weak JWT secret.

Fixed during this audit: the rehearsal's "Pinecone vectors removed on delete" check compared
index-wide statistics, which lag behind writes. It passed with `before=1538 after=1538` and proved
nothing. It now fetches the document's own vector ids in the user's namespace: 1 after ingestion,
0 after delete.

**Automated suites** (latest runs today): agent-core 199, backend 199 (+4 skipped), rag-core 65
SQLite / 69 PostgreSQL, contract 52, frontend e2e 21. All pass. These use fakes or mocks where
noted in each suite. One backend ingestion-lease test failed once and passed on rerun
(intermittent, timing-related).

## 4. Gaps for a mobile client

### G1 — BLOCKER: the LLM endpoint is your PC

`llm/.env` has `FREELLMAPI_BASE_URL=http://127.0.0.1:31415/v1`: the FreeLLMAPI desktop app on
this computer. When the PC is off, or from any cloud host, every chat turn fails
(`LLM_UNAVAILABLE`). Production startup already refuses this URL (verified), so this cannot slip
through silently, but it means production cannot run until one of these exists:

- **Option A (recommended, documented):** FreeLLMAPI's server image
  (`ghcr.io/tashfeenahmed/freellmapi`) as a **private Railway service** with a persistent volume.
  Set `FREELLMAPI_BASE_URL=http://<service>.railway.internal:3001/v1` and
  `FREELLMAPI_ALLOW_RAILWAY_PRIVATE_HTTP=true`. Provider keys must be re-entered in the hosted
  dashboard. See [production-freellmapi.md](production-freellmapi.md) §4.
- **Option B:** FreeLLMAPI on any host with a public **HTTPS** endpoint and a strong unified key.

Not verified and not claimed: whether the FreeLLMAPI container is reachable over Railway's
private network, and how it handles Railway's `PORT`. The LLM service supports one provider type
(`LLM_PROVIDER=freellmapi`, an OpenAI-compatible client). Pointing it at a different gateway has
**not** been tested.

Done when: `python -m llm_service.healthcheck --chat` succeeds from the production LLM service,
and agent-core `/ready` reports `llm: true` in production.

### G2 — Refresh token is cookie-only (decision needed)

Login and register return `access_token` (15 min) in the body. The refresh token (7 days,
rotated on every use) is **only** an HttpOnly cookie on path `/api/v1/auth`. Sending it in a
JSON body is not accepted (verified: 401).

A native app therefore either:
1. **keeps the cookie:** an HTTP client with a persistent cookie store (OkHttp `CookieJar`,
   Dio + cookie manager, React Native's native cookie store). It must persist across app restarts,
   or users sign in again every 15 minutes. Works today; verified with a cookie-jar client. Or:
2. **gets a mobile token mode (backend change, Phase 2):** for example a header-selected mode
   where login/refresh return the refresh token in the body and `/auth/refresh` and `/auth/logout`
   accept it in the body. The app stores it in the Android Keystore (EncryptedSharedPreferences
   / SecureStore). Rotation, the grace window and reuse detection stay unchanged.

Option 2 is the usual choice for mobile and avoids cookie-store edge cases, but it is an API
change. Decide before writing the app's auth layer.

### G3 — One entry point for web and mobile

Rate limits key on the client IP, taken `TRUSTED_PROXY_HOPS` entries from the right of
`X-Forwarded-For`. The web runbook uses proxy mode (Vercel → Railway, `TRUSTED_PROXY_HOPS=2`).
A mobile app calling the Railway URL **directly** passes only one proxy, so with `2`:
- without an `X-Forwarded-For` header, every mobile user shares one rate-limit bucket (the Railway
  edge address), so 10 registrations per hour across **all** mobile users;
- with a forged header, a client chooses its own rate-limit key (login brute-force limits
  bypassed).

**Requirement:** the app must call the same origin as the web app (the Vercel proxy domain), or
both must use direct mode with one custom API domain (`TRUSTED_PROXY_HOPS=1`). Never mix the two.

Related, for carrier-grade NAT: many mobile users can share one public IP. The per-IP limits are
register 10/hour, login 10/5 min, forgot-password 5/hour and refresh 60/min (plus per-email limits).
Monitor 429s after launch.

### G4 — Streaming on Android

`POST /conversations/{id}/messages/stream` is SSE over a **POST** with an `Authorization` header.
Browser-style `EventSource` cannot do that. Use a streaming HTTP client:
- OkHttp: read the response body line by line;
- React Native / Expo: `expo/fetch` streaming, or a POST-capable SSE library;
- Flutter: `http` with a streamed response.

Handle the events `content.delta`, `content.replace` (the verified answer replaces the draft),
`message.completed` and `error`. The non-streaming `POST /conversations/{id}/messages` works as a
fallback (verified). Responses send `Cache-Control: no-cache` and `X-Accel-Buffering: no` (from code, `messages.py`).

### G5 — Links that point at the web app

Share links and password-reset links use `FRONTEND_PUBLIC_URL` (dev: `http://localhost:5173`). The
mobile app needs the web app deployed at a public HTTPS URL for these, or Android App Links to
open them in the app.

### Smaller notes

- **CORS** applies only to browsers. Native requests need no change (a WebView would).
- **Upload size:** up to 25 MB, multipart field `file`, types pdf/txt/md/csv/docx. Ingestion is
  asynchronous: poll `GET /documents/{id}` until `ready` or `failed` (5 s for a small PDF, 30 s
  for a 5,000-row CSV here). Big PDFs take minutes; show progress and allow leaving the screen.
  Not verified: whether the Vercel proxy accepts 25 MB request bodies. Test a maximum-size upload
  through the production entry point.
- **OpenAPI** is disabled in production (from code: `docs_enabled = not settings.is_production`). Generate
  the mobile API client from a development backend (`http://localhost:8000/api/v1/openapi.json`).
- **Errors:** every error is `{"error": {"code", "message", ...}}` with stable codes
  (`UNAUTHORIZED`, `INVALID_CREDENTIALS`, `REFRESH_TOKEN_REUSED`, `UNSUPPORTED_FILE_TYPE`, …).
  On 401 the app should refresh once, then send the user to sign-in.

## 5. API inventory for the mobile app

All under `/api/v1`. Authorization: `Bearer <access_token>`, except where noted.

| Feature | Method and path | Notes |
|---|---|---|
| Sign up | `POST /auth/register` `{email, password, full_name}` | 201, access token + refresh cookie |
| Sign in | `POST /auth/login` `{email, password}` | |
| Refresh | `POST /auth/refresh` | refresh cookie (G2); rotates it |
| Current user | `GET /auth/me` | |
| Sign out / everywhere | `POST /auth/logout`, `POST /auth/logout-all` | |
| Password reset | `POST /auth/forgot-password`, `POST /auth/reset-password` | no auth; the email link opens the web app (G5) |
| Profile | `GET`/`PATCH /profile`, `DELETE /profile/data` | clear all data |
| Settings | `GET`/`PATCH /settings` | theme |
| Chats | `GET /conversations` (paged), `POST /conversations`, `GET`/`PATCH`/`DELETE /conversations/{id}` | |
| Search chats | `GET /search/chats?q=` | |
| Messages | `GET /conversations/{id}/messages?limit&offset&order` | history with citations |
| Send (streaming) | `POST /conversations/{id}/messages/stream` `{content}` | SSE (G4) |
| Send (non-streaming) | `POST /conversations/{id}/messages` `{content}` | 201 with both messages |
| Retry failed answer | `POST /conversations/{id}/messages/{message_id}/retry` | SSE; only failed or cancelled answers |
| Knowledge Base | `GET /documents` (paged), `POST /documents` (multipart `file`), `GET /documents/{id}`, `GET /documents/{id}/content`, `POST /documents/{id}/reprocess`, `DELETE /documents/{id}` | searched in every chat; nothing to attach |
| Share | `POST`/`GET`/`DELETE /conversations/{id}/share` | |
| Shared view | `GET /shares/{token}` | public, no auth |
| Health | `GET /health`, `GET /ready` (outside `/api/v1`) | for app status checks |

Not needed by the app: `GET`/`POST`/`DELETE /conversations/{id}/documents` (legacy attachments;
they no longer affect retrieval).

## 6. Deployment dependencies and production reachability

| Dependency | Used by | Production target | Status |
|---|---|---|---|
| PostgreSQL ×2 | backend, rag-core | Neon (pooled, `sslmode=require`) | config rules verified in rehearsal (local Postgres); Neon itself not provisioned or verified |
| Object storage | backend | S3 / Cloudflare R2 | verified with S3-compatible storage in rehearsal; real R2 bucket not verified |
| Redis | backend rate limits | Railway Redis | verified in rehearsal |
| Pinecone `advancerag-bge384` (384 dims) | rag-core | Pinecone cloud (HTTPS) | **reachable and verified** (live and rehearsal: upsert, fetch, rerank, delete) |
| BAAI/bge-small-en-v1.5 | rag-core | baked into the rag-core image | verified (image builds; `/ready` shows the model loaded) |
| **FreeLLMAPI** | llm | hosted FreeLLMAPI (G1) | **not reachable from production: blocker** |
| Resend (email) | backend password reset | Resend API | not verified (no email sent in this audit) |
| Web app | share and reset links | Vercel | not deployed / not verified |

## 7. Before releasing the APK

1. Host FreeLLMAPI (G1) and pass `python -m llm_service.healthcheck --chat` in production.
2. Deploy per [production-deployment.md](production-deployment.md). Choose the single entry point
   (G3) and set `TRUSTED_PROXY_HOPS` to match.
3. Decide G2 (cookie store vs mobile token mode). If mode 2, implement it with tests first.
4. Run the post-deploy smoke test ([production-deployment.md](production-deployment.md) §7)
   against the production URL. Repeat this report's §3 checks there with a synthetic account,
   including a 25 MB upload and a streamed answer over mobile data.
5. Bake the production API base URL into the app's release build. Debug builds may use a dev URL.
