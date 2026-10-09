# FreeLLMAPI in production

AdvancedRAG generates answers through FreeLLMAPI, an OpenAI-compatible gateway that routes each request to an upstream provider (OpenRouter, Groq, Google AI Studio, and others). This document covers how the LLM service talks to FreeLLMAPI on your machine and in production, and what you have to host yourself.

> **Status:** the AdvancedRAG side is production-ready. **Hosting FreeLLMAPI itself is a separate infrastructure task.** Its server is not part of this repository. Until a FreeLLMAPI instance is reachable over HTTPS (or Railway's private network) and its URL and key are configured, the LLM service will not start with `ENVIRONMENT=production`. This is by design.

## 1. Architecture

```
Browser ──HTTPS──► Backend ──► Agent Core ──► LLM Service ──HTTPS + unified key──► FreeLLMAPI ──► upstream provider(s)
                                               (Railway)                         (you host it)     (OpenRouter, Groq, ...)
```

Rules:

- Only the **LLM service** talks to FreeLLMAPI. The browser, backend, Agent Core and RAG Core never do.
- **Provider keys stay inside FreeLLMAPI.** AdvancedRAG holds one secret for it, the FreeLLMAPI *unified* API key, and only in the LLM service's environment.
- FreeLLMAPI's OpenAI-format stream is converted by the LLM service into AdvancedRAG's internal `token`/`done`/`error` events. The browser only ever sees the backend's events (`message.created`, `content.delta`, `message.completed`, ...).
- Embeddings never go through FreeLLMAPI. RAG Core embeds locally (bge-small) and uses Pinecone.

## 2. Local development

| Variable | Value |
|---|---|
| `ENVIRONMENT` | `development` |
| `FREELLMAPI_BASE_URL` | `http://127.0.0.1:31415/v1` (the desktop app; this is also the default when left empty) |
| `FREELLMAPI_API_KEY` | `<local-unified-key>` from the FreeLLMAPI dashboard → **Keys** |
| `FREELLMAPI_MODEL` | `auto` |

Start the FreeLLMAPI desktop app and add at least one provider key in **its** dashboard. Then check the connection:

```
cd llm
python -m llm_service.healthcheck          # config, connection, auth, /v1/models, configured model
python -m llm_service.healthcheck --chat   # plus ONE real completion (costs a request)
```

## 3. Production

| Variable | Value |
|---|---|
| `ENVIRONMENT` | `production` |
| `FREELLMAPI_BASE_URL` | `https://<reachable-freellmapi-host>/v1` |
| `FREELLMAPI_API_KEY` | `<production-unified-key>` |
| `FREELLMAPI_MODEL` | `auto` |
| `INTERNAL_SERVICE_TOKEN` | at least 32 random characters, the same value in backend, Agent Core, RAG Core and the LLM service |

Optional settings: `LLM_TIMEOUT_SECONDS` (90), `LLM_CONNECT_TIMEOUT_SECONDS` (5), `LLM_MAX_RETRIES` (2), `LLM_RETRY_BACKOFF_SECONDS` (1), `LLM_MAX_BACKOFF_SECONDS` (8), `LLM_MAX_RETRY_AFTER_SECONDS` (10) and `FREELLMAPI_STREAM_USAGE` (false).

### Startup validation

With `ENVIRONMENT=production` (or `staging`), the service **refuses to start** if any of these is true:

- `FREELLMAPI_BASE_URL` is missing. There is no fallback to localhost.
- `FREELLMAPI_BASE_URL` is local: `localhost`, `*.localhost`, `127.0.0.0/8`, `0.0.0.0`, `::1`, link-local addresses or `host.docker.internal`.
- `FREELLMAPI_BASE_URL` is plain `http://`. The single exception is the Railway private-network opt-in below.
- `FREELLMAPI_API_KEY` is missing.
- `INTERNAL_SERVICE_TOKEN` is missing or shorter than 32 characters.

In every environment, a URL that is not an absolute `http(s)://host/...` URL is also rejected.

Error messages name the setting and the rule but never print a value. For example:

```
LLM configuration invalid: FREELLMAPI_BASE_URL must be an HTTPS non-local endpoint in production: it points to localhost, ...
```

### Model

`auto` lets FreeLLMAPI's router choose. Your desktop instance lists `auto` in `/v1/models` (checked). `auto:<profile>` routing profiles, such as `auto:fast`, are accepted by the router but are **not** listed, so readiness accepts them only when the gateway lists `auto`. Any other value must be a model id that `/v1/models` lists. AdvancedRAG contains no provider-specific model logic.

## 4. Hosting FreeLLMAPI (external infrastructure)

The desktop app installed on this machine (`freellmapi-desktop`, an Electron tray app) **cannot be the production server**. It runs on your PC, it is meant to listen on loopback, and Railway cannot reach it.

The upstream project, [tashfeenahmed/freellmapi](https://github.com/tashfeenahmed/freellmapi), also ships a **server/Docker distribution**. Everything in this section comes from its install guide ([`docs/en/install/01-install.md`](https://github.com/tashfeenahmed/freellmapi/blob/main/docs/en/install/01-install.md)). Read that guide before deploying, because upstream details can change.

| Item | Upstream documentation |
|---|---|
| Image | `ghcr.io/tashfeenahmed/freellmapi:latest` (amd64 + arm64) |
| Port | `3001` (`PORT`) |
| Required secret | `ENCRYPTION_KEY`: 64 hex characters (`openssl rand -hex 32`). It encrypts the stored provider keys (AES-256-GCM). |
| Persistent data | SQLite at `/app/server/data` (or `FREEAPI_DB_PATH`). **It must be on a persistent volume**, or provider keys and accounts are lost on redeploy. |
| Dashboard account | Server installs use email and password. **The first account can be claimed by whoever reaches the dashboard first**, unless it is provisioned through `FREEAPI_CONFIG_JSON` / `FREEAPI_CONFIG_PATH` (`"admin": {"email": ..., "password": ...}`). |
| Security model | Upstream describes it as "single-user and guarded only by the unified API key". The dashboard is served on the same port as the API. |
| Behind a proxy / TLS | Forward `X-Forwarded-Proto`. |
| Health endpoint | **Not documented.** AdvancedRAG checks it through authenticated `GET /v1/models`. |
| Backups | Optional encrypted backups (`FREEAPI_DB_BACKUP_*`) |

Provider keys must be entered again in the hosted instance's dashboard. Do **not** copy the desktop app's data folder, because it is encrypted with a different key.

### Option A (recommended): FreeLLMAPI as a private Railway service

This option exposes neither the dashboard nor the API to the internet.

1. Create a Railway service from the image `ghcr.io/tashfeenahmed/freellmapi:latest`.
2. Attach a **volume** at `/app/server/data`.
3. Set `ENCRYPTION_KEY`, `PORT=3001` and `FREEAPI_CONFIG_JSON` containing the admin account. Do this **before** the first start, so nobody else can claim the dashboard.
4. **Do not generate a public domain.** To use the dashboard (to add provider keys and copy the unified key), temporarily use a public domain, or `railway connect`/port-forwarding if available. Remove the public domain afterwards.
5. In the LLM service, set:
   ```
   FREELLMAPI_BASE_URL=http://<freellmapi-service>.railway.internal:3001/v1
   FREELLMAPI_ALLOW_RAILWAY_PRIVATE_HTTP=true
   ```
   This opt-in allows plain http **only** for `*.railway.internal` hosts, whose traffic stays inside Railway's encrypted private network. Every other plain-http URL, and localhost, is still refused.

Checked locally on 2026-10-09 with the image `ghcr.io/tashfeenahmed/freellmapi:latest` (built 2026-10-07, commit `ffef850`):
- it honours `PORT` (tested with 4567) and listens on `[::]` with `bindv6only=0`, so it accepts IPv6 and IPv4. That suits Railway's private network;
- `GET /health` returns 200 without a key, so use it as the Railway health check. `GET /v1/models` returns 401 without the unified key;
- `FREEAPI_CONFIG_JSON` with an `admin` object creates the dashboard account before the listener starts. On restart (same volume and `ENCRYPTION_KEY`) the account and the unified key are kept;
- data lives in `/app/server/data` (`freeapi.db`), which must be the Railway volume.

Still to verify on Railway itself: the private-network connection from the LLM service (`python -m llm_service.healthcheck --chat` from a shell on the LLM service).

### Option B: FreeLLMAPI on any host with a public HTTPS endpoint

Use this for Railway with a public domain, a VPS, Fly or similar.

- **Use HTTPS only.** Put TLS in front, either through the platform or a reverse proxy, and forward `X-Forwarded-Proto`.
- Provision the admin account through declarative config **before** the instance is reachable, and use a strong password.
- Restrict the dashboard paths (everything outside `/v1`) by IP allow-list or proxy authentication where your platform supports it. The upstream image is single-user.
- Use a long unified key and rotate it if it leaks. It is the only thing protecting `/v1`.
- Set `FREELLMAPI_BASE_URL=https://<host>/v1` in the LLM service.

**Never expose the desktop app to the internet** (for example with a tunnel). Railway must not depend on your PC being on.

## 5. Security model

| Secret | Lives in | Never in |
|---|---|---|
| Provider keys (OpenRouter, Groq, ...) | FreeLLMAPI only, encrypted in its database | anywhere in AdvancedRAG |
| FreeLLMAPI unified key | LLM service environment (`FREELLMAPI_API_KEY`) | frontend / `VITE_*`, backend, Agent Core, RAG Core, PostgreSQL, logs, errors, `/ready`, metrics |
| `INTERNAL_SERVICE_TOKEN` | backend, Agent Core, RAG Core, LLM service | frontend |

How this is enforced in code:

- The key is a `SecretStr` and is sent only as `Authorization: Bearer …` to FreeLLMAPI.
- Logs pass through a redacting formatter (Bearer tokens, JWTs, `key=`/`token=` pairs). httpx is logged at WARNING and above.
- Errors expose a safe `code` and message, never provider bodies, URLs or headers.
- `/ready` returns booleans plus a generic detail string.
- The `healthcheck` CLI prints the URL without any credentials or query string.

All of this is covered by `llm/tests/test_production_freellmapi.py`.

## 6. Railway configuration (LLM service)

[`llm/railway.json`](../llm/railway.json): set it as the service's config-as-code path (`/llm/railway.json`).

- Builds `llm/Dockerfile` from the repository root.
- Health check: `GET /ready`, which must pass before a deploy goes live. It checks FreeLLMAPI connectivity, authentication, the `/v1/models` format and the configured model. It **never** runs a chat completion.
- The container command is `exec uvicorn ... --port ${PORT:-8003} --timeout-graceful-shutdown 30`. It listens on Railway's `PORT`, receives `SIGTERM` directly and lets in-flight streams finish.
- Runs as a non-root user, with no secrets baked into the image. All configuration comes from runtime variables.
- Keep the LLM service **private** (no public domain). Agent Core reaches it at `http://<llm-service>.railway.internal:<port>` with `INTERNAL_SERVICE_TOKEN`.

## 7. Health checks

| Check | What it does | When it runs |
|---|---|---|
| `GET /health` | Liveness only (the process is up) | Docker `HEALTHCHECK` |
| `GET /ready` | Key configured, FreeLLMAPI reachable, key accepted, `/v1/models` is OpenAI-compatible JSON, configured model offered | Platform deploy health check, Agent Core `/ready` |
| `python -m llm_service.healthcheck --chat` | All of the above plus **one real completion** | Manually, when diagnosing. Never automatic. |

The `/ready` response has the shape `{"status", "provider", "default_model", "checks": {configured, reachable, authenticated, models_response_valid, model_available}, "detail"}`.

## 8. Timeouts, retries and errors

- **Timeouts:** connect 5 s, read 90 s (LLM service → FreeLLMAPI); Agent Core → LLM service 120 s. None are infinite.
- **Retried** (at most `LLM_MAX_RETRIES`=2 extra attempts, exponential backoff 1 s → 2 s → … capped at 8 s, `Retry-After` honoured up to 10 s): connection failures and connect timeouts, and HTTP 429/500/502/503/504.
- **Never retried:** 400/401/403/404/413/422, read timeouts (the model is slow, not down) and failures after the first streamed token (to avoid duplicate output).

| FreeLLMAPI outcome | Internal code |
|---|---|
| 401 / 403 | `LLM_AUTH_FAILED` |
| 429 | `LLM_RATE_LIMITED` |
| 5xx | `LLM_PROVIDER_UNAVAILABLE` |
| read timeout | `LLM_TIMEOUT` |
| connection failure | `LLM_UNAVAILABLE` |
| non-OpenAI JSON/SSE | `LLM_INVALID_RESPONSE` |
| 400/404/413/422 | `LLM_BAD_REQUEST` |

Agent Core passes `LLM_TIMEOUT` and `LLM_RATE_LIMITED` through to the user and shows every other code as "temporarily unavailable".

## 9. Observability

Prometheus text at `GET /metrics` (requires the internal token):

- `llm_call_ms{kind, outcome, model}`: latency histogram. Its `_count` gives request, success and error counts per outcome. Streams record `ok`, an error code or `cancelled`.
- `llm_retries_total{reason}`: for example `http_503` or `connection_error`.
- `llm_upstream_responses_total{status_class}`: `2xx`/`4xx`/`5xx` from FreeLLMAPI.
- `llm_timeouts_total{kind}`.

Logs are JSON lines with `request_id`. The request ID flows backend → Agent Core → LLM service → FreeLLMAPI as `X-Request-ID`. Prompts, responses, keys and Authorization headers are never logged.

## 10. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Startup: `FREELLMAPI_BASE_URL is required` | Set it. Production never defaults to localhost. |
| Startup: `must be an HTTPS non-local endpoint` | You used the desktop URL or plain http. Point it to the hosted instance. |
| `/ready` → `reachable: false` | Wrong URL or port, the instance is down, or private networking is not reachable (see Option A notes). |
| `/ready` → `authenticated: false` | Wrong or rotated unified key. Copy it again from the hosted instance's **Keys** page. |
| `/ready` → `model_available: false` | `FREELLMAPI_MODEL` isn't listed by `/v1/models` (typo, or a provider was removed). Use `auto` or a listed id. |
| `/ready` → `models_response_valid: false` | The URL points at something that isn't the API, such as a login page or proxy error page. Check that it ends in `/v1`. |
| Chat fails with "busy" (`LLM_RATE_LIMITED`) | The upstream free tiers are exhausted. Add keys in FreeLLMAPI or wait. |
| `503 no_providers_configured` from FreeLLMAPI | No provider keys have been added in the FreeLLMAPI dashboard. |

## 11. Local vs production

| | Local | Production |
|---|---|---|
| FreeLLMAPI | Desktop app on your PC | Hosted server image (you operate it) |
| URL | `http://127.0.0.1:31415/v1` | `https://…/v1`, or `http://*.railway.internal` with the opt-in |
| Validation | Any absolute http(s) URL | HTTPS, non-local, URL and key required |
| Provider keys | Desktop dashboard | The hosted instance's dashboard (entered again) |
| Depends on your PC | Yes | **No** |

## 12. Verification procedure

1. **Local (desktop):** `python -m llm_service.healthcheck --chat` in `llm/`, then send a chat in the app.
2. **Hosted FreeLLMAPI, from your machine:** call `GET https://<host>/v1/models` with `Authorization: Bearer <key>`. Expect 200 with JSON containing `auto`.
3. **Railway LLM service:** the deploy health check (`/ready`) must pass. Then run `python -m llm_service.healthcheck --chat` from a Railway shell on the service.
4. **End to end:** sign in to the deployed frontend and send one plain and one document-grounded message. The answer should stream and show citations. Check that `llm_call_ms_count{outcome="ok"}` increased.

Only step 3 and step 4 against the real hosted instance count as **production verified**. The mock gateway tests (`llm/tests/test_mock_freellmapi_integration.py`) prove the contract, not connectivity.
