# AdvancedRAG API Contract (v1)

This file describes the API as it is implemented in `backend/app`. The test suite in `backend/tests` checks it, and the frontend client in `frontend/src/api.ts` is built against it. In development, the live OpenAPI docs are at `/docs` and `/redoc`.

- **Base URL:** `/api/v1`. In development the Vite server proxies `/api` to the backend, so the frontend and API share one origin.
- **Format:** request and response bodies are JSON unless stated otherwise. Timestamps are ISO-8601 in UTC, and IDs are UUIDs.
- **Auth:** send `Authorization: Bearer <access_token>`. The refresh token is an `HttpOnly` cookie named `refresh_token` with `Path=/api/v1/auth` and `SameSite=Lax`. It is `Secure` everywhere except development, and JavaScript can never read it.
- **Correlation:** you may send `X-Request-ID` (8–64 characters from `[A-Za-z0-9._-]`). The response always echoes a request ID, and the backend forwards it to Agent Core and RAG Core.

## Errors

Every error uses the same shape:

```json
{ "error": { "code": "INVALID_CREDENTIALS", "message": "Invalid email or password.", "request_id": "…", "details": null } }
```

The response never contains stack traces, SQL, internal URLs, or the values you submitted. For example, 422 errors list `details: [{field, message, type}]` but never echo the input.

| Status | Typical codes |
|---|---|
| 400 | `INVALID_TOKEN`, `INVALID_CURRENT_PASSWORD`, `PASSWORD_UNCHANGED` |
| 401 | `UNAUTHORIZED`, `INVALID_TOKEN`, `INVALID_CREDENTIALS`, `SESSION_EXPIRED`, `REFRESH_TOKEN_REUSED` |
| 404 | `NOT_FOUND`, `CONVERSATION_NOT_FOUND`, `DOCUMENT_NOT_FOUND`, `MESSAGE_NOT_FOUND`, `SHARE_NOT_FOUND`. Another user's resource also returns 404, never 403. |
| 409 | `EMAIL_ALREADY_REGISTERED`, `DOCUMENT_QUOTA_EXCEEDED`, `STORAGE_QUOTA_EXCEEDED`, `MESSAGE_NOT_RETRYABLE`, `DOCUMENT_NOT_RETRYABLE` |
| 413 | `PAYLOAD_TOO_LARGE`, `FILE_TOO_LARGE` |
| 415 | `UNSUPPORTED_FILE_TYPE`, `FILE_CONTENT_MISMATCH` |
| 422 | `VALIDATION_ERROR`, `EMPTY_FILE` |
| 429 | `RATE_LIMITED`, sent with a `Retry-After` header |
| 500 | `INTERNAL_SERVER_ERROR` |
| 502/503 | `AGENT_SERVICE_ERROR`, `AGENT_SERVICE_UNAVAILABLE`, `RAG_SERVICE_UNAVAILABLE`, `LLM_UNAVAILABLE`, `LLM_TIMEOUT`, `LLM_RATE_LIMITED`, `GENERATION_FAILED`, `EMPTY_RESPONSE`, `RATE_LIMITER_UNAVAILABLE` (only outside development, when Redis is down: the limiter fails closed) |

Endpoints that return a list return a page:

```json
{ "items": [...], "total": 42, "limit": 50, "offset": 0 }
```

## Shared shapes

```text
User          { id, email, full_name|null, is_verified, plan, created_at }
TokenResponse { access_token, token_type: "bearer", expires_in (seconds), user: User }
Conversation  { id, title, created_at, updated_at, archived_at|null }
Message       { id, conversation_id, role: "user"|"assistant", content,
                status: "pending"|"streaming"|"completed"|"failed"|"cancelled",
                attachments: [{id, filename}], citations: [{document_id, filename, page, snippet}],
                error_code|null, created_at }
Document      { id, filename, mime_type, size_bytes, status: "uploaded"|"processing"|"ready"|"failed",
                processing_error|null, created_at, updated_at }
```

## Auth (`/auth`)

| Method & path | Auth | Body | Response | Notes |
|---|---|---|---|---|
| `POST /auth/register` | – | `{full_name, email, password}` | 201 `TokenResponse` and sets the refresh cookie | Creates the account and signs the user in. Name is required. Email is trimmed and lowercased; the password must be 8–128 characters with at least one letter and one number. An email that is already registered returns 409 `EMAIL_ALREADY_REGISTERED` and is never overwritten. Rate limits: 10 per hour per IP, 5 per hour per email. |
| `POST /auth/login` | – | `{email, password}` | 200 `TokenResponse` and sets the refresh cookie | Unknown email and wrong password give the same 401 `INVALID_CREDENTIALS`. Rate limits: 10 per 5 minutes per IP and 10 per 15 minutes per account. |
| `POST /auth/refresh` | cookie | – | 200 `TokenResponse`; usually rotates the cookie | Rotation is compare-and-set, so of two concurrent refreshes exactly one rotates. The token rotated away in the last `REFRESH_REUSE_GRACE_SECONDS` (30 s) is still accepted, but returns an **access token only, with no new cookie**, so tabs of one browser never log each other out. Presenting an older token, or the previous one after the grace window, revokes the session (`REFRESH_TOKEN_REUSED`). The frontend also serializes refreshes across tabs with the Web Locks API. Rate limit: 60 per minute. |
| `POST /auth/logout` | cookie | – | 204 and clears the cookie | Revokes the session on the server. Existing access tokens stop working immediately. |
| `POST /auth/logout-all` | bearer | – | 204 | Revokes every session the user has. |
| `GET /auth/me` | bearer | – | `User` | The frontend calls this on startup. |
| `POST /auth/forgot-password` | – | `{email}` | 200 `{message}` | The response is the same whether or not the account exists. The token is single-use and expires after 30 minutes. Rate limits: 5 per hour per IP, 3 per hour per account. |
| `POST /auth/reset-password` | – | `{token, new_password}` | 200 `{message}` | Revokes every session and marks the email as verified. |

**Passwords.** Sign-up and login use email and password only; no email is sent for either. Passwords are hashed with Argon2id. They **cannot be changed from the profile**; the only way to set a new one is the emailed reset link (`/auth/forgot-password`), which needs an email provider (SMTP or Resend) to be configured.

**Frontend start-up flow.** The access token is held only in memory. On load the frontend calls `POST /auth/refresh`, which uses the cookie:

- If the refresh succeeds, the user is signed in.
- If it fails, the frontend shows the login screen.

Any API call that returns 401 triggers one refresh and one retry. If that refresh also fails, the frontend shows the login screen.

## Profile and settings

| Method & path | Body | Response | Notes |
|---|---|---|---|
| `GET /profile` | – | `User` | |
| `PATCH /profile` | `{full_name}` | `User` | Any other field returns 422. Email, role, and ID cannot be changed. |
| `DELETE /profile/data` | – | 204 | "Clear All Data": deletes all of the user's conversations and documents but keeps the account. |
| `GET /settings` | – | `{theme}` | `theme` is one of `dark`, `light`, or `system`. |
| `PATCH /settings` | `{theme?}` | `{theme}` | |

## Conversations

| Method & path | Body / query | Response |
|---|---|---|
| `POST /conversations` | `{title?}` (defaults to `"New Chat"`) | 201 `Conversation` |
| `GET /conversations` | `limit` (1–100, default 50), `offset`, `q` (title filter), `archived` (`exclude` by default, or `only`/`include`), `sort` (`updated_at`/`created_at`/`title`), `order` (`asc`/`desc`) | `Page<Conversation>`. Messages are not included. |
| `GET /conversations/{id}` | – | `Conversation` |
| `PATCH /conversations/{id}` | `{title?, archived?: bool}` | `Conversation` |
| `DELETE /conversations/{id}` | – | 204. Also deletes the conversation's messages, share links, and document links. |
| `GET /conversations/{id}/documents` | – | `Document[]` |
| `POST /conversations/{id}/documents` | `{document_ids: [...]}` (1–50) | `Document[]` (all documents now attached). Idempotent. Every document must belong to the caller, or the request returns 404. |
| `DELETE /conversations/{id}/documents/{document_id}` | – | 204 |

The first message in a conversation titled "New Chat" renames the conversation, using the first 60 characters of the message.

## Messages

| Method & path | Body | Response |
|---|---|---|
| `GET /conversations/{id}/messages` | `limit` (1–500, default 100), `offset`, `order` (`asc` default, `desc` = newest first; the frontend loads the newest 50 then pages back with "Load earlier") | `Page<Message>` |
| `POST /conversations/{id}/messages` | `{content, document_ids?}` | 201 `{user_message, assistant_message}`. If the AI call fails, returns 502/503 with `details: {user_message_id, assistant_message_id}`. |
| `POST /conversations/{id}/messages/stream` | `{content, document_ids?}` | `text/event-stream` (see below) |
| `POST /conversations/{id}/messages/{message_id}/retry` | – | `text/event-stream`. Only the latest assistant message can be retried, and only if it is `failed` or `cancelled`. |

Clients cannot set `role`, so they cannot write assistant messages; any field other than `content` and `document_ids` returns 422. `content` must be 1 to `MAX_MESSAGE_CHARS` characters (32,000 by default). `document_ids` (optional, for older clients) must belong to the caller; each is linked to the conversation and recorded as an attachment on the user message, but **attachments do not affect retrieval**. Every turn searches the caller's Knowledge Base: all of their documents with status `ready` (up to `LIBRARY_DISCOVERY_MAX_DOCUMENTS`, 500), in every conversation. Documents are added only through `POST /documents` (the Knowledge Context panel); chats have no upload of their own. Rate limit: 30 messages per minute per user.

### SSE events (`/stream` and `/retry`)

Validation and ownership errors arrive as normal JSON errors, before the stream opens. After that, events arrive in this order:

```text
event: message.created      data: {user_message, assistant_message}   # both persisted; assistant status "pending"
event: agent.started        data: {}
event: retrieval.started    data: {}
event: retrieval.completed  data: {retrieved_chunks?}
event: generation.started   data: {}
event: content.delta        data: {delta}                              # repeated
event: content.replace      data: {content}                            # optional, at most once: the verified answer replaces the streamed text
event: generation.completed data: {}
event: message.completed    data: {assistant_message}                  # final, persisted
-- or on failure --
event: error                data: {code, message, assistant_message?}  # assistant persisted as "failed"
```

Only the event types above are forwarded, and only the fields shown. Anything else Agent Core emits, including reasoning, prompts, and internal fields, is dropped. Error codes from Agent Core pass through only if they are in the safe list (`RAG_SERVICE_UNAVAILABLE`, `LLM_UNAVAILABLE`, `LLM_TIMEOUT`, `LLM_RATE_LIMITED`, `GENERATION_FAILED`); anything else becomes `GENERATION_FAILED`.

**Assistant message lifecycle:** `pending → streaming → completed | failed | cancelled`.

- While streaming, partial text is flushed to the database every `STREAM_FLUSH_SECONDS` (2 s).
- On failure, any partial text is kept, the status is `failed` with a safe `error_code`, and the message can be retried.
- If the client disconnects, the partial text is saved as `cancelled`.
- If the process crashes, the reconciler marks messages still `pending`/`streaming` after `STALE_GENERATION_SECONDS` as `failed` with `INTERRUPTED`; they can be retried.

No answer is ever fabricated.

## Documents

| Method & path | Body | Response | Notes |
|---|---|---|---|
| `POST /documents` | multipart with a `file` field (one file) | 201 `Document` (`status: "uploaded"`) | Allowed types: `pdf`, `txt`, `md`, `csv`, `docx` (images need OCR in RAG Core first). The server checks the file's bytes against its extension and ignores the MIME type the browser sends. The filename is cleaned up and the storage key is generated by the server. Maximum size is `MAX_UPLOAD_BYTES` (25 MB by default). Per-user quotas apply. Rate limit: 60 per 10 minutes. |
| `GET /documents` | `limit`, `offset`, `status` | `Page<Document>` | Deleted documents are not listed. |
| `GET /documents/{id}` | – | `Document` | Poll this endpoint for status. |
| `GET /documents/{id}/content` | – | The file, sent with `Content-Disposition: attachment` and `nosniff` | |
| `POST /documents/{id}/reprocess` | – | `Document` | Only allowed when the status is `failed`. |
| `DELETE /documents/{id}` | – | 204 | The document is hidden and detached from conversations immediately. Its vectors and stored file are deleted in the background. |

**Status lifecycle:**

```text
uploaded → processing → ready
                      ↘ failed
```

Failures are handled as follows:

- **Transient** (RAG Core, embeddings or vector store unavailable): retried up to `JOB_MAX_ATTEMPTS` (3) times with backoff.
- **Permanent** (`UNSUPPORTED_CONTENT`, `NO_EXTRACTABLE_TEXT`, `PARSE_FAILED`, ...): fails at once with a matching message.

In every case `processing_error` only holds a safe, user-facing message.

The reconciler recovers jobs left `running` by a crashed worker, respecting `max_attempts`. It also handles documents stuck in `uploaded`/`processing` with no live job, and deletes that gave up. Deletes get `DELETE_JOB_MAX_ATTEMPTS` (6) attempts and at most 5 re-drives, after which `document_delete_abandoned` is logged for manual attention. If a document is deleted while RAG Core is still indexing it, the new vectors are removed afterwards.

## Search

`GET /search/chats?q=…&limit=20&offset=0` returns `Page<Conversation & {snippet}>`.

- Matching is case-insensitive and covers both conversation titles and message content.
- Only the caller's own conversations are searched.
- `%` and `_` are matched literally, not as wildcards.
- `q` must be 1–200 characters.

## Sharing

| Method & path | Auth | Response |
|---|---|---|
| `POST /conversations/{id}/share` | bearer, body `{expires_in_days?}` (1–365) | 201 `{token, url, created_at, expires_at}`. The raw token is returned only once; the server stores only a SHA-256 hash. Creating a new link revokes the previous one. Rate limit: 30 per hour. |
| `GET /conversations/{id}/share` | bearer | `{active, created_at, expires_at}` |
| `DELETE /conversations/{id}/share` | bearer | 204. The link stops working immediately. |
| `GET /shares/{token}` | **public** | `{title, shared_at, messages: [{role, content, created_at}]}` |

The public view includes only the completed messages that existed when the link was created. It never includes IDs, email addresses, documents, metadata, or citations. Revoked, expired, and unknown tokens all return the same 404 `SHARE_NOT_FOUND`. Rate limit: 120 per minute per IP. The frontend serves shared conversations at `/share/:token`.

## Health (no `/api/v1` prefix)

| Path | Response |
|---|---|
| `GET /health` | `{"status": "ok"}` |
| `GET /health/live` | `{"status": "alive"}` |
| `GET /ready` (alias `/health/ready`) | 200 or 503: `{status, checks: {database, storage, agent_core, rag_core}, services_required}`. Database and storage are always required. Agent Core and RAG Core are required when `READINESS_REQUIRE_SERVICES` is on (the default outside development); their own `/ready` endpoints check the LLM layer, FreeLLMAPI, the embeddings and the vector store. The checks are booleans only, with no URLs. |
| `GET /metrics` | Prometheus text format. Requires `Authorization: Bearer <INTERNAL_SERVICE_TOKEN>` when a token is configured. |

---

## Internal service contracts

Every call between services sends two headers:

- `X-Request-ID`, so the request can be traced across services;
- `Authorization: Bearer <INTERNAL_SERVICE_TOKEN>`.

Agent Core, RAG Core and the LLM layer reject calls without the token. It is optional only in development, and required (32+ characters) in staging and production.

Internal errors use one envelope: `{"error": {"code", "message", "retryable", "request_id"}}`.

Each internal service exposes:

- `GET /health` (liveness);
- `GET /ready` (its dependencies; booleans only);
- `GET /metrics` (requires the token).

### Backend → Agent Core (`AGENT_CORE_URL`)

`POST /api/v1/agent/stream` and `POST /api/v1/agent/invoke` take the same body:

```json
{
  "user_id": "uuid", "conversation_id": "uuid", "message_id": "uuid (assistant message)",
  "message": "user text", "document_ids": [],
  "history": [{"role": "user|assistant", "content": "…"}],
  "context": {"pending_document_count": 0},
  "library_documents": [{"document_id": "uuid", "filename": "…"}]
}
```

`library_documents` is the user's Knowledge Base: every ready document they own (backend-verified; RAG Core re-checks ownership). `document_ids` is kept for compatibility and is empty from the backend. `pending_document_count` counts the user's documents still being indexed.

```text
```

**`/stream`** returns `text/event-stream`. This is the contract the backend forwards to the browser:

```text
agent.started        {}
retrieval.started    {}                          # only when Agent Core decides to retrieve
retrieval.completed  {retrieved_chunks}
generation.started   {}
content.delta        {delta}                     # repeated
content.replace      {content}                   # optional, at most once, after the deltas: the final verified answer
generation.completed {citations: [{source_id, document_id, filename, page, chunk_id, snippet}], metadata: {...}}  # source_id = the n in [n]
error                {code, message}             # code ∈ RAG_SERVICE_UNAVAILABLE | LLM_UNAVAILABLE | LLM_TIMEOUT | LLM_RATE_LIMITED | GENERATION_FAILED
```

**`/invoke`** returns `{content, citations, metadata}`. On failure it returns 503 with the same error codes.

The metadata the backend keeps is limited to:

- `model`
- `retrieval_strategy`
- `retrieved_chunks`
- `retrieval_latency_ms`
- `rerank_latency_ms`
- `generation_latency_ms`
- `total_latency_ms`
- `token_usage`
- `citation_count`
- `confidence`
- `verification_status`

Prompts and reasoning are never part of this contract.

All ids in the request must be UUIDs; anything else returns 422.

`verification_status` in the metadata is one of:

- `verified`: cited answer; every claim was checked against the cited passages;
- `revised`: unsupported claims were removed (the stored answer is the revised one);
- `computed`: exact figures computed over a whole stored CSV table;
- `reformatted`: an earlier answer of the same conversation reused (table, summary, rewording), no new facts;
- `insufficient_evidence`: the answer is exactly "I couldn't find sufficient evidence for this answer in your Knowledge Base.";
- `unverified`: cited answer, but verification could not run;
- `cited`: cited answer with verification disabled;
- `uncited`: sources were retrieved, but the answer cites none of them;
- `no_relevant_documents`: nothing relevant in the Knowledge Base; a general answer was given;
- `not_applicable`: no retrieval was done.

Additional safe metadata keys: `retrieval_scope` (`knowledge_base` | `named` | `none`), `answer_mode`, `evidence_grade`, `correction_attempts`, `answer_replaced`, `table_analysis`, `followup`. `structured_result` (exact table-analysis results behind an answer) is stored server-side only, never returned to the browser, and sent back to Agent Core as part of that conversation's history (`history[].structured`, with `history[].citations`).

**What Agent Core decides for each turn** (full description in [docs/agent-core-architecture.md](../docs/agent-core-architecture.md)):

- It retrieves when documents are attached and the message is not small talk. Without documents it answers from general knowledge and does not call RAG Core.
- When there is chat history, it rewrites the follow-up into a standalone search query with `AGENT_FAST_MODEL`. If the rewrite fails, it searches with the previous question plus the new message.
- It validates RAG Core's response. Every chunk must belong to a requested document, and the numbering must be 1…n.
- It builds a grounded prompt with numbered, delimited sources, which are treated as untrusted data, and streams generation.
- It turns `[n]` / `【n】` markers into citations, and only for sources that were actually provided.
- If RAG Core fails, it emits `error` and does **not** generate an answer.
- It never retries; the LLM layer's own retries are bounded. A follow-up rewrite that fails because the model is rate-limited or unreachable fails the turn immediately, and no second call is made.

### Backend worker → RAG Core (`RAG_CORE_URL`)

**`POST /api/v1/rag/ingest`** is multipart: a `file` part plus a `metadata` part containing `{document_id, user_id, filename, mime_type}`.

- On success it returns `{document_id, chunks, pages, embedding_model}`.
- Re-ingesting the same document replaces its chunks and vectors.
- If the `document_id` belongs to another user, it returns 409.

Errors the backend acts on:

| Status | Codes | Backend behaviour |
|---|---|---|
| 4xx (permanent) | `UNSUPPORTED_CONTENT` (415), `NO_EXTRACTABLE_TEXT` (422), `PARSE_FAILED` (422), `DOCUMENT_TOO_LARGE` (413), `EMBEDDING_REJECTED` (422) | Marks the document `failed` with a matching safe message. |
| 5xx (retryable) | `EMBEDDING_UNAVAILABLE` (local model could not load or run), `VECTOR_STORE_UNAVAILABLE`, `VECTOR_STORE_MISCONFIGURED` (wrong key, missing index, or index dimension ≠ 384) | Retries the job. |

**`DELETE /api/v1/rag/documents/{document_id}?user_id=`** returns 204. It deletes every vector id the document ever had (it tracks the highest chunk count), then the chunks. It returns 404 when there is nothing indexed for that user, which the backend treats as success.

### Agent Core → RAG Core

`POST /api/v1/rag/retrieve` takes `{user_id, query, document_ids, top_k?}` and returns:

```json
{
  "chunks": [{"source_id": 1, "chunk_id": "…", "document_id": "…", "filename": "…", "page": 3,
              "chunk_index": 0, "text": "…", "score": 0.91}],
  "context": "[1] file.pdf (page 3)\n…\n\n[2] …",
  "metadata": {"retrieval_strategy": "hybrid_rrf_rerank", "semantic_candidates": 20, "keyword_candidates": 12,
               "retrieved_chunks": 6, "reranker": "pinecone|none|skipped", "retrieval_latency_ms": 0, "rerank_latency_ms": 0}
}
```

Retrieval works in these steps:

1. **Scope.** Retrieval is limited to documents that this `user_id` actually has indexed. Each user has their own Pinecone namespace, and queries also filter by `document_id`.
2. **Semantic search.** The query is embedded locally with the same model as the documents (`BAAI/bge-small-en-v1.5`, 384 dimensions, bge's query instruction), and Pinecone returns the closest vectors.
3. **Keyword search.** PostgreSQL full-text search with OR semantics.
4. **Fusion.** The two result lists are combined with RRF (`k=60`).
5. **Reranking.** Pinecone's `bge-reranker-v2-m3` reorders the top candidates. If the reranker fails, retrieval falls back to the RRF order and reports `reranker: "skipped"`.
6. **Context.** The best passages are packed into numbered sources, up to `MAX_CONTEXT_CHARS`.

### Agent Core → LLM layer (`LLM_SERVICE_URL`)

The request body is `{messages: [{role, content}], model?, temperature?, max_tokens?, response_format?, tools?, tool_choice?}`.

- **`POST /v1/generate`** returns `{content, model, finish_reason, usage, tool_calls?, latency_ms}`.
- **`POST /v1/stream`** returns `text/event-stream` with these events:
  - `token {delta}`;
  - `done {model, finish_reason, usage, latency_ms, first_token_ms}`;
  - `error {code, message, retryable}`.

**`model`** is a routing hint passed through to FreeLLMAPI (for example `auto`, `auto:smart` or `auto:fast`). If omitted, the service uses `FREELLMAPI_MODEL`.

**Error codes:**

| Code | Meaning |
|---|---|
| `LLM_NOT_CONFIGURED` | No FreeLLMAPI key is configured. |
| `LLM_UNAVAILABLE` | FreeLLMAPI could not be reached (connection failure), after bounded retries. |
| `LLM_PROVIDER_UNAVAILABLE` | FreeLLMAPI answered 5xx, after bounded retries. |
| `LLM_TIMEOUT` | The provider timed out. Not retried. |
| `LLM_AUTH_FAILED` | The provider rejected the key (401/403). Not retried. |
| `LLM_RATE_LIMITED` | The provider returned 429; `Retry-After` was honoured (capped) and retries are bounded. |
| `LLM_BAD_REQUEST` | Unknown model or invalid parameters (400/404/413/422). Not retried. |
| `LLM_INVALID_RESPONSE` | The response was not OpenAI-compatible JSON or SSE. |
| `LLM_PROVIDER_ERROR` | The provider reported an error during generation, or an unexpected status. |
| `LLM_STREAM_INTERRUPTED` | The stream broke after tokens were already sent. Never retried, to avoid duplicate output. |

Provider messages, URLs and keys never appear in responses or logs. Agent Core passes `LLM_TIMEOUT` and `LLM_RATE_LIMITED` through, and maps every other code to `LLM_UNAVAILABLE`.

### LLM layer → FreeLLMAPI

The LLM layer uses FreeLLMAPI's OpenAI-compatible API: `POST {FREELLMAPI_BASE_URL}/chat/completions` (streaming uses `data:` chunks ending with `[DONE]`) and `GET /models`. It authenticates with `Authorization: Bearer <FREELLMAPI_API_KEY>`. The key lives only in `llm/.env`.

The LLM layer refuses to start in staging or production when either of these is true:

- the base URL is localhost or plain http;
- the key or the internal token is missing.
