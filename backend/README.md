# AdvancedRAG backend

The FastAPI application layer handles:

- authentication and sessions;
- authorization and user isolation;
- conversations, messages and SSE transport;
- document uploads and the background job lifecycle;
- sharing, settings, rate limiting and readiness.

It holds **no** model-provider keys and contains no retrieval or orchestration logic. Chat goes to Agent Core; document ingestion and deletion go to RAG Core.

Setup, the architecture and the commands for running the whole system are in the [root README](../README.md). All endpoints and internal contracts are in [API_CONTRACT.md](API_CONTRACT.md).

| Path | Contents |
|---|---|
| `app/api/routes/` | Thin HTTP handlers. `deps.py` resolves the user and session for every protected route. |
| `app/services/` | Business logic. Every query on user-owned data is scoped to the owner here. |
| `app/db/models/`, `alembic/` | SQLAlchemy models. Migrations are the only way the schema is created; there is no `create_all` at startup. |
| `app/integrations/` | Agent Core and RAG Core clients, storage (local and S3), email. |
| `app/workers/` | Database-backed job worker, stale-job recovery and the reconciler. |
| `app/middleware/` | Request IDs and access logs (route templates only), body limits, errors, rate limiting. |

```bash
../.venv/Scripts/python -m alembic upgrade head
../.venv/Scripts/python -m uvicorn app.main:app --port 8000
../.venv/Scripts/python -m pytest                  # SQLite by default; set TEST_DATABASE_URL for PostgreSQL
```
