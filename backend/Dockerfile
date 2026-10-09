# Build from the repository root:  docker build -f backend/Dockerfile .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY shared shared
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir ./shared -r backend/requirements.txt

COPY backend/app backend/app
COPY backend/alembic.ini backend/alembic.ini
COPY backend/alembic backend/alembic
WORKDIR /srv/backend
# var/storage exists (owned by appuser) so a mounted named volume inherits writable ownership.
RUN useradd --create-home appuser && mkdir -p var/storage var/outbox && chown -R appuser var
USER appuser
EXPOSE 8000
# Liveness only; /ready (database, storage, Agent Core, RAG Core) is the platform's readiness probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3   CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)"
# Migrate, then `exec` the server so it is PID 1 and receives SIGTERM (in-flight streams get 30 s).
# Migrations are idempotent; with several replicas set RUN_MIGRATIONS=0 and run them once as a
# release step instead. PORT is set by Railway (8000 locally).
# HOST (default 0.0.0.0; :: = dual-stack IPv6 + IPv4, see advancerag_common/serve.py) and PORT come from the platform.
CMD ["sh", "-c", "if [ \"${RUN_MIGRATIONS:-1}\" = \"1\" ]; then alembic upgrade head; fi && exec python -m advancerag_common.serve app.main:app 8000"]
