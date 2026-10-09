# Build from the repository root:  docker build -f rag-core/Dockerfile .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY shared shared
COPY rag-core/requirements.txt rag-core/requirements.txt
RUN pip install --no-cache-dir ./shared -r rag-core/requirements.txt

COPY rag-core/rag_core rag-core/rag_core
COPY rag-core/alembic.ini rag-core/alembic.ini
COPY rag-core/alembic rag-core/alembic
WORKDIR /srv/rag-core
RUN useradd --create-home appuser && mkdir -p var/models && chown -R appuser var
USER appuser
# Bake the embedding model into the image so containers start without downloading it.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBEDDING_MODEL}', cache_dir='var/models')"
EXPOSE 8002
# Liveness only; /ready (database, Pinecone index + dimension, embedding model) is the readiness probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3   CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8002'), timeout=4)"
# Migrate, then `exec` the server (PID 1, receives SIGTERM). PORT is set by Railway (8002 locally).
# HOST (default 0.0.0.0; :: = dual-stack IPv6 + IPv4, see advancerag_common/serve.py) and PORT come from the platform.
CMD ["sh", "-c", "if [ \"${RUN_MIGRATIONS:-1}\" = \"1\" ]; then alembic upgrade head; fi && exec python -m advancerag_common.serve rag_core.main:app 8002"]
