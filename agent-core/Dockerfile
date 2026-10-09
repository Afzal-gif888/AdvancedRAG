# Build from the repository root:  docker build -f agent-core/Dockerfile .
# Internal service: keep it on a private network; it requires INTERNAL_SERVICE_TOKEN outside development.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY shared shared
COPY agent-core/requirements.txt agent-core/requirements.txt
RUN pip install --no-cache-dir ./shared -r agent-core/requirements.txt

COPY agent-core/agent_core agent-core/agent_core
WORKDIR /srv/agent-core
RUN useradd --create-home appuser
USER appuser
EXPOSE 8001
# Liveness only; /ready (RAG Core + LLM service) is the dependency check used by the platform.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8001'), timeout=4)"
# `exec` makes the server PID 1 so SIGTERM reaches it; in-flight streams get 30 s to finish.
# HOST (default 0.0.0.0; :: = dual-stack IPv6 + IPv4, see advancerag_common/serve.py) and PORT come from the platform.
CMD ["sh", "-c", "exec python -m advancerag_common.serve agent_core.main:app 8001"]
