# Build from the repository root:  docker build -f llm/Dockerfile .
#
# Production: set ENVIRONMENT=production, FREELLMAPI_BASE_URL=https://<hosted FreeLLMAPI>/v1,
# FREELLMAPI_API_KEY and INTERNAL_SERVICE_TOKEN as runtime variables (never build args).
# The service refuses to start with a localhost or plain-http FreeLLMAPI URL.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY shared shared
COPY llm/requirements.txt llm/requirements.txt
RUN pip install --no-cache-dir ./shared -r llm/requirements.txt

COPY llm/llm_service llm/llm_service
WORKDIR /srv/llm
RUN useradd --create-home appuser
USER appuser
EXPOSE 8003
# Liveness only (no FreeLLMAPI call): /ready is the dependency check used by the platform.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8003'), timeout=4)"
# `exec` makes the server PID 1 so SIGTERM reaches it; in-flight streams get 30 s to finish.
# PORT is set by Railway (and most PaaS); 8003 locally.
# HOST (default 0.0.0.0; :: = dual-stack IPv6 + IPv4, see advancerag_common/serve.py) and PORT come from the platform.
CMD ["sh", "-c", "exec python -m advancerag_common.serve llm_service.main:app 8003"]
