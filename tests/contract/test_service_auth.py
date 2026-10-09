"""Service-to-service authentication guard: every endpoint of the internal services (Agent Core,
RAG Core, LLM service) except liveness/readiness must require INTERNAL_SERVICE_TOKEN, and a call
with a missing or wrong token is rejected before any work is done."""
import httpx
import pytest
from fastapi.routing import APIRoute

PUBLIC = {"/health", "/ready"}  # liveness/readiness: booleans only, no data


def _apps():
    from agent_core import main as agent_main
    from llm_service import main as llm_main
    from rag_core import main as rag_main

    return {"agent-core": agent_main.app, "rag-core": rag_main.app, "llm": llm_main.app}


def _guarded(route: APIRoute) -> bool:
    return any(getattr(d.call, "__qualname__", "").startswith("internal_auth.") for d in route.dependant.dependencies)


@pytest.mark.parametrize("service", ["agent-core", "rag-core", "llm"])
def test_every_internal_endpoint_requires_the_service_token(service):
    app = _apps()[service]
    routes = [r for r in app.routes if isinstance(r, APIRoute)]
    unguarded = sorted(f"{sorted(r.methods)} {r.path}" for r in routes if r.path not in PUBLIC and not _guarded(r))
    assert routes and unguarded == [], f"{service} endpoints without INTERNAL_SERVICE_TOKEN: {unguarded}"


@pytest.mark.parametrize("service,method,path", [
    ("agent-core", "POST", "/api/v1/agent/stream"), ("agent-core", "POST", "/api/v1/agent/invoke"), ("agent-core", "GET", "/metrics"),
    ("rag-core", "POST", "/api/v1/rag/retrieve"), ("rag-core", "POST", "/api/v1/rag/ingest"), ("rag-core", "GET", "/metrics"),
    ("llm", "POST", "/v1/generate"), ("llm", "POST", "/v1/stream"), ("llm", "GET", "/metrics"),
])
async def test_missing_or_wrong_token_is_rejected(stack, service, method, path):
    app = _apps()[service]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc") as c:
        for headers in ({}, {"Authorization": "Bearer wrong-token"}, {"Authorization": "Basic abc"}):
            r = await c.request(method, path, headers=headers, json={})
            assert r.status_code == 401, (service, path, headers, r.status_code)
            assert r.json()["error"]["code"] == "UNAUTHORIZED"
