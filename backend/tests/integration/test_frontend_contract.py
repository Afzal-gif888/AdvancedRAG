"""The browser client and the deployed proxy must agree with the backend's routes and HTTP methods.

A request that reaches the right host with the wrong path or method fails as 404/405, which in the browser
looks like "login does not work". These tests compare the endpoints that `frontend/src/api.ts` calls with the
routes FastAPI actually serves, and check that the Vercel rewrite forwards them unchanged.
"""
import json
import re
from pathlib import Path

import pytest
from app.main import app

ROOT = Path(__file__).resolve().parents[3]
API_TS = ROOT / "frontend" / "src" / "api.ts"
VERCEL = ROOT / "frontend" / "vercel.json"

pytestmark = pytest.mark.skipif(not API_TS.exists(), reason="frontend sources not present in this checkout")


def _norm(path: str) -> str:
    path = path.split("?")[0]
    path = re.sub(r"\$\{[^}]*\}", "{}", path)
    path = re.sub(r"\{[^}/]*\}", "{}", path)
    return path.rstrip("/") or "/"


def frontend_calls():
    src = API_TS.read_text(encoding="utf8")
    calls = set()
    for m in re.finditer(r"this\.(?:json|raw|send)(?:<[^>(]*>)?\(\s*([`'])(.*?)\1(?=(.{0,260}))", src, re.S):
        path, tail = m.group(2), m.group(3)
        tail = tail.split("this.", 1)[0]
        method = re.search(r"method:\s*'(\w+)'", tail)
        calls.add((method.group(1) if method else "GET", _norm(path)))
    for m in re.finditer(r"fetch\(`\$\{API_BASE_URL\}([^`]*)`,\s*\{\s*method:\s*'(\w+)'", src):
        calls.add((m.group(2), _norm(m.group(1))))
    for m in re.finditer(r"xhr\.open\('(\w+)',\s*`\$\{API_BASE_URL\}([^`]*)`", src):
        calls.add((m.group(1), _norm(m.group(2))))
    return calls


def backend_routes():
    """(method, path-without-/api/v1) for every operation in the OpenAPI schema (FastAPI nests included routers)."""
    prefix = "/api/v1"
    routes = set()
    for path, ops in app.openapi()["paths"].items():
        if path.startswith(prefix):
            for method in ops:
                if method.upper() in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                    routes.add((method.upper(), _norm(path[len(prefix):])))
    return routes


def test_extractor_finds_the_known_calls():
    calls = frontend_calls()
    for expected in [("POST", "/auth/login"), ("POST", "/auth/refresh"), ("POST", "/auth/register"), ("POST", "/auth/logout"),
                     ("POST", "/documents"), ("POST", "/conversations/{}/messages/stream"), ("DELETE", "/documents/{}"),
                     ("PATCH", "/conversations/{}"), ("GET", "/auth/me")]:
        assert expected in calls, expected
    assert len(calls) >= 25


def test_every_frontend_call_is_served_by_the_backend_with_the_same_method():
    missing = sorted(c for c in frontend_calls() if c not in backend_routes())
    assert not missing, f"the frontend calls endpoints the backend does not serve with that method: {missing}"


def _rewrites():
    return json.loads(VERCEL.read_text(encoding="utf8"))["rewrites"]


def test_vercel_forwards_api_paths_unchanged_to_the_backend():
    rule = next(r for r in _rewrites() if r["source"].startswith("/api/"))
    assert rule["destination"].startswith("https://") and rule["destination"].endswith("/api/$1")
    for path in ["/api/v1/auth/login", "/api/v1/auth/refresh", "/api/v1/conversations/abc/messages/stream"]:
        m = re.fullmatch(r"/api/(.*)", path)
        assert m, path
        forwarded = rule["destination"].split("onrender.com", 1)[1].replace("$1", m.group(1))
        assert forwarded == path, "the backend must receive exactly the path the browser sent"


def test_spa_fallback_never_swallows_api_requests():
    fallback = next(r for r in _rewrites() if r["destination"] == "/index.html")
    # path-to-regexp "(?!api/|assets/|favicon\\.svg)" behaves like this Python regex
    pattern = re.compile(r"/((?!api/|assets/|favicon\.svg).*)")
    assert fallback["source"] == r"/((?!api/|assets/|favicon\.svg).*)"
    for path in ["/api/v1/auth/login", "/api/v1/auth/refresh"]:
        assert not pattern.fullmatch(path), path
    assert pattern.fullmatch("/share/some-token") and pattern.fullmatch("/")


def test_wake_relay_only_touches_health_endpoints():
    wake = [r for r in _rewrites() if r["source"].startswith("/wake/")]
    assert wake and all(r["destination"].endswith("/health") for r in wake)
