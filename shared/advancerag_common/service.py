"""Uniform error envelope, internal service authentication and outbound header helpers."""
import hmac
import ipaddress
import logging
from typing import Dict, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from advancerag_common.observability import METRICS, get_request_id

log = logging.getLogger("advancerag.errors")

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}


class ServiceError(Exception):
    """Error safe to return to the calling service. `retryable` tells callers whether to retry."""

    def __init__(self, code: str, message: str, status_code: int = 502, retryable: bool = False):
        super().__init__(message)
        self.code, self.message, self.status_code, self.retryable = code, message, status_code, retryable


def error_body(code: str, message: str, retryable: bool = False) -> dict:
    return {"error": {"code": code, "message": message, "retryable": retryable, "request_id": get_request_id()}}


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status_code, content=error_body(exc.code, exc.message, exc.retryable))

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        details = [{"field": ".".join(str(p) for p in e.get("loc", ())[1:]), "message": e.get("msg")} for e in exc.errors()]
        body = error_body("VALIDATION_ERROR", "The request is invalid.")
        body["error"]["details"] = details
        return JSONResponse(status_code=422, content=body)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {401: "UNAUTHORIZED", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(status_code=exc.status_code, content=error_body(code, str(exc.detail)), headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled_error", extra={"error_type": type(exc).__name__})
        return JSONResponse(status_code=500, content=error_body("INTERNAL_ERROR", "Internal error."))


def internal_auth(expected_token: Optional[str]):
    """FastAPI dependency enforcing `Authorization: Bearer <INTERNAL_SERVICE_TOKEN>`.

    When no token is configured (development only — production config refuses that) the
    check is skipped so services can run without setup.
    """

    async def _check(request: Request) -> None:
        if not expected_token:
            return
        header = request.headers.get("authorization", "")
        presented = header[7:] if header.lower().startswith("bearer ") else ""
        if not presented or not hmac.compare_digest(presented.encode(), expected_token.encode()):
            METRICS.inc("internal_auth_rejected")
            raise StarletteHTTPException(401, "Invalid service credentials.")

    return _check


def outbound_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {}
    rid = get_request_id()
    if rid:
        headers["X-Request-ID"] = rid
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def metrics_endpoint(app: FastAPI, token: Optional[str]) -> None:
    from fastapi import Depends

    @app.get("/metrics", include_in_schema=False, dependencies=[Depends(internal_auth(token))])
    async def metrics():
        return PlainTextResponse(METRICS.render(), media_type="text/plain; version=0.0.4")


def is_local_url(url: str) -> bool:
    """True for URLs that only make sense on the developer's machine (localhost names,
    loopback/unspecified/link-local IPs such as 127.0.0.2 or 0.0.0.0)."""
    host = (urlparse(url).hostname or "").lower()
    if host in LOCAL_HOSTS or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_unspecified or ip.is_link_local


def is_https(url: str) -> bool:
    return urlparse(url).scheme == "https"
