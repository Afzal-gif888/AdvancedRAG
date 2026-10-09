"""Centralised error responses: {"error": {"code", "message", "request_id", "details?"}}.

Never leaks stack traces, SQL, internal URLs or submitted values (e.g. passwords).
"""
import logging
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from advancerag_common.observability import route_template

from app.core.context import get_request_id
from app.core.exceptions import AppError

logger = logging.getLogger("advancerag.errors")

_HTTP_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
}


def _body(code: str, message: str, request_id: Optional[str], details: Any = None) -> dict:
    error = {"code": code, "message": message, "request_id": request_id or get_request_id()}
    if details is not None:
        error["details"] = details
    return {"error": error}


def error_response(exc: AppError, request_id: Optional[str] = None) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_body(exc.code, exc.message, request_id, exc.details),
        headers=exc.headers,
    )


def _safe_validation_details(exc: RequestValidationError) -> list:
    # Drop "input"/"ctx": they echo the submitted values, which may include passwords.
    details = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", ()) if p not in ("body", "query", "path")]
        details.append({"field": ".".join(loc) or None, "message": err.get("msg", "Invalid value"), "type": err.get("type")})
    return details


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError):
        return error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=_body("VALIDATION_ERROR", "The request is invalid.", None, _safe_validation_details(exc)),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        code = _HTTP_CODES.get(exc.status_code, "HTTP_ERROR")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        return JSONResponse(status_code=exc.status_code, content=_body(code, message, None), headers=getattr(exc, "headers", None))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # Route template, never the raw path (it can carry share tokens).
        logger.exception("unhandled_error", extra={"route": route_template(request.scope), "error_type": type(exc).__name__})
        return JSONResponse(
            status_code=500,
            content=_body("INTERNAL_SERVER_ERROR", "An unexpected error occurred.", None),
        )
