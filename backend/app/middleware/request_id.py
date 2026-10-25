"""Request context middleware: correlation ID, access log, latency and DB timing.

Implemented as pure ASGI (not BaseHTTPMiddleware) so it does not buffer streaming responses.
"""
import logging
import re
import time
import uuid

from advancerag_common.observability import METRICS, new_trace_id, parse_traceparent, route_template, trace_id_var

from app.core.context import db_time_var, request_id_var

logger = logging.getLogger("advancerag.access")

REQUEST_ID_HEADER = "x-request-id"
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")


class RequestContextMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        incoming = dict(scope["headers"]).get(REQUEST_ID_HEADER.encode(), b"").decode("latin-1")
        # Only accept well-formed client IDs to avoid log injection.
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        rid_token = request_id_var.set(request_id)
        trace_token = trace_id_var.set(parse_traceparent(dict(scope["headers"]).get(b"traceparent", b"").decode("latin-1")) or new_trace_id())
        db_token = db_time_var.set([0.0, 0])
        scope.setdefault("state", {})["request_id"] = request_id

        started = time.perf_counter()
        status_holder = {"status": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                headers = list(message.get("headers", []))
                headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            db_time, db_queries = db_time_var.get() or (0.0, 0)
            # The matched route *template* (e.g. /api/v1/shares/{share_token}), never the raw
            # path or query string, which can carry share tokens or user search terms.
            route = route_template(scope)
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            METRICS.observe_ms("http_request", duration_ms, route=route, method=scope["method"], status=str(status_holder["status"]))
            logger.info(
                "request",
                extra={
                    "method": scope["method"],
                    "route": route,
                    "status": status_holder["status"],
                    "duration_ms": duration_ms,
                    "db_ms": round(db_time * 1000, 2),
                    "db_queries": db_queries,
                },
            )
            db_time_var.reset(db_token)
            request_id_var.reset(rid_token)
            trace_id_var.reset(trace_token)
