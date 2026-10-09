"""Request context, redacting JSON logs, timing and a tiny Prometheus-compatible metrics registry.

The metrics/timing API (`timed`, `METRICS`) is intentionally narrow so an OpenTelemetry
exporter can be dropped in behind it later without touching call sites.
"""
import json
import logging
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")

# Values that must never reach logs, whatever the call site passes.
_REDACTIONS = [
    (re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [REDACTED]"),
    (re.compile(r"eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]+"), "[REDACTED_JWT]"),
    (re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization)(\"?\s*[:=]\s*\"?)[^\s\",&]+"), r"\1\2[REDACTED]"),
    (re.compile(r"(?i)([?&]key=)[^&\s]+"), r"\1[REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}"), "[REDACTED_KEY]"),
]


def redact(text: str) -> str:
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def get_request_id() -> Optional[str]:
    return request_id_var.get()


_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str):
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": get_request_id(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key not in payload and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, default=str))


def configure_logging(service: str, level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    logging.getLogger("uvicorn.access").disabled = True
    # httpx logs full URLs at INFO (may include query-string keys).
    logging.getLogger("httpx").setLevel(logging.WARNING)


def route_template(scope) -> str:
    """The request path with every path-parameter value replaced by `{name}`
    (e.g. /api/v1/shares/{share_token}). Never returns raw IDs or tokens, independent of
    how routers are nested. Unmatched requests are reported as "<unmatched>"."""
    params = scope.get("path_params") or {}
    if scope.get("route") is None and not params:
        return "<unmatched>"
    segments = scope.get("path", "").split("/")
    for name, value in params.items():
        value = str(value)
        for i in range(len(segments) - 1, -1, -1):
            if segments[i] == value:
                segments[i] = "{" + name + "}"
                break
    return "/".join(segments)


# ---------------------------------------------------------------- metrics

_BUCKETS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000)


class Metrics:
    """Thread-safe counters and latency histograms (ms), rendered in Prometheus text format."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[Tuple[str, Tuple], float] = {}
        self._hist: Dict[Tuple[str, Tuple], list] = {}

    @staticmethod
    def _key(name: str, labels: Dict[str, str]) -> Tuple[str, Tuple]:
        return name, tuple(sorted(labels.items()))

    def inc(self, name: str, value: float = 1, **labels: str) -> None:
        with self._lock:
            k = self._key(name, labels)
            self._counters[k] = self._counters.get(k, 0) + value

    def observe_ms(self, name: str, ms: float, **labels: str) -> None:
        with self._lock:
            k = self._key(name, labels)
            h = self._hist.setdefault(k, [0] * (len(_BUCKETS) + 1) + [0.0, 0])
            for i, b in enumerate(_BUCKETS):
                if ms <= b:
                    h[i] += 1
            h[len(_BUCKETS)] += 1  # +Inf
            h[-2] += ms
            h[-1] += 1

    def render(self) -> str:
        def fmt(labels: Tuple, extra: str = "") -> str:
            parts = [f'{k}="{v}"' for k, v in labels] + ([extra] if extra else [])
            return "{" + ",".join(parts) + "}" if parts else ""

        lines = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"{name}_total{fmt(labels)} {value}")
            for (name, labels), h in sorted(self._hist.items()):
                for i, b in enumerate(_BUCKETS):
                    le = 'le="%s"' % b
                    lines.append(f"{name}_ms_bucket{fmt(labels, le)} {h[i]}")
                le_inf = 'le="+Inf"'
                lines.append(f"{name}_ms_bucket{fmt(labels, le_inf)} {h[len(_BUCKETS)]}")
                lines.append(f"{name}_ms_sum{fmt(labels)} {round(h[-2], 3)}")
                lines.append(f"{name}_ms_count{fmt(labels)} {h[-1]}")
        return "\n".join(lines) + "\n"


METRICS = Metrics()
_log = logging.getLogger("advancerag.timing")


@asynccontextmanager
async def timed(operation: str, **labels: str):
    """Time a dependency call: logs `{operation, outcome, duration_ms}` and records a histogram."""
    started = time.perf_counter()
    outcome = "ok"
    try:
        yield
    except Exception as exc:
        outcome = getattr(exc, "code", None) or type(exc).__name__
        raise
    finally:
        ms = round((time.perf_counter() - started) * 1000, 2)
        METRICS.observe_ms(operation, ms, outcome=outcome, **labels)
        _log.info("dependency_call", extra={"operation": operation, "outcome": outcome, "duration_ms": ms, **labels})


# ---------------------------------------------------------------- ASGI middleware

class RequestContextMiddleware:
    """Request ID (accept a well-formed incoming one, else generate), access log by route
    template (never the raw path, which may embed tokens) and request metrics."""

    def __init__(self, app):
        self.app = app
        self.log = logging.getLogger("advancerag.access")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        incoming = dict(scope["headers"]).get(b"x-request-id", b"").decode("latin-1")
        request_id = incoming if _VALID_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message["headers"] = list(message.get("headers", [])) + [(b"x-request-id", request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            path = route_template(scope)
            ms = round((time.perf_counter() - started) * 1000, 2)
            METRICS.observe_ms("http_request", ms, route=path, method=scope["method"], status=str(status["code"]))
            self.log.info("request", extra={"method": scope["method"], "route": path, "status": status["code"], "duration_ms": ms})
            request_id_var.reset(token)
