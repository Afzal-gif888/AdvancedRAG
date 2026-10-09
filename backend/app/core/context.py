"""Per-request context shared by logging, DB timing and outbound service calls."""
from contextvars import ContextVar
from typing import Optional

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
db_time_var: ContextVar[Optional[list]] = ContextVar("db_time", default=None)


def get_request_id() -> Optional[str]:
    return request_id_var.get()
