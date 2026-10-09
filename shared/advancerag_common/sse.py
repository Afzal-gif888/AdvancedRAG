"""Server-Sent Events helpers used between internal services."""
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict

import httpx


@dataclass
class Event:
    event: str
    data: Dict[str, Any]


def format_event(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def _decode(event: str, lines) -> Event:
    try:
        data = json.loads("\n".join(lines))
    except json.JSONDecodeError:
        data = {}
    return Event(event, data if isinstance(data, dict) else {})


async def parse_events(response: httpx.Response) -> AsyncIterator[Event]:
    event, lines = "message", []
    async for line in response.aiter_lines():
        if line == "":
            if lines:
                yield _decode(event, lines)
            event, lines = "message", []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            lines.append(line[5:].lstrip())
    if lines:
        yield _decode(event, lines)
