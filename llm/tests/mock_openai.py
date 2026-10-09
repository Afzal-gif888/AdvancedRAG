"""A real HTTP server speaking the OpenAI-compatible API, with switchable failure scenarios.
Used instead of FreeLLMAPI in automated tests (no real key needed)."""
import asyncio
import json
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

VALID_KEY = "test-freellmapi-key"


class Scenario:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = "ok"            # ok | slow | malformed | error_status | stream_break | no_models | models_invalid
        self.status = 500
        self.fail_times = 0         # fail this many requests with `status`, then succeed
        self.retry_after = None
        self.delay = 0.0
        self.calls = 0
        self.last_body = None
        self.last_auth = None
        self.last_request_id = None
        self.models = ["auto", "auto:fast", "auto:smart"]
        self.reply = "Hello from the mock model."


scenario = Scenario()
app = FastAPI()


def _auth_ok(request: Request) -> bool:
    scenario.last_auth = request.headers.get("authorization")
    scenario.last_request_id = request.headers.get("x-request-id")
    return scenario.last_auth == f"Bearer {VALID_KEY}"


@app.get("/v1/models")
async def models(request: Request):
    if not _auth_ok(request):
        return JSONResponse({"error": {"type": "invalid_api_key", "message": "bad key"}}, status_code=401)
    if scenario.mode == "no_models":
        return PlainTextResponse("not found", status_code=404)
    if scenario.mode == "models_invalid":
        return PlainTextResponse("<html>gateway login page</html>", status_code=200)
    if scenario.mode == "models_error":
        return JSONResponse({"error": {"type": "upstream"}}, status_code=scenario.status)
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in scenario.models]}


@app.post("/v1/chat/completions")
async def completions(request: Request):
    scenario.calls += 1
    body = await request.json()
    scenario.last_body = body
    if not _auth_ok(request):
        return JSONResponse({"error": {"type": "invalid_api_key", "message": "Incorrect API key sk-should-not-leak"}}, status_code=401)
    if scenario.fail_times > 0:
        scenario.fail_times -= 1
        headers = {"Retry-After": str(scenario.retry_after)} if scenario.retry_after is not None else None
        return JSONResponse({"error": {"type": "upstream", "message": "provider exploded at http://10.0.0.5"}}, status_code=scenario.status, headers=headers)
    if scenario.mode == "error_status":
        return JSONResponse({"error": {"type": "bad_model"}}, status_code=scenario.status)
    if scenario.delay:
        await asyncio.sleep(scenario.delay)
    model = "mock-provider/" + body.get("model", "auto")

    if not body.get("stream"):
        if scenario.mode == "malformed":
            return {"unexpected": True}
        return {
            "id": "c1",
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": scenario.reply}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
        }

    async def gen():
        if scenario.mode == "malformed":
            yield "data: {not json\n\n"
            return
        words = scenario.reply.split(" ")
        for i, word in enumerate(words):
            piece = word if i == 0 else " " + word
            yield "data: " + json.dumps({"model": model, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}) + "\n\n"
            await asyncio.sleep(0.01)
            if scenario.mode == "stream_break" and i == 1:
                raise RuntimeError("connection dropped")
        yield "data: " + json.dumps({"model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerThread:
    def __init__(self, asgi_app, port: int):
        config = uvicorn.Config(asgi_app, host="127.0.0.1", port=port, log_level="warning", lifespan="off")
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started:
            if time.time() > deadline:
                raise RuntimeError("mock server did not start")
            time.sleep(0.02)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)
