"""Mock FreeLLMAPI (OpenAI-compatible) for automated tests and offline development.

    python scripts/mock_freellmapi.py --port 3901 --key test-key

NOT a model: it answers deterministically. When the prompt contains numbered sources (the
grounded prompt Agent Core builds from RAG Core context) it quotes the first source and cites
it as [1], so the whole citation path can be exercised end to end. Agent Core's self-correction
checks get keyword-overlap stand-ins (grader), a keyword query (correction), "supported"
(verifier) and a simple table plan (planner). Tests that use this gateway exercise the wiring,
not model judgement.
"""
import argparse
import asyncio
import json
import re
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

DEFAULT_KEY = "mock-freellmapi-key"

_STOP = set("""what which when where who whom whose does did have has with from that this these those there their about
into than then them they your yours mine ours please tell show give explain describe according document documents file files
uploaded knowledge base say says said information much many more most some any were will would could should""".split())
_GENERAL = re.compile(r"^\s*(what is|what are|explain|define|how do i|how to|write|who invented)\b", re.I)
_PERSONAL = re.compile(r"\b(my|our|uploaded|document|file|policy|according)\b", re.I)


def keywords(text: str):
    return [w for w in dict.fromkeys(re.findall(r"[a-z0-9]{4,}", text.lower())) if w not in _STOP]


def grade(prompt: str) -> dict:
    """Deterministic stand-in for the evidence grader: a passage is relevant when it contains at
    least half of the question's keywords."""
    question = re.search(r"QUESTION:\s*(.+)\s*$", prompt, re.S)
    q = question.group(1).strip() if question else ""
    words = keywords(q)
    relevant = []
    for n, text in re.findall(r'<source id="(\d+)"[^>]*>\n(.*?)\n</source>', prompt, re.S):
        hits = sum(1 for w in words if w in text.lower())
        if words and hits * 2 >= len(words):
            relevant.append(int(n))
    qtype = "general" if _GENERAL.match(q) and not _PERSONAL.search(q) else "document"
    return {"verdict": "sufficient" if relevant else "insufficient", "relevant": relevant, "conflict": False, "question_type": qtype}


def plan(prompt: str) -> dict:
    """Deterministic table planner: the first table; distribution of a column named in the
    question (stats when numeric), otherwise a row count."""
    table = re.search(r"TABLE document_id=(\S+) .*?\n  columns: (.+)", prompt)
    question = (re.search(r"QUESTION:\s*(.+)", prompt) or re.search(r"(.+)", prompt)).group(1).lower()
    if not table:
        return {"document_id": None, "queries": []}
    columns = re.findall(r"(?:^|; )([^;(]+?) \((numeric|text)", table.group(2))
    for name, ctype in columns:
        if name.lower() in question:
            op = "stats" if ctype == "numeric" else "distribution"
            return {"document_id": table.group(1), "queries": [{"operation": op, "column": name}]}
    return {"document_id": table.group(1), "queries": [{"operation": "count"}]}


def build_app(api_key: str = DEFAULT_KEY) -> FastAPI:
    app = FastAPI(title="mock-freellmapi")

    def authorized(request: Request) -> bool:
        return request.headers.get("authorization") == f"Bearer {api_key}"

    def answer(messages) -> str:
        last = next((m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"), "")
        system = next((m.get("content") or "" for m in messages if m.get("role") == "system"), "")
        if "standalone search query" in system:
            return re.sub(r"^.*Latest message:\s*", "", last, flags=re.S).strip()
        if "EVIDENCE GRADER" in system:
            return json.dumps(grade(last))
        if system.startswith("SEARCH QUERY CORRECTION"):
            question = re.search(r"Question:\s*(.+)", last)
            return " ".join(keywords(question.group(1) if question else last)) or "search"
        if "ANSWER VERIFIER" in system:
            return json.dumps({"supported": True, "unsupported": [], "revised": ""})
        if "TABLE QUERY PLANNER" in system:
            return json.dumps(plan(last))
        if "YOUR EARLIER ANSWERS" in system:  # follow-up rewrite: return the most recent earlier answer as is
            answers = re.findall(r'<answer n="\d+" \([^)]*\)>\n(.*?)\n</answer>', last, re.S)
            return answers[-1] if answers else "NEEDS_NEW_ANALYSIS"
        if "<evidence>" in last:
            source = re.search(r'<source id="1"[^>]*>\n(.+?)\n</source>', last, re.S)
            quote = (source.group(1).strip() if source else "the provided source")[:200]
            return f"## Answer\n\nAccording to your document: {quote} [1]\n\n```text\ngrounded=true\n```"
        question = last.strip()[:200]
        return f"**Mock model reply** to: {question}\n\n- point one\n- point two"

    @app.get("/v1/models")
    async def models(request: Request):
        if not authorized(request):
            return JSONResponse({"error": {"type": "invalid_api_key"}}, status_code=401)
        return {"object": "list", "data": [{"id": m} for m in ("auto", "auto:fast", "auto:smart")]}

    @app.post("/v1/chat/completions")
    async def completions(request: Request):
        if not authorized(request):
            return JSONResponse({"error": {"type": "invalid_api_key"}}, status_code=401)
        body = await request.json()
        text = answer(body.get("messages") or [])
        model = f"mock/{body.get('model', 'auto')}"
        usage = {"prompt_tokens": 10, "completion_tokens": len(text.split()), "total_tokens": 10 + len(text.split())}
        if not body.get("stream"):
            return {"id": "mock", "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}], "usage": usage}

        async def stream():
            words = text.split(" ")
            for i, word in enumerate(words):
                piece = word if i == 0 else " " + word
                yield "data: " + json.dumps({"model": model, "choices": [{"index": 0, "delta": {"content": piece}}]}) + "\n\n"
                await asyncio.sleep(0.01)
            yield "data: " + json.dumps({"model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": usage}) + "\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=3901)
    parser.add_argument("--key", default=DEFAULT_KEY)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(build_app(args.key), host=args.host, port=args.port, log_level="warning")
