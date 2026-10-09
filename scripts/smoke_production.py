"""Post-deploy smoke test of a deployed AdvancedRAG, through its public HTTPS URL only.

    python scripts/smoke_production.py --url https://<your-app>.vercel.app [--backend-url https://<backend>.up.railway.app]

Uses only the public API, exactly like the browser: register a synthetic account, upload a small
document and a CSV, wait for indexing, ask a grounded question (streamed), an absent fact, an
exact CSV statistic and a follow-up, share and revoke, then delete everything it created
(documents, chats, all data of the synthetic account). It never needs secrets and prints none.

Verifies, end to end: Vercel → backend → agent-core → rag-core (Neon, Pinecone, embeddings) and
→ llm → hosted FreeLLMAPI, S3/R2 storage, streaming through the proxy, and the refresh cookie.
The synthetic account itself remains (there is no account-deletion endpoint); its data is cleared.
Exit code 0 only if every check passes.
"""
import argparse
import csv
import io
import json
import re
import secrets
import sys
import time
import uuid

import httpx

ABSTAIN = "I couldn't find sufficient evidence for this answer in your Knowledge Base."
FACT = ("Smoke-test handbook. The Vantor K7 kettle carries a 19-month warranty from the date of purchase. "
        "Limescale damage is not covered.")
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" — {str(detail)[:200]}" if detail else ""), flush=True)
    return ok


def sse(client, path, body):
    events, name, started, first = [], None, time.perf_counter(), None
    with client.stream("POST", path, json=body, timeout=300) as r:
        if r.status_code != 200:
            return r.status_code, [], None, None
        for line in r.iter_lines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                if name == "content.delta" and first is None:
                    first = time.perf_counter() - started
                events.append((name, json.loads(line[5:].strip())))
    final = next((d.get("assistant_message") for n, d in events if n == "message.completed"), None)
    return 200, [n for n, _ in events], final, first


def wait_ready(c, doc, limit_s=600):
    deadline = time.time() + limit_s
    while time.time() < deadline:
        st = c.get(f"/documents/{doc}").json().get("status")
        if st in ("ready", "failed"):
            return st
        time.sleep(3)
    return "timeout"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="public web URL (Vercel), e.g. https://app.vercel.app")
    ap.add_argument("--backend-url", help="backend public URL, to check /ready directly (optional)")
    ap.add_argument("--allow-http", action="store_true", help="local validation of this script only (skips HTTPS/header checks)")
    ap.add_argument("--api-only", action="store_true", help="--url is the API host itself (no web app there): skip the web-app checks")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    api = base + "/api/v1"
    if not base.startswith("https://") and not args.allow_http:
        check("URL is HTTPS", False, base)
        return 1

    r = httpx.get(base + "/", timeout=30, follow_redirects=True)
    if not args.api_only:
        check("web app served" + ("" if args.allow_http else " over HTTPS"), r.status_code == 200 and "<div id=\"root\"" in r.text, r.status_code)
    if not args.allow_http and not args.api_only:
        check("security headers (CSP, HSTS)", "content-security-policy" in r.headers and "strict-transport-security" in r.headers)
    if args.backend_url:
        r = httpx.get(args.backend_url.rstrip("/") + "/ready", timeout=60)
        check("backend /ready (database, storage, agent-core, rag-core)", r.status_code == 200, r.text[:200])

    c = httpx.Client(base_url=api, timeout=120)
    email = f"smoke-{uuid.uuid4().hex[:10]}@example.com"
    password = "Smoke-" + secrets.token_urlsafe(14) + "9"
    created = []
    try:
        r = c.post("/auth/register", json={"email": email, "password": password, "full_name": "Smoke Test"})
        if not check("register", r.status_code == 201, r.status_code):
            return 1
        c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
        cookie = r.headers.get("set-cookie", "").lower()
        check("refresh cookie is HttpOnly" + ("" if args.allow_http else " + Secure"), "httponly" in cookie and ("secure" in cookie or args.allow_http))
        r = c.post("/auth/refresh")
        check("token refresh through the public URL", r.status_code == 200, r.status_code)
        c.headers["Authorization"] = f"Bearer {r.json().get('access_token', '')}"

        r = c.post("/documents", files={"file": ("smoke-handbook.txt", FACT.encode(), "text/plain")})
        check("upload (storage)", r.status_code == 201, r.status_code)
        doc = r.json()["id"]
        created.append(doc)
        check("ingestion reaches ready (Neon, embeddings, Pinecone)", wait_ready(c, doc) == "ready")

        conv = c.post("/conversations", json={}).json()["id"]
        code, names, final, first = sse(c, f"/conversations/{conv}/messages/stream",
                                        {"content": "How long is the Vantor K7 warranty?"})
        cites = (final or {}).get("citations") or []
        check("streamed grounded answer with citation (agent-core, llm, hosted FreeLLMAPI)",
              code == 200 and final and "19" in final["content"] and any(x.get("document_id") == doc for x in cites),
              f"first token after {first and round(first, 1)} s; events={len(names)}")
        code, names, final, _ = sse(c, f"/conversations/{conv}/messages/stream",
                                    {"content": "What is the disaster recovery RTO in my Knowledge Base?"})
        check("absent fact gets the exact abstain answer", final and final["content"].strip() == ABSTAIN)

        buf = io.StringIO()
        rows = [[f"R{i:04d}", "positive" if i % 10 < 5 else "negative" if i % 10 < 8 else "neutral"] for i in range(1000)]
        csv.writer(buf, lineterminator="\n").writerows([["review_id", "sentiment"], *rows])
        r = c.post("/documents", files={"file": ("smoke-reviews.csv", buf.getvalue().encode(), "text/csv")})
        csv_doc = r.json()["id"]
        created.append(csv_doc)
        check("CSV ingestion reaches ready", wait_ready(c, csv_doc) == "ready")
        conv2 = c.post("/conversations", json={}).json()["id"]
        code, names, final, _ = sse(c, f"/conversations/{conv2}/messages/stream",
                                    {"content": "What is the sentiment distribution in smoke-reviews.csv?"})
        text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", (final or {}).get("content", ""))
        check("exact CSV statistics over all rows", all(n in text for n in ("500", "300", "200")), text[:120])
        code, names, final, _ = sse(c, f"/conversations/{conv2}/messages/stream", {"content": "give above in table format"})
        check("follow-up reuses the previous result", final and "| Total | 1000 | 100.0% |" in final["content"])

        r = c.post(f"/conversations/{conv2}/share")
        token = r.json().get("url", "").rstrip("/").rsplit("/", 1)[-1]
        check("share link created", r.status_code == 201 and token)
        check("shared conversation readable without login", httpx.get(f"{api}/shares/{token}", timeout=30).status_code == 200)
        c.delete(f"/conversations/{conv2}/share")
        check("revoked share link is gone", httpx.get(f"{api}/shares/{token}", timeout=30).status_code == 404)

        r = c.delete(f"/documents/{doc}")
        check("document delete", r.status_code == 204)
        created.remove(doc)
    finally:
        for d in created:
            c.delete(f"/documents/{d}")
        r = c.delete("/profile/data")
        check("synthetic account's data cleared", r.status_code == 204, f"account {email} remains, empty")
        c.post("/auth/logout")

    passed = sum(ok for _, ok in results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
