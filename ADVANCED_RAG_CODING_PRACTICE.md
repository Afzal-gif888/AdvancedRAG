# AdvancedRAG Coding Practice

Part of the [master course](ADVANCED_RAG_2026_MASTER_COURSE.md). Self-contained: you can work through this with no AI session, only Python 3.11+.

**What is here and what is not**

- [Module 8](docs/course/MODULE_8_CODING_AND_PRACTICAL_INTERVIEWS.md) holds 30 full exercises (E1–E30) and 3 drills (P1–P3), each with problem, hints, solution, tests, complexity, edge cases and follow-ups. This document **adds** a difficulty ladder over them, a consolidated test-first toolkit, SQL exercises, API-testing patterns, debugging drills and pipeline exercises, so nothing here repeats a Module 8 solution.
- **Update (2026-10-23):** the real implementations that these drills model now include OCR (`rag_core/ocr.py`), text clean-up (`textclean.py`), evaluation metrics with intervals (`evaluation.py`), heading-aware/semantic chunking and upload dedup/versioning; their tests (`rag-core/tests`, `backend/tests/integration/test_durability_versions.py`) are good worked examples to read after you attempt the drills. Drills that say "no OCR" or "no dedup" describe `31bad7e`.
- All code is **educational**. It is not the production code of AdvancedRAG; production equivalents are listed in Module 8 §5.
- Every block marked `# runnable` contains its own assertions and prints `ok` when everything passes. I executed all of them (Python 3.13) as part of validating this course. Blocks using `fastapi`/`httpx` need `pip install fastapi httpx python-multipart`; the SQL blocks use only the standard-library `sqlite3`.
- Method: try the problem first, then open the solution. Write your own tests before reading mine.

## Difficulty ladder (suggested order)

| Level | Goal | Do these |
|---|---|---|
| 1 Warm-up (≈ 30 min) | Python and vector basics | M8 E1, E2, E3, E9, E12; this file: T1 toolkit |
| 2 Core (≈ 2 h) | chunking, batching, metrics | M8 E4–E8, E11, E14, E15; this file: C1–C3 |
| 3 Data and APIs (≈ 3 h) | SQL, FastAPI, tests | M8 E16–E21; this file: S1–S5, A1–A3 |
| 4 Concurrency and LLM (≈ 3 h) | async, mocks, retries | M8 E22–E26, P1; this file: D1–D6 |
| 5 Systems (≈ 4 h) | idempotency, evaluation, whole pipelines | M8 E27–E30, P2, P3; this file: R1–R3 |

---

## T1. Test-first toolkit (one file you can keep)

Reference implementations of the functions that interviews ask for most, plus the test suite. Copy to `rag_toolkit.py`.

```python
# runnable
import hashlib, heapq, math, re
from collections import Counter

# ---------- vectors
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def norm(a): return math.sqrt(dot(a, a))
def unit(a):
    n = norm(a)
    return [x / n for x in a] if n else list(a)
def cosine(a, b):
    na, nb = norm(a), norm(b)
    return dot(a, b) / (na * nb) if na and nb else 0.0

def top_k(query, vectors, k):
    """vectors: {id: vec}. Highest cosine first, ties by id."""
    scored = ((-cosine(query, v), vid) for vid, v in vectors.items())
    return [(vid, -s) for s, vid in heapq.nsmallest(max(k, 0), scored)]

# ---------- text
def normalize(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(re.sub(r"[ \t\f\v]+", " ", l).strip() for l in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def chunk(text, size, overlap):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("need size > 0 and 0 <= overlap < size")
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        if i + size >= len(text):
            break
        i += size - overlap
    return out

def hash_embed(text, dim=32):
    v = [0.0] * dim
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1.0
    return unit(v)

# ---------- fusion and metrics
def rrf(rankings, k=60):
    s = {}
    for r in rankings:
        for i, d in enumerate(r, 1):
            s[d] = s.get(d, 0.0) + 1.0 / (k + i)
    return [d for d, _ in sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))]

def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant) if relevant else 0.0
def precision_at_k(ranked, relevant, k): return len(set(ranked[:k]) & set(relevant)) / k
def reciprocal_rank(ranked, relevant): return next((1 / i for i, d in enumerate(ranked, 1) if d in relevant), 0.0)
def ndcg_at_k(gains, k):
    dcg = lambda g: sum(x / math.log2(i + 1) for i, x in enumerate(g, 1))
    best = dcg(sorted(gains, reverse=True)[:k])
    return dcg(gains[:k]) / best if best else 0.0
def percentile(values, p):
    s = sorted(values); return s[max(math.ceil(p / 100 * len(s)), 1) - 1]

# ---------- tests (pytest would collect functions named test_*; here we call them directly)
def test_vectors():
    assert abs(cosine([3, 4], [4, 3]) - 0.96) < 1e-12
    assert cosine([0, 0], [1, 1]) == 0.0
    assert abs(norm(unit([3, 4])) - 1) < 1e-12
    assert top_k([1, 0], {"b": [1, 0], "a": [1, 0], "c": [0, 1]}, 2)[0][0] == "a"      # tie -> id order
    assert top_k([1, 0], {}, 3) == [] and top_k([1, 0], {"a": [1, 0]}, 0) == []

def test_text():
    assert normalize("a  b\r\n\r\n\r\n\r\nc") == "a b\n\nc"
    assert chunk("abcdefghij", 4, 1) == ["abcd", "defg", "ghij"]
    assert "".join(c[:3] for c in chunk("abcdefghij", 4, 1)[:-1]) + chunk("abcdefghij", 4, 1)[-1] == "abcdefghij"
    for bad in [(0, 0), (3, 3), (3, -1)]:
        try: chunk("x", *bad)
        except ValueError: pass
        else: raise AssertionError(bad)
    e = hash_embed("hotel limit"); assert len(e) == 32 and abs(norm(e) - 1) < 1e-9

def test_fusion_and_metrics():
    assert rrf([list("ABCD"), list("CAEB")]) == list("ACBED")
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c"}, 3) == 2 / 3
    assert precision_at_k(["a", "x", "b"], {"a", "b", "c"}, 3) == 2 / 3
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5 and reciprocal_rank(["x"], {"a"}) == 0.0
    assert abs(ndcg_at_k([3, 2, 0, 1], 4) - 0.98544) < 1e-4
    assert percentile(list(range(1, 101)), 95) == 95

for name, fn in list(globals().items()):
    if name.startswith("test_"):
        fn()
print("ok")
```

---

## Level 2: three new core problems

### C1. Word-boundary chunking

**Problem.** Chunk `text` into pieces of at most `size` characters, but never cut inside a word (unless a single word is longer than `size`).
**Example.** `chunk_words("alpha beta gamma delta", 11)` → `["alpha beta", "gamma delta"]`.
**Hints.** Split on whitespace; accumulate words; flush when adding the next word would exceed `size`.

```python
# runnable
def chunk_words(text, size):
    if size <= 0:
        raise ValueError("size must be positive")
    chunks, cur = [], ""
    for w in text.split():
        if len(w) > size:                                   # a single over-long word is hard-split
            if cur: chunks.append(cur); cur = ""
            chunks += [w[i:i + size] for i in range(0, len(w), size)]
            continue
        candidate = w if not cur else cur + " " + w
        if len(candidate) > size:
            chunks.append(cur); cur = w
        else:
            cur = candidate
    if cur: chunks.append(cur)
    return chunks

assert chunk_words("alpha beta gamma delta", 11) == ["alpha beta", "gamma delta"]
assert chunk_words("", 5) == [] and chunk_words("   ", 5) == []
assert chunk_words("abcdefghij kl", 4) == ["abcd", "efgh", "ij", "kl"]       # the over-long word is hard-split
assert all(len(c) <= 11 for c in chunk_words("alpha beta gamma delta", 11))
print("ok")
```
*Complexity:* O(n). *Edge cases:* long words, multiple spaces (lost: whitespace is normalised, state this), empty. *Follow-up:* "Preserve original whitespace?" (Split with `re.finditer(r'\S+\s*')`.)

### C2. Generator batching

**Problem.** Write `batched(iterable, n)` yielding lists of up to `n` items without loading the whole iterable (the last batch may be shorter).

```python
# runnable
from itertools import islice

def batched(iterable, n):
    if n <= 0:
        raise ValueError("n must be positive")
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]
assert list(batched([], 3)) == []
def endless():
    i = 0
    while True: yield i; i += 1
assert next(batched(endless(), 3)) == [0, 1, 2]                 # lazy: works on infinite input
print("ok")
```
*Use:* slicing documents into embedding batches or 100-vector upsert requests. *Follow-up:* "Python 3.12 has `itertools.batched`; why implement it?" (Interview practice; the builtin yields tuples.)

### C3. Hit@K from labelled questions (JSONL)

**Problem.** Parse JSONL lines `{"q":..., "relevant":[ids]}` and a dict of retrieved ranked ids per question; report Hit@K and MRR, ignoring (and counting) malformed lines.

```python
# runnable
import json

def evaluate(jsonl_lines, retrieved, k=6):
    ok = bad = 0; hits = rr = 0.0
    for line in jsonl_lines:
        try:
            row = json.loads(line); q, rel = row["q"], set(row["relevant"])
            if not rel: raise ValueError
        except (ValueError, KeyError, TypeError):
            bad += 1; continue
        ranked = retrieved.get(q, [])
        ok += 1
        hits += 1.0 if set(ranked[:k]) & rel else 0.0
        rr += next((1 / i for i, d in enumerate(ranked, 1) if d in rel), 0.0)
    return {"n": ok, "bad": bad, f"hit@{k}": hits / ok if ok else None, "mrr": rr / ok if ok else None}

lines = ['{"q": "hotel", "relevant": ["c1"]}', '{"q": "remote", "relevant": ["c9"]}', "not json", '{"q": "x", "relevant": []}']
res = evaluate(lines, {"hotel": ["c1"], "remote": ["c3", "c9"]})
assert res == {"n": 2, "bad": 2, "hit@6": 1.0, "mrr": 0.75}
assert evaluate([], {})["hit@6"] is None
print("ok")
```
*Follow-up:* "Why report `bad`?" (Silent data loss makes metrics look better.)

---

## Level 3: SQL exercises

Schema used (SQLite syntax; PostgreSQL differs only in date functions):

```sql
users(id, email)
conversations(id, user_id, title)
messages(id, conversation_id, role, content, created_at)
documents(id, user_id, filename, checksum, size_bytes, status, updated_at)   -- updated_at: integer seconds
background_jobs(id, job_type, status, created_at, attempts)                  -- created_at: integer seconds
```

### S1. Latest message per conversation (window function)

```python
# runnable
import sqlite3
db = sqlite3.connect(":memory:")
db.executescript("""
CREATE TABLE messages (id INTEGER PRIMARY KEY, conversation_id INT, role TEXT, content TEXT, created_at INT);
INSERT INTO messages VALUES (1,10,'user','q1',100),(2,10,'assistant','a1',101),(3,11,'user','q2',200),(4,10,'user','q3',300);
""")
rows = db.execute("""
SELECT conversation_id, content FROM (
  SELECT conversation_id, content, ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY created_at DESC, id DESC) AS rn
  FROM messages) WHERE rn = 1 ORDER BY conversation_id
""").fetchall()
assert rows == [(10, "q3"), (11, "q2")]
print("ok")
```
*Alternative:* correlated subquery on `MAX(created_at)` (ties return duplicates). *Follow-up:* index `(conversation_id, created_at)`.

### S2. Stuck documents

**Problem.** List documents in `processing` that have not been updated for more than 600 seconds as of `now`.

```python
# runnable
import sqlite3
db = sqlite3.connect(":memory:")
db.executescript("""
CREATE TABLE documents (id INTEGER PRIMARY KEY, filename TEXT, status TEXT, updated_at INT);
INSERT INTO documents VALUES (1,'a.pdf','processing',1000),(2,'b.pdf','processing',1900),(3,'c.pdf','ready',100),(4,'d.pdf','uploaded',900);
""")
now = 2000
rows = db.execute("SELECT id FROM documents WHERE status IN ('processing','uploaded') AND updated_at < ? ORDER BY id", (now - 600,)).fetchall()
assert rows == [(1,), (4,)]            # 'uploaded' too: a lost job leaves that state (the reconciler handles both)
print("ok")
```
*Follow-up:* "Why include `uploaded`?" (Module 1: a lost job leaves the document uploaded forever.)

### S3. Duplicate uploads by checksum

```python
# runnable
import sqlite3
db = sqlite3.connect(":memory:")
db.executescript("""
CREATE TABLE documents (id INTEGER PRIMARY KEY, user_id INT, filename TEXT, checksum TEXT, size_bytes INT);
INSERT INTO documents VALUES (1,1,'a.pdf','h1',10),(2,1,'copy.pdf','h1',10),(3,2,'a.pdf','h1',10),(4,1,'b.pdf','h2',20);
""")
rows = db.execute("""
SELECT user_id, checksum, COUNT(*) AS copies, SUM(size_bytes) AS bytes
FROM documents GROUP BY user_id, checksum HAVING COUNT(*) > 1
""").fetchall()
assert rows == [(1, "h1", 2, 20)]      # duplicates are per user; user 2's identical file is not a duplicate
# the fix: a unique index makes the database enforce it
db.execute("DELETE FROM documents WHERE id = 2")
db.execute("CREATE UNIQUE INDEX uq_user_checksum ON documents(user_id, checksum)")
try:
    db.execute("INSERT INTO documents VALUES (9,1,'again.pdf','h1',10)")
except sqlite3.IntegrityError:
    pass
else:
    raise AssertionError("unique index must reject the duplicate")
print("ok")
```
*Link:* the project stores the checksum but does not use it (Module 3 §4.6).

### S4. Queue health report

```python
# runnable
import sqlite3
db = sqlite3.connect(":memory:")
db.executescript("""
CREATE TABLE background_jobs (id INTEGER PRIMARY KEY, job_type TEXT, status TEXT, created_at INT);
INSERT INTO background_jobs VALUES (1,'document_ingest','pending',100),(2,'document_ingest','running',150),
 (3,'document_ingest','pending',400),(4,'document_delete','failed',50),(5,'document_delete','pending',500);
""")
now = 1000
rows = db.execute("""
SELECT job_type,
       SUM(status='pending') AS pending, SUM(status='running') AS running, SUM(status='failed') AS failed,
       MAX(CASE WHEN status='pending' THEN ? - created_at END) AS oldest_pending_age
FROM background_jobs GROUP BY job_type ORDER BY job_type
""", (now,)).fetchall()
assert rows == [("document_delete", 1, 0, 1, 500), ("document_ingest", 2, 1, 0, 900)]
print("ok")
```
*Use:* alert on `oldest_pending_age` (queue stalled) rather than on counts alone.

### S5. Running storage total per user (window sum)

```python
# runnable
import sqlite3
db = sqlite3.connect(":memory:")
db.executescript("""
CREATE TABLE documents (id INTEGER PRIMARY KEY, user_id INT, size_bytes INT, created_at INT);
INSERT INTO documents VALUES (1,1,100,1),(2,1,50,2),(3,2,70,1),(4,1,25,3);
""")
rows = db.execute("""
SELECT id, SUM(size_bytes) OVER (PARTITION BY user_id ORDER BY created_at, id) AS running
FROM documents ORDER BY id
""").fetchall()
assert rows == [(1, 100), (2, 150), (3, 70), (4, 175)]
print("ok")
```
*Use:* find the upload that crossed a per-user quota (the production check is a single `SUM` before insert).

---

## Level 3: API testing patterns

### A1. Override an auth dependency in tests

```python
# runnable
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

app = FastAPI()
DOCS = {"alice": ["d1"], "bob": ["d2"]}

async def current_user(authorization: str = Header(default="")):
    token = authorization.removeprefix("Bearer ")
    if token not in DOCS:
        raise HTTPException(401, "invalid token")
    return token

@app.get("/documents/{doc_id}")
async def get_doc(doc_id: str, user: str = Depends(current_user)):
    if doc_id not in DOCS[user]:
        raise HTTPException(404, "not found")           # 404 not 403: do not reveal existence
    return {"id": doc_id}

c = TestClient(app)
assert c.get("/documents/d1").status_code == 401
assert c.get("/documents/d1", headers={"Authorization": "Bearer alice"}).status_code == 200
assert c.get("/documents/d2", headers={"Authorization": "Bearer alice"}).status_code == 404     # IDOR check
app.dependency_overrides[current_user] = lambda: "bob"                                           # skip real auth in a test
assert c.get("/documents/d2").json() == {"id": "d2"}
app.dependency_overrides.clear()
print("ok")
```

### A2. Async test with ASGITransport (no server, no TestClient)

```python
# runnable
import asyncio, httpx
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
async def health(): return {"status": "ok"}

async def main():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/health")
        assert r.status_code == 200 and r.json() == {"status": "ok"}
        assert (await c.get("/nope")).status_code == 404
asyncio.run(main())
print("ok")
```
*Use:* this is the style the project's own test suites use for the async stack.

### A3. Error envelope and request-id contract

```python
# runnable
import re, uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

VALID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
app = FastAPI()

@app.middleware("http")
async def request_id(request: Request, call_next):
    incoming = request.headers.get("x-request-id", "")
    rid = incoming if VALID.match(incoming) else uuid.uuid4().hex          # never reflect hostile input
    request.state.rid = rid
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response

@app.exception_handler(Exception)
async def boom(request: Request, exc: Exception):
    # An unhandled exception is answered by Starlette's outermost error layer, *outside* the middleware above,
    # so this handler must add the header itself (a classic trap: the middleware never sees this response).
    return JSONResponse(status_code=500, headers={"x-request-id": request.state.rid},
                        content={"error": {"code": "INTERNAL_ERROR", "message": "Internal error.", "request_id": request.state.rid}})

@app.get("/crash")
async def crash(): raise RuntimeError("SELECT * FROM users -- secret dsn postgresql://u:p@db")

c = TestClient(app, raise_server_exceptions=False)
r = c.get("/crash", headers={"X-Request-ID": "trace-12345678"})
assert r.status_code == 500 and r.headers["x-request-id"] == "trace-12345678"
assert r.json()["error"]["request_id"] == "trace-12345678"
assert "SELECT" not in r.text and "postgresql" not in r.text                 # internals never leak
bad = c.get("/crash", headers={"X-Request-ID": "bad id"})
assert bad.headers["x-request-id"] != "bad id" and len(bad.headers["x-request-id"]) == 32
print("ok")
```
*Mirrors:* `test_errors_carry_request_id_and_no_internals` in the backend suite.

---

## Level 4: debugging drills

Each drill shows buggy code, a test that exposes the bug, and the fix. Try to spot the bug before reading.

### D1. The infinite loop

```python
# runnable
def chunk_buggy(text, size, overlap):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap            # BUG: if overlap >= size the index never advances (or goes backwards)
    return out

def chunk_fixed(text, size, overlap):
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out

# demonstrate the bug safely with an iteration cap instead of hanging
steps, i, size, overlap = 0, 0, 4, 4
while i < 10 and steps < 100:
    i += size - overlap; steps += 1
assert steps == 100                    # step 0: never terminates
try: chunk_fixed("abc", 4, 4)
except ValueError: pass
else: raise AssertionError
assert chunk_fixed("abcdefgh", 4, 1) == ["abcd", "defg", "gh"]
print("ok")
```
*Lesson:* validate parameters at the boundary; the production config validator requires `CHUNK_OVERLAP < CHUNK_SIZE`.

### D2. Shared mutable default

```python
# runnable
def add_chunk_buggy(chunk, seen=[]):           # BUG: one list shared by every call
    seen.append(chunk); return seen

def add_chunk(chunk, seen=None):
    seen = [] if seen is None else seen
    seen.append(chunk); return seen

assert add_chunk_buggy("a") == ["a"] and add_chunk_buggy("b") == ["a", "b"]      # state leaked between calls
assert add_chunk("a") == ["a"] and add_chunk("b") == ["b"]
print("ok")
```

### D3. MRR that divides by the wrong count

```python
# runnable
def mrr_buggy(ranks):                          # ranks: first-relevant rank or None
    found = [1 / r for r in ranks if r]
    return sum(found) / len(found)             # BUG: questions that failed are ignored -> inflated

def mrr(ranks):
    return sum(1 / r for r in ranks if r) / len(ranks)

ranks = [1, None, 2, None]
assert round(mrr_buggy(ranks), 3) == 0.75      # looks great...
assert mrr(ranks) == 0.375                     # ...the true value counts the misses as 0
print("ok")
```

### D4. Lost order and shared state in async batches

```python
# runnable
import asyncio

async def embed_all_buggy(texts):
    out = []
    async def one(t):
        await asyncio.sleep(0.001 * (10 - len(t)))      # shorter texts "finish" later
        out.append(len(t))                              # BUG: append in completion order
    await asyncio.gather(*(one(t) for t in texts))
    return out

async def embed_all(texts):
    async def one(t):
        await asyncio.sleep(0.001 * (10 - len(t)))
        return len(t)
    return await asyncio.gather(*(one(t) for t in texts))     # gather returns results in input order

texts = ["aaaaaaaa", "b", "ccc"]
assert asyncio.run(embed_all_buggy(texts)) != [8, 1, 3]      # completion order scrambled the results
assert asyncio.run(embed_all(texts)) == [8, 1, 3]
print("ok")
```

### D5. LEFT JOIN that silently becomes an INNER JOIN

```python
# runnable
import sqlite3
db = sqlite3.connect(":memory:")
db.executescript("""
CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT);
CREATE TABLE documents (id INTEGER PRIMARY KEY, user_id INT, status TEXT);
INSERT INTO users VALUES (1,'a'),(2,'b');
INSERT INTO documents VALUES (1,1,'ready'),(2,1,'failed');
""")
buggy = db.execute("SELECT u.email, COUNT(d.id) FROM users u LEFT JOIN documents d ON d.user_id=u.id WHERE d.status='ready' GROUP BY u.id").fetchall()
fixed = db.execute("SELECT u.email, COUNT(d.id) FROM users u LEFT JOIN documents d ON d.user_id=u.id AND d.status='ready' GROUP BY u.id ORDER BY u.id").fetchall()
assert buggy == [("a", 1)]                          # user b vanished: WHERE on the right table removed the NULL row
assert fixed == [("a", 1), ("b", 0)]
print("ok")
```

### D6. Retrying the non-retryable

```python
# runnable
import asyncio

class HTTPFail(Exception):
    def __init__(self, status): self.status = status

async def call_buggy(send, retries=3):
    for _ in range(retries):
        try: return await send()
        except HTTPFail: continue                  # BUG: retries 401/400 too and hides the real error
    raise RuntimeError("failed")

async def call(send, retries=3):
    for attempt in range(retries):
        try: return await send()
        except HTTPFail as e:
            if e.status not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise

def counter(status):
    n = {"calls": 0}
    async def send():
        n["calls"] += 1; raise HTTPFail(status)
    return send, n

s, n = counter(401)
try: asyncio.run(call_buggy(s))
except RuntimeError: pass
assert n["calls"] == 3                               # bad credentials retried 3 times: wasted calls, lockout risk
s, n = counter(401)
try: asyncio.run(call(s))
except HTTPFail as e: assert e.status == 401 and n["calls"] == 1
s, n = counter(503)
try: asyncio.run(call(s))
except HTTPFail: assert n["calls"] == 3
print("ok")
```

---

## Level 5: pipeline exercises

### R1. A minimal end-to-end RAG pipeline with abstention

**Problem.** Ingest documents (chunk → embed → store), answer a question (embed → retrieve → build prompt → call an injected LLM), abstain when nothing relevant is found, and return citations that only reference supplied sources.

```python
# runnable
import hashlib, math, re

def embed(text, dim=64):
    v = [0.0] * dim
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

def cos(a, b): return sum(x * y for x, y in zip(a, b))

class Pipeline:
    def __init__(self, llm, min_score=0.3, k=3, size=80, overlap=20):
        self.llm, self.min_score, self.k, self.size, self.overlap, self.store = llm, min_score, k, size, overlap, {}
    def ingest(self, user, doc_id, text):
        step = self.size - self.overlap
        pieces = [text[i:i + self.size] for i in range(0, max(len(text) - self.overlap, 1), step)]
        for n, p in enumerate(pieces):                              # deterministic ids -> idempotent
            self.store[(user, f"{doc_id}:{n}")] = (embed(p), p, doc_id)
        for key in [k for k in self.store if k[0] == user and k[1].startswith(f"{doc_id}:") and int(k[1].split(":")[1]) >= len(pieces)]:
            del self.store[key]
    def ask(self, user, question):
        q = embed(question)
        scored = sorted(((cos(q, v), cid, t, d) for (u, cid), (v, t, d) in self.store.items() if u == user), reverse=True)
        hits = [h for h in scored[:self.k] if h[0] >= self.min_score]
        if not hits:
            return {"answer": "I couldn't find sufficient evidence.", "citations": [], "llm_calls": 0}
        context = "\n".join(f"[{i}] {t}" for i, (_, _, t, _) in enumerate(hits, 1))
        answer = self.llm(f"Answer only from the sources and cite [n].\n{context}\nQ: {question}")
        valid = {int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(hits)}
        return {"answer": answer, "citations": sorted(valid), "llm_calls": 1}

calls = []
def llm(prompt): calls.append(prompt); return "Hotels are reimbursed up to 180 euros per night [1] and [9]."

p = Pipeline(llm)
p.ingest("alice", "handbook", "Hotel stays are reimbursed up to 180 euros per night in all European cities. Receipts within 30 days.")
p.ingest("bob", "secret", "Bob's launch code is 4471.")
r = p.ask("alice", "What is the hotel reimbursement limit per night?")
assert r["citations"] == [1] and r["llm_calls"] == 1                          # [9] was never a source
assert "4471" not in calls[-1]                                                # other user's text never reaches the prompt
assert p.ask("alice", "Who won the football world cup?")["llm_calls"] == 0    # abstained without calling the LLM
n_before = len(p.store); p.ingest("alice", "handbook", "Hotel stays are reimbursed up to 180 euros per night in all European cities. Receipts within 30 days.")
assert len(p.store) == n_before                                               # re-ingest is idempotent
p.ingest("alice", "handbook", "Short.")                                       # shorter new version removes stale chunks
assert [k for k in p.store if k[0] == "alice" and k[1].startswith("handbook:")] == [("alice", "handbook:0")]
print("ok")
```
*Discuss:* where the toy differs (score threshold vs LLM grading, brute force vs ANN, no verifier, no history).

### R2. Crash-safe ingestion

**Problem.** Simulate an ingestion that crashes after writing some vectors. Show that recording the extent *before* writing lets a later cleanup remove everything.

```python
# runnable
class CrashingIndex:
    def __init__(self): self.vectors, self.extent = {}, {}
    def ingest(self, doc, chunks, crash_after=None):
        self.extent[doc] = max(self.extent.get(doc, 0), len(chunks))      # durable "may-exist" record FIRST
        for i, c in enumerate(chunks):
            if crash_after is not None and i == crash_after:
                raise RuntimeError("crash")
            self.vectors[f"{doc}:{i}"] = c
    def cleanup(self, doc):
        for i in range(self.extent.get(doc, 0)):
            self.vectors.pop(f"{doc}:{i}", None)
        self.extent.pop(doc, None)

ix = CrashingIndex()
try: ix.ingest("d", ["a", "b", "c", "d"], crash_after=2)
except RuntimeError: pass
assert set(ix.vectors) == {"d:0", "d:1"}                  # partial state exists
ix.cleanup("d")
assert ix.vectors == {} and ix.extent == {}               # nothing left behind

# the buggy ordering: record the extent only after writing -> the partial writes are untracked
class Buggy(CrashingIndex):
    def ingest(self, doc, chunks, crash_after=None):
        for i, c in enumerate(chunks):
            if crash_after is not None and i == crash_after: raise RuntimeError("crash")
            self.vectors[f"{doc}:{i}"] = c
        self.extent[doc] = len(chunks)
b = Buggy()
try: b.ingest("d", ["a", "b", "c"], crash_after=1)
except RuntimeError: pass
b.cleanup("d")
assert set(b.vectors) == {"d:0"}                          # orphan vector survives: the reason for "extent first"
print("ok")
```

### R3. Hybrid retrieval with graceful degradation

**Problem.** Combine a dense retriever and a keyword retriever with RRF. If the dense retriever raises, fall back to keyword-only and report the strategy; if both fail, raise.

```python
# runnable
class Unavailable(Exception): ...

def rrf(rankings, k=60):
    s = {}
    for r in rankings:
        for i, d in enumerate(r, 1): s[d] = s.get(d, 0) + 1 / (k + i)
    return [d for d, _ in sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))]

def retrieve(query, dense, keyword, top_k=3):
    strategy, rankings = "hybrid_rrf", []
    try:
        rankings.append(dense(query))
    except Unavailable:
        strategy = "keyword_only_degraded"
    kw = keyword(query)
    rankings.append(kw)
    if strategy != "hybrid_rrf" and not kw:
        raise Unavailable("no dense results and no keyword results")
    return {"ids": rrf(rankings)[:top_k], "strategy": strategy}

def dense_ok(q): return ["a", "b", "c"]
def dense_down(q): raise Unavailable()
def kw(q): return ["c", "a"]

assert retrieve("q", dense_ok, kw) == {"ids": ["a", "c", "b"], "strategy": "hybrid_rrf"}
assert retrieve("q", dense_down, kw) == {"ids": ["c", "a"], "strategy": "keyword_only_degraded"}
try: retrieve("q", dense_down, lambda q: [])
except Unavailable: pass
else: raise AssertionError("both empty while degraded must raise")
print("ok")
```
*Mirrors:* `pipeline.retrieve` (Module 2 §5). *Follow-up:* "Why raise rather than return empty?" (An empty answer would be indistinguishable from "no relevant documents".)

---

## Self-check

- [ ] Level 1–2 done without hints; Level 3 with at most one hint per problem.
- [ ] For every solution you can state time and space complexity and two edge cases.
- [ ] You can explain how each educational snippet differs from the production code (Module 8 §5).
- [ ] You ran every `# runnable` block and all printed `ok`.
