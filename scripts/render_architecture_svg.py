"""Generate docs/architecture-3d.svg: an isometric (3D-style) picture of the whole system.

    python scripts/render_architecture_svg.py

Pure standard library. Edit FLOORS / COMPONENTS / LINKS / STEPS below when the architecture changes,
re-run, commit the SVG. Every box and arrow matches the code (see README.md, "Architecture").

Layout: three floors stacked like a tower. Top: client and edge. Middle: the four application
services. Bottom: data stores and the external LLM gateway. Arrows that go down a floor are calls
from a service to what it depends on.
"""
import math
from pathlib import Path
from xml.sax.saxutils import escape as esc

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture-3d.svg"
S = 38.0                          # pixels per grid unit
COS, SIN = math.cos(math.radians(30)), 0.5
OX, OY = 370.0, 835.0             # screen position of the grid origin on the ground floor
W, H = 1320, 1215
SIZE = 2.3                        # footprint of every block (grid units)
P = 3.8                           # spacing between blocks (grid units)


def iso(gx, gy, z=0.0):
    return OX + (gx - gy) * COS * S, OY + (gx + gy) * SIN * S - z


def pts(points):
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


# floor key: (z in px, title, subtitle, footprint gx, gy, slab colour)
FLOORS = {
    "edge": (720, "1 · CLIENT & EDGE", "Vercel free tier", 1 * P + SIZE + 0.4, SIZE + 0.4, "#134e4a"),
    "apps": (400, "2 · APPLICATION SERVICES", "four Render free web services", 2 * P + SIZE + 0.4, 1 * P + SIZE + 0.4, "#1e3a8a"),
    "data": (0,   "3 · DATA & EXTERNAL", "Neon · Pinecone · Redis · storage · Railway", 2 * P + SIZE + 0.4, 1 * P + SIZE + 0.4, "#334155"),
}

# key: (floor, name, subtitle lines, local gx, local gy, block height px, (top, left, right))
COMPONENTS = {
    "browser":  ("edge", "Browser",        ["React 19 UI", "SSE chat stream"],             0 * P, 0 * P, 34, ("#5eead4", "#14b8a6", "#0f766e")),
    "vercel":   ("edge", "Vercel",         ["static app · /api proxy", "/wake relay"],     1 * P, 0 * P, 42, ("#c4b5fd", "#8b5cf6", "#6d28d9")),
    "backend":  ("apps", "Backend",        ["FastAPI · auth · jobs", "in-process worker"], 0 * P, 0 * P, 56, ("#93c5fd", "#3b82f6", "#1d4ed8")),
    "agent":    ("apps", "Agent Core",     ["plan · prompt", "citations"],                 1 * P, 0 * P, 56, ("#a5b4fc", "#6366f1", "#4338ca")),
    "llm":      ("apps", "LLM service",    ["provider adapter", "streaming"],              2 * P, 0 * P, 56, ("#fcd34d", "#f59e0b", "#b45309")),
    "rag":      ("apps", "RAG Core",       ["parse · chunk · embed", "hybrid retrieval"],  1 * P, 1 * P, 56, ("#86efac", "#22c55e", "#15803d")),
    "neonapp":  ("data", "Neon: app DB",   ["users · chats", "documents · jobs"],          0 * P, 0 * P, 34, ("#7dd3fc", "#0ea5e9", "#0369a1")),
    "storage":  ("data", "File storage",   ["original uploads", "disk or S3/R2"],          1 * P, 0 * P, 34, ("#cbd5e1", "#94a3b8", "#475569")),
    "redis":    ("data", "Redis",          ["rate limits", "Render Key Value"],            2 * P, 0 * P, 34, ("#fca5a5", "#ef4444", "#b91c1c")),
    "neonrag":  ("data", "Neon: RAG DB",   ["chunks · full-text", "ingestion attempts"],   0 * P, 1 * P, 34, ("#7dd3fc", "#0ea5e9", "#0369a1")),
    "pinecone": ("data", "Pinecone",       ["384-d cosine vectors", "namespace per user"], 1 * P, 1 * P, 34, ("#6ee7b7", "#10b981", "#047857")),
    "freellm":  ("data", "FreeLLMAPI",     ["Railway gateway", "routes to providers"],     2 * P, 1 * P, 34, ("#fda4af", "#f43f5e", "#be123c")),
}

# from, to, label, kind (http | data | wake), label offset (dx, dy)
LINKS = [
    ("browser", "vercel", "HTTPS", "http", (0, 0)),
    ("vercel", "backend", "/api/v1 proxy", "http", (0, 0)),
    ("backend", "agent", "chat turn (SSE)", "http", (0, 0)),
    ("agent", "llm", "generate", "http", (0, 0)),
    ("agent", "rag", "retrieve", "http", (0, 0)),
    ("backend", "rag", "ingest (async)", "http", (0, 0)),
    ("backend", "neonapp", "SQL", "data", (0, 0)),
    ("backend", "storage", "files", "data", (0, 0)),
    ("backend", "redis", "limits", "data", (0, 0)),
    ("rag", "neonrag", "chunks + FTS", "data", (0, 0)),
    ("rag", "pinecone", "vectors", "data", (0, 0)),
    ("llm", "freellm", "OpenAI API", "http", (0, 0)),
    ("vercel", "agent", "wake", "wake", (0, 0)),
    ("vercel", "rag", "", "wake", (0, 0)),
    ("vercel", "llm", "", "wake", (0, 0)),
]
KIND = {"http": ("#38bdf8", None), "data": ("#34d399", None), "wake": ("#fb923c", "6 5")}

STEPS = [
    ("1", "Upload", "Browser → Vercel → Backend. The backend checks the file, stores it and queues a job."),
    ("2", "Index", "The worker sends it to RAG Core: parse, chunk, embed (384-d), chunks to Neon, vectors to Pinecone."),
    ("3", "Ask", "The browser sends a question over SSE. The backend calls Agent Core with the chat history."),
    ("4", "Retrieve", "Agent Core asks RAG Core: semantic + keyword search, fused (RRF) and reranked."),
    ("5", "Answer", "The LLM service streams an answer through FreeLLMAPI. Agent Core adds real citations."),
    ("6", "Stream", "The backend saves the message and streams it back to the browser."),
]
RULES = ["Every document and vector belongs to one user", "Ingestion is idempotent and retries are bounded",
         "Ready only after vectors are read back", "Internal services need a Bearer token"]


def cuboid(gx, gy, sx, sy, z0, h, colours, edge=.35):
    top, left, right = colours
    t = iso
    z1 = z0 + h
    f_top = pts([t(gx, gy, z1), t(gx + sx, gy, z1), t(gx + sx, gy + sy, z1), t(gx, gy + sy, z1)])
    f_left = pts([t(gx, gy + sy, z0), t(gx + sx, gy + sy, z0), t(gx + sx, gy + sy, z1), t(gx, gy + sy, z1)])
    f_right = pts([t(gx + sx, gy, z0), t(gx + sx, gy + sy, z0), t(gx + sx, gy + sy, z1), t(gx + sx, gy, z1)])
    st = f'stroke="#0b1220" stroke-opacity="{edge}"'
    return f'<polygon points="{f_left}" fill="{left}" {st}/><polygon points="{f_right}" fill="{right}" {st}/><polygon points="{f_top}" fill="{top}" {st}/>'


def place(key):
    floor, name, sub, gx, gy, h, colours = COMPONENTS[key]
    return FLOORS[floor][0], gx, gy, h


def top_centre(key):
    z, gx, gy, h = place(key)
    return iso(gx + SIZE / 2, gy + SIZE / 2, z + h)


def arrow(a, b, label, kind, off):
    (x1, y1), (x2, y2) = top_centre(a), top_centre(b)
    colour, dash = KIND[kind]
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy) or 1
    def exit_distance(key):          # how far along the line from the block centre the label pill ends
        w_, h_ = pill(key)
        return min((w_ / 2 + 6) / max(abs(dx / n), 1e-6), (h_ / 2 + 8) / max(abs(dy / n), 1e-6))
    d1, d2 = exit_distance(a), exit_distance(b)
    sx, sy, ex, ey = x1 + dx / n * d1, y1 + dy / n * d1, x2 - dx / n * d2, y2 - dy / n * d2
    da = f' stroke-dasharray="{dash}"' if dash else ""
    out = [f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{colour}" stroke-width="2.5" stroke-linecap="round"{da} marker-end="url(#h-{kind})"/>']
    if label:
        out.append(f'<text x="{(sx + ex) / 2 + off[0]:.1f}" y="{(sy + ey) / 2 + off[1] - 5:.1f}" class="lnk" text-anchor="middle" fill="{colour}">{esc(label)}</text>')
    return "".join(out)


def pill(key):
    floor, name, sub, gx, gy, h, _ = COMPONENTS[key]
    return max(len(name) * 8.0, max(len(s) for s in sub) * 5.9) + 18, 16 + 12.5 * len(sub) + 4


def block_label(key):
    floor, name, sub, gx, gy, h, _ = COMPONENTS[key]
    x, y = top_centre(key)
    w, hgt = pill(key)
    y0 = y - hgt / 2 - 2
    parts = [f'<rect x="{x - w / 2:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{hgt:.1f}" rx="7" fill="#0b1220" fill-opacity=".82"/>',
             f'<text x="{x:.1f}" y="{y0 + 14:.1f}" class="nm" text-anchor="middle">{esc(name)}</text>']
    for i, line in enumerate(sub):
        parts.append(f'<text x="{x:.1f}" y="{y0 + 27 + 12 * i:.1f}" class="sb" text-anchor="middle">{esc(line)}</text>')
    return "".join(parts)


def wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        if len(cur) + len(w_) + 1 > width and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = f"{cur} {w_}".strip()
    return lines + [cur]


def main():
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Segoe UI, Helvetica, Arial, sans-serif">',
           '<title>AdvancedRAG system architecture (isometric 3D view)</title>',
           '<desc>Three stacked floors: Browser and Vercel; Backend, Agent Core, RAG Core and the LLM service; Neon PostgreSQL, Pinecone, Redis, file storage and FreeLLMAPI. Arrows show which service calls which.</desc>',
           '<defs><linearGradient id="bg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0b1220"/><stop offset="1" stop-color="#172036"/></linearGradient>'
           + "".join(f'<marker id="h-{k}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>' for k, (c, _) in KIND.items())
           + '</defs>',
           '<style>.nm{font-size:14px;font-weight:700;fill:#fff}.sb{font-size:10.5px;fill:#cbd5e1}.lnk{font-size:11px;font-weight:600;paint-order:stroke;stroke:#0b1220;stroke-width:4px;stroke-linejoin:round}'
           '.zn{font-size:13px;font-weight:700;fill:#e2e8f0;letter-spacing:1.4px}.zs{font-size:10.5px;fill:#94a3b8}.ttl{font-size:27px;font-weight:800;fill:#fff}.sub{font-size:13px;fill:#94a3b8}'
           '.lg{font-size:11.5px;fill:#cbd5e1}.ph{font-size:15px;font-weight:700;fill:#fff}.pt{font-size:12.5px;fill:#cbd5e1}.pn{font-size:12.5px;font-weight:800;fill:#0b1220}.pk{font-size:12.5px;font-weight:700;fill:#fff}</style>',
           f'<rect width="{W}" height="{H}" rx="18" fill="url(#bg)"/>',
           '<text x="40" y="56" class="ttl">AdvancedRAG: system architecture</text>',
           '<text x="40" y="80" class="sub">Chat with your documents. Isometric view of the deployed system: three floors, four internal services, one gateway to the LLMs.</text>']

    for key in ("data", "apps", "edge"):                              # floors, lowest first
        z, title, subtitle, fx, fy, colour = FLOORS[key]
        svg.append(cuboid(-0.8, -0.8, fx + 0.6, fy + 0.6, z - 14, 14, (colour, "#0b1220", "#0b1220"), edge=.5))
        lx, ly = iso(-0.8, fy - 0.2, z - 14)
        svg.append(f'<g transform="translate({lx + 14:.1f},{ly + 22:.1f}) rotate(30)"><text class="zn">{esc(title)}</text><text y="14" class="zs">{esc(subtitle)}</text></g>')
        for k in sorted((k for k in COMPONENTS if COMPONENTS[k][0] == key), key=lambda k: COMPONENTS[k][3] + COMPONENTS[k][4]):
            _, _, _, gx, gy, h, colours = COMPONENTS[k]
            svg.append(cuboid(gx, gy, SIZE, SIZE, z, h, colours))
    for a, b, label, kind, off in LINKS:
        svg.append(arrow(a, b, label, kind, off))
    for k in COMPONENTS:
        svg.append(block_label(k))

    px, py = 880, 130                                                   # right panel: follow a request
    svg.append(f'<rect x="{px}" y="{py}" width="400" height="{200 + 80 * len(STEPS)}" rx="14" fill="#0b1220" fill-opacity=".72" stroke="#334155"/>')
    svg.append(f'<text x="{px + 24}" y="{py + 34}" class="ph">Follow a request</text>')
    y = py + 64
    for n_, head, body in STEPS:
        svg.append(f'<circle cx="{px + 36}" cy="{y + 2}" r="12" fill="#38bdf8"/><text x="{px + 36}" y="{y + 7}" class="pn" text-anchor="middle">{n_}</text>')
        svg.append(f'<text x="{px + 60}" y="{y + 6}" class="pk">{esc(head)}</text>')
        for i, line in enumerate(wrap(body, 50)):
            svg.append(f'<text x="{px + 60}" y="{y + 24 + 15 * i}" class="pt">{esc(line)}</text>')
        y += 46 + 15 * len(wrap(body, 50))
    svg.append(f'<text x="{px + 24}" y="{y + 6}" class="ph">Guarantees</text>')
    for i, r in enumerate(RULES):
        svg.append(f'<text x="{px + 24}" y="{y + 32 + 20 * i}" class="pt">✓  {esc(r)}</text>')

    lx, ly = px, py + 230 + 80 * len(STEPS)                                                # legend
    svg.append(f'<rect x="{lx}" y="{ly}" width="400" height="84" rx="10" fill="#0b1220" fill-opacity=".7" stroke="#334155"/>')
    for i, (kind, text) in enumerate((("http", "Call between services (HTTPS, SSE, Bearer token)"), ("data", "Reads and writes a data store"),
                                      ("wake", "Wake relay: outside traffic wakes sleeping services"))):
        c, d = KIND[kind]
        da = f' stroke-dasharray="{d}"' if d else ""
        svg.append(f'<line x1="{lx + 18}" y1="{ly + 22 + 22 * i}" x2="{lx + 62}" y2="{ly + 22 + 22 * i}" stroke="{c}" stroke-width="2.6"{da} marker-end="url(#h-{kind})"/>'
                   f'<text x="{lx + 74}" y="{ly + 26 + 22 * i}" class="lg">{esc(text)}</text>')
    svg.append(f'<text x="40" y="{H - 28}" class="sub">Embeddings: BAAI/bge-small-en-v1.5 (384 dims) run inside RAG Core; no external embedding API.</text>')
    svg.append("</svg>")
    OUT.write_text("\n".join(svg), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
