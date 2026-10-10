"""Write frontend/vercel.json for a production deployment.

    # A) Vercel forwards /api/* to the Railway backend (works on default *.vercel.app domains;
    #    the API becomes same-origin, so the refresh cookie is first-party). Backend: TRUSTED_PROXY_HOPS=2.
    python scripts/configure_vercel.py --mode proxy --backend-url https://advancerag-backend.up.railway.app

    # B) Own domain: app.example.com (Vercel) calls api.example.com (Railway) directly (same site).
    #    Build with VITE_API_BASE_URL=https://api.example.com/api/v1. Backend: TRUSTED_PROXY_HOPS=1.
    python scripts/configure_vercel.py --mode direct --backend-url https://api.example.com

    # Render free: also give each sleeping service a wake path (Render wakes a service only for
    # requests from outside Render, so the services wake each other through these). Each maps
    # /wake/<first label of the host> to that service's unauthenticated GET /health, nothing else.
    python scripts/configure_vercel.py --mode proxy --backend-url https://advancerag-backend.onrender.com         --wake-url https://advancerag-agent-core.onrender.com --wake-url https://advancerag-rag-core.onrender.com         --wake-url https://advancerag-llm.onrender.com

Always: SPA fallback (client-side routes like /share/<token> and /reset-password) and strict
security headers. The Content-Security-Policy allows only this site plus, in direct mode, the API.
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def build(mode: str, backend_url: str | None, wake_urls: list[str] | None = None) -> dict:
    connect = ["'self'"]
    rewrites = []
    for url in wake_urls or []:
        u = urlparse(url)
        if u.scheme != "https" or not u.hostname:
            raise SystemExit("--wake-url must be an https URL, e.g. https://advancerag-llm.onrender.com")
        rewrites.append({"source": f"/wake/{u.hostname.split('.')[0]}", "destination": f"https://{u.netloc}/health"})
    if backend_url:
        u = urlparse(backend_url)
        if u.scheme != "https" or not u.hostname:
            raise SystemExit("--backend-url must be an https URL, e.g. https://advancerag-backend.up.railway.app")
        origin = f"{u.scheme}://{u.netloc}"
        if mode == "proxy":
            rewrites.append({"source": "/api/:path*", "destination": f"{origin}/api/:path*"})
        else:
            connect.append(origin)
    rewrites.append({"source": "/((?!api/|assets/|favicon\\.svg).*)", "destination": "/index.html"})
    csp = "; ".join([
        "default-src 'self'",
        "script-src 'self'",
        "style-src 'self' 'unsafe-inline'",   # React inline style attributes
        "img-src 'self' data:",               # answers never load remote images (blocked in Markdown)
        "font-src 'self'",
        f"connect-src {' '.join(connect)}",
        "frame-ancestors 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "object-src 'none'",
    ])
    security = [
        {"key": "Content-Security-Policy", "value": csp},
        {"key": "X-Content-Type-Options", "value": "nosniff"},
        {"key": "X-Frame-Options", "value": "DENY"},
        {"key": "Referrer-Policy", "value": "no-referrer"},   # share/reset tokens live in URLs
        {"key": "Permissions-Policy", "value": "camera=(), microphone=(), geolocation=(), payment=()"},
        {"key": "Strict-Transport-Security", "value": "max-age=63072000; includeSubDomains"},
    ]
    return {
        "$schema": "https://openapi.vercel.sh/vercel.json",
        "framework": "vite",
        "buildCommand": "npm run build",
        "outputDirectory": "dist",
        "rewrites": rewrites,
        "headers": [
            {"source": "/(.*)", "headers": security},
            {"source": "/assets/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=31536000, immutable"}]},
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["proxy", "direct"], default="proxy")
    ap.add_argument("--backend-url", help="https URL of the Railway backend (omit to write SPA/headers only)")
    ap.add_argument("--wake-url", action="append", default=[], help="https URL of a service to expose at /wake/<name> (repeatable)")
    ap.add_argument("--out", default=str(ROOT / "frontend" / "vercel.json"))
    args = ap.parse_args()
    cfg = build(args.mode, args.backend_url, args.wake_url)
    Path(args.out).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out} (mode={args.mode}, backend={'set' if args.backend_url else 'not set'})")


if __name__ == "__main__":
    main()
