"""Print fresh production secrets and where each one goes. Nothing is written to disk.

    python scripts/generate_secrets.py

Paste the values into each Railway service's Variables (never into git, never into Vercel).
Re-run to rotate: change JWT secrets to sign everyone out; change INTERNAL_SERVICE_TOKEN in all
four services together.
"""
import secrets


def main() -> None:
    internal = secrets.token_urlsafe(48)
    jwt_access, jwt_refresh = secrets.token_urlsafe(48), secrets.token_urlsafe(48)
    print("# ---- backend (Railway service: backend)")
    print(f"JWT_SECRET={jwt_access}")
    print(f"JWT_REFRESH_SECRET={jwt_refresh}")
    print(f"INTERNAL_SERVICE_TOKEN={internal}")
    print("\n# ---- the SAME internal token in each internal service")
    for service in ("agent-core", "rag-core", "llm"):
        print(f"# {service}: INTERNAL_SERVICE_TOKEN={internal}")
    print("\n# ---- secrets you obtain from providers (not generated here)")
    print("# backend : DATABASE_URL (Neon, ?sslmode=require), REDIS_URL, STORAGE_ACCESS_KEY/STORAGE_SECRET_KEY,")
    print("#           SMTP_PASSWORD or EMAIL_API_KEY (password-reset email, optional)")
    print("# rag-core: DATABASE_URL (its own database), PINECONE_API_KEY")
    print("# llm     : FREELLMAPI_API_KEY (the hosted FreeLLMAPI's unified key)")
    print("# frontend (Vercel): no secrets — only VITE_API_BASE_URL in direct mode")


if __name__ == "__main__":
    main()
