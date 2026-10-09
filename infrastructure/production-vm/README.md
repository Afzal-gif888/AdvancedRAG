# Single-VM production deployment (free-tier friendly)

All application services on **one virtual machine** with Docker Compose. Managed free tiers are used
for everything else:

```
browser ──https──▶ app.<domain>   Vercel (frontend, static)
browser ──https──▶ api.<domain>   VM: Caddy (TLS) ─▶ backend ─▶ agent-core ─▶ rag-core ─▶ Neon advancerag_rag
                                                 │            └▶ llm ─▶ freellmapi ─▶ LLM providers   └▶ Pinecone advancerag-bge384
                                                 ├▶ redis (rate limits)
                                                 └▶ Neon advancerag
```

- **Public:** Caddy only (80, 443).
- **Private Docker network:** backend, agent-core, rag-core, llm, freellmapi, redis.
- **FreeLLMAPI dashboard:** bound to the VM's loopback (`127.0.0.1:3001`); reach it with an SSH tunnel.
- **Frozen, set in `compose.yml`:** index `advancerag-bge384`, model `BAAI/bge-small-en-v1.5` (384 dims).

## Sizing (measured)

Measured with the production images during the rehearsal and a 300-page PDF + 5,000-row CSV ingestion:

| Service | Peak RAM | Notes |
|---|---|---|
| rag-core | **1,107 MiB** (276 MiB idle with the model loaded) | CPU-bound while embedding; uses every core |
| backend | 115 MiB | |
| llm | 52 MiB | |
| agent-core | 45 MiB | |
| freellmapi | about 50–150 MiB | Node.js; the measured 49 MiB was the mock |
| redis, caddy | under 50 MiB together | |
| **Total** | **about 1.5 GB at peak** | plus about 300 MB for the OS and Docker |

Disk: images about 3 GB (rag-core about 740 MB with the model baked in; FreeLLMAPI about 860 MB), plus
uploads if `STORAGE_BACKEND=local`. Use at least a **20 GB** disk.

**Minimum:** 2 vCPU, 2 GB RAM + 4 GB swap (`setup-vm.sh` adds it). Large ingestions then swap and run
slower, but don't crash. **Comfortable:** 4 GB RAM.

## Steps

1. **VM**: Ubuntu 22.04/24.04, x86_64. The images build on arm64 too, but that is not verified.
   Open inbound TCP 22 (your IP only), 80 and 443 in the cloud firewall.
2. **DNS**: an A record (and AAAA, if the VM has IPv6) for `api.<domain>` → the VM's public IP.
   Caddy obtains the certificate automatically once DNS resolves and ports 80/443 are open.
3. **Setup**: copy the repository to the VM (git clone with a read-only deploy key, or `scp`), then:
   ```
   sudo bash infrastructure/production-vm/setup-vm.sh      # Docker, 4 GB swap, firewall
   ```
4. **Secrets**: in `infrastructure/production-vm/`:
   ```
   cp .env.example .env
   for f in env/*.env.example; do cp "$f" "${f%.example}"; done
   chmod 600 .env env/*.env
   ```
   Fill in every value (see the comments in each file). Never commit these files; they are git-ignored.
5. **Neon**: databases `advancerag` and `advancerag_rag`. Use the **pooled** connection strings with
   `sslmode=require` in `env/backend.env` and `env/rag-core.env`. Migrations run on container start:
   backend through revision 0004, rag-core through revision 0003, each only against its own database.
6. **Start FreeLLMAPI first**, then add provider keys through the dashboard over an SSH tunnel:
   ```
   docker compose up -d freellmapi
   # on your PC:  ssh -L 3001:127.0.0.1:3001 <user>@<vm>   then open http://localhost:3001
   ```
   Sign in with the admin from `env/freellmapi.env`, add your provider keys, copy the unified key into
   `env/llm.env` (`FREELLMAPI_API_KEY`), and close the tunnel.
7. **Everything**:
   ```
   docker compose up -d --build
   docker compose ps                         # all healthy
   curl https://api.<domain>/ready           # database, storage, agent_core, rag_core: true
   docker compose exec llm python -m llm_service.healthcheck --chat
   ```
8. **Frontend** (Vercel, custom domain `app.<domain>` on the same site as the API):
   ```
   python scripts/configure_vercel.py --mode direct --backend-url https://api.<domain>
   ```
   Set `VITE_API_BASE_URL=https://api.<domain>/api/v1` in Vercel, root directory `frontend`.
   Set `FRONTEND_ORIGIN` / `FRONTEND_PUBLIC_URL` in `env/backend.env` to `https://app.<domain>`, then
   run `docker compose up -d backend`.
9. **Verify end to end** (with your PC off):
   ```
   python scripts/smoke_production.py --url https://app.<domain> --backend-url https://api.<domain>
   ```

## Why direct mode (own domain)

The browser calls `api.<domain>` directly, and Caddy is the only proxy, so `TRUSTED_PROXY_HOPS=1` and
the client IP used for rate limiting cannot be forged. `app.` and `api.` on the same registrable domain
keep the refresh cookie same-site. Proxying `/api` through Vercel instead would require trusting
Vercel's (unpublished) egress addresses in Caddy.

## Operations

- **Update**: `git pull && docker compose up -d --build` (migrations run on start).
- **Logs**: `docker compose logs -f <service>`. They are rotated at 5 × 10 MB per service.
- **Back up the volumes** `freellmapi_data` (provider keys and unified key) and, with local storage,
  `backend_storage` (uploaded files). Neon and Pinecone are managed.
- **Local validation** of this configuration (stand-ins for Neon, FreeLLMAPI and TLS):
  `compose.local-test.yml`.
