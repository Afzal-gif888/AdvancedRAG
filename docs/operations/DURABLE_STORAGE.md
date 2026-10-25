# Enabling durable file storage (operator guide)

Status: **prepared, not applied.** Nothing in this repository has changed production settings, deployed, or migrated data.
`render.yaml` still declares `STORAGE_BACKEND=local` with `ALLOW_LOCAL_STORAGE_IN_PRODUCTION=true`, i.e. uploaded originals
live on Render's ephemeral disk and disappear when the service restarts or sleeps.

## What the risk is, precisely

- **Search keeps working** after a restart: chunk text is in the RAG database and vectors are in Pinecone. Questions about already-indexed documents are unaffected.
- **Lost:** the original file. *Download*, *reprocess* of a failed document, and *replace*-by-reprocess of that document's old bytes need it. A document whose file is gone fails with the explicit, non-retryable error `FILE_MISSING` (not a mystery "could not be found").
- A document remembers the backend that holds its bytes (`documents.storage_backend`). Switching the setting affects **new uploads only**; existing rows keep pointing at `local`, and their files stay lost if the disk was already wiped. Nothing is migrated automatically.

## Choose a backend

| | `s3` (R2 / S3 / MinIO) | `db` (backend PostgreSQL) |
|---|---|---|
| Extra infrastructure | a bucket and an API token | none |
| Where bytes live | outside the database | `stored_objects` / `stored_object_parts` rows, committed in the same transaction as the document and job |
| Capacity | bucket quota | counts against the database quota (Neon free is about 0.5 GB per project) |
| Backups | bucket versioning/replication (you configure it) | covered by the database backup |
| Recommended for | anything beyond a small volume | a small deployment that wants durability with no new service |

Per-user caps (`MAX_STORAGE_BYTES_PER_USER`, `MAX_DOCUMENTS_PER_USER`) and `MAX_UPLOAD_BYTES` (25 MiB) apply to both.

## Steps (the same for either backend)

1. **Back up first.** Take a Neon backup/branch of the `advancerag` database (provider feature; not tested by this repository). For `s3`, create the bucket and enable versioning.
2. **Validate the change offline** (no network, no secrets):
   `python scripts/verify_production_config.py --overlay infrastructure/durable-storage/backend-db.overlay.yaml`
   (or `backend-s3.overlay.yaml`). It must print `[OK]` for all four services.
3. **Set the variables on `advancerag-backend` only** (Render dashboard > Environment, or a reviewed commit to `render.yaml`; note that `render.yaml` has `autoDeployTrigger: commit`, so a commit deploys):
   - `db`: `STORAGE_BACKEND=db`, and remove `ALLOW_LOCAL_STORAGE_IN_PRODUCTION`.
   - `s3`: `STORAGE_BACKEND=s3`, `STORAGE_BUCKET`, `STORAGE_ACCESS_KEY`, `STORAGE_SECRET_KEY`, `STORAGE_ENDPOINT` (R2: `https://<account-id>.r2.cloudflarestorage.com`), `STORAGE_REGION` (`auto` for R2), and remove `ALLOW_LOCAL_STORAGE_IN_PRODUCTION`. Keep the keys in the dashboard; never commit them. Use a token scoped to this one bucket.
4. **Do not change anything in RAG Core, Agent Core or the LLM service.** The embedding model, Pinecone index `advancerag-bge384`, dimension and metric are untouched by this change.
5. **Preflight against the real target** before sending users to it. In a shell that has the backend's environment (for example Render's shell for the backend service):
   `cd backend && python ../scripts/check_storage.py`
   Expected: `write/read/verify: OK` and `cleanup: preflight object removed`. For `db`, at least one user must exist. The probe writes one ~300 KB object under a key no document can have, reads it back, checks the hash and deletes it; it touches nothing else and never prints a secret.
6. **Deploy, then verify end to end** with a throw-away account: upload a small file, wait for `ready`, restart the backend service from the dashboard, then download the file and click *reprocess* path tests (e.g. upload a second file, restart, ask a question about it). The restart must not produce `FILE_MISSING` for the new documents.
7. **Old documents.** Anything uploaded before the switch keeps `storage_backend=local`. If its file still exists it keeps working; if not, the owner re-uploads or uses *replace* with a new file. Search over their already-indexed text is unaffected. There is no automatic migration, because the lost bytes cannot be recovered by any tool.

## Rollback

Set `STORAGE_BACKEND` back to its previous value. Documents created while the new backend was active keep their recorded backend, so the application must still be able to read it: **for `s3`, keep the credentials configured; for `db`, nothing is needed.** Removing the credentials after files were written to the bucket would make those documents unreadable (`STORAGE_UNAVAILABLE`/`FILE_MISSING`).

## Limits of the evidence

- Verified locally: both backends round-trip in the test suite (`s3` against `moto`, `db` against SQLite and PostgreSQL), integrity (sha256) is checked on read, legacy base64 jobs still work, the preflight tool, and that the configuration passes the services' production fail-fast rules with placeholder secrets.
- **Not verified:** a real S3/R2 bucket, Neon quotas and backup/restore, behaviour of Render restarts, throughput of the `db` backend for large files (a 25 MiB file is ~25 parts of 1 MiB in one transaction), and object-store backup/replication (not part of this repository).
