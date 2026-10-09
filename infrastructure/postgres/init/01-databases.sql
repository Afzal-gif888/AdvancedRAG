-- Extra databases created on first start of the docker PostgreSQL volume.
-- advancerag      : backend (created by POSTGRES_DB)
-- advancerag_rag  : RAG Core (chunks, keyword index)
-- *_test          : automated test runs (TEST_DATABASE_URL)
CREATE DATABASE advancerag_rag OWNER advancerag;
CREATE DATABASE advancerag_test OWNER advancerag;
CREATE DATABASE advancerag_rag_test OWNER advancerag;
