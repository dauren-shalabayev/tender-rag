-- pgvector + chunks tied to lot_id (from your lots microservice or local ingest)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tender_chunks (
  id BIGSERIAL PRIMARY KEY,
  lot_id TEXT NOT NULL,
  chunk_index INT NOT NULL,
  content TEXT NOT NULL,
  embedding vector(384) NOT NULL,
  source_hint TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (lot_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS tender_chunks_lot_id_idx ON tender_chunks (lot_id);

-- Approximate NN index (ok to create on empty table in pgvector 0.5+)
CREATE INDEX IF NOT EXISTS tender_chunks_embedding_hnsw
  ON tender_chunks USING hnsw (embedding vector_cosine_ops);
