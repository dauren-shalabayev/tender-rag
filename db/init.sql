-- База знаний: чанки документов с эмбеддингами (pgvector)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS kb_chunks (
  id BIGSERIAL PRIMARY KEY,
  doc_key TEXT NOT NULL,
  chunk_index INT NOT NULL,
  content TEXT NOT NULL,
  embedding vector(384) NOT NULL,
  source_hint TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (doc_key, chunk_index)
);

CREATE INDEX IF NOT EXISTS kb_chunks_doc_key_idx ON kb_chunks (doc_key);

CREATE INDEX IF NOT EXISTS kb_chunks_embedding_hnsw
  ON kb_chunks USING hnsw (embedding vector_cosine_ops);
