-- Старая схема (tender_chunks, lot_spec_summaries) -> kb_chunks
DO $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'tender_chunks'
  ) THEN
    INSERT INTO kb_chunks (doc_key, chunk_index, content, embedding, source_hint, created_at)
    SELECT lot_id, chunk_index, content, embedding, source_hint, created_at
    FROM tender_chunks
    ON CONFLICT (doc_key, chunk_index) DO UPDATE SET
      content = EXCLUDED.content,
      embedding = EXCLUDED.embedding,
      source_hint = EXCLUDED.source_hint,
      created_at = EXCLUDED.created_at;

    DROP TABLE tender_chunks;
  END IF;

  DROP TABLE IF EXISTS lot_spec_summaries;
END $$;
