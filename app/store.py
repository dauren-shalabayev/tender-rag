from __future__ import annotations

from collections import defaultdict
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from app.config import DATABASE_URL


def get_conn():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def replace_lot_chunks(
    conn,
    lot_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    source_hint: str | None,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")
    if not chunks:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tender_chunks WHERE lot_id = %s", (lot_id,))
        conn.commit()
        return

    with conn.cursor() as cur:
        cur.execute("DELETE FROM tender_chunks WHERE lot_id = %s", (lot_id,))
        for idx, (content, emb) in enumerate(zip(chunks, embeddings, strict=True)):
            cur.execute(
                """
                INSERT INTO tender_chunks (lot_id, chunk_index, content, embedding, source_hint)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (lot_id, idx, content, emb, source_hint),
            )
    conn.commit()


def match_profile(
    conn,
    query_embedding: list[float],
    *,
    top_lots: int = 10,
    chunk_hits: int = 60,
    snippets_per_lot: int = 3,
) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT lot_id, content,
                   (1 - (embedding <=> %(q)s::vector))::float AS score
            FROM tender_chunks
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(lim)s
            """,
            {"q": query_embedding, "lim": chunk_hits},
        )
        rows = cur.fetchall()

    by_lot: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for lot_id, content, score in rows:
        by_lot[lot_id].append((float(score), content))

    ranked: list[dict[str, Any]] = []
    for lot_id, items in by_lot.items():
        items.sort(key=lambda x: x[0], reverse=True)
        best = items[0][0]
        snippets = [p[1] for p in items[:snippets_per_lot]]
        ranked.append({"lot_id": lot_id, "score": best, "snippets": snippets})

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_lots]


def health_db() -> bool:
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        return True
    except Exception:
        return False
