from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Json

from app.config import DATABASE_URL

logger = logging.getLogger(__name__)


def _schema_sql_path() -> Path:
    override = os.environ.get("SCHEMA_SQL_PATH", "").strip()
    if override:
        return Path(override)
    root = Path(__file__).resolve().parent.parent
    return root / "db" / "init.sql"


def _parse_init_sql(script: str) -> list[str]:
    lines: list[str] = []
    for line in script.splitlines():
        s = line.strip()
        if s.startswith("--") or not s:
            continue
        lines.append(line)
    bulk = "\n".join(lines)
    return [p.strip() + ";" for p in bulk.split(";") if p.strip()]


def apply_schema() -> None:
    """Создаёт расширение и таблицы из db/init.sql (идемпотентно). Без register_vector — до SETUP vector."""
    if os.environ.get("SKIP_SCHEMA_APPLY", "").strip() in ("1", "true", "yes"):
        logger.info("SKIP_SCHEMA_APPLY set, skipping schema apply")
        return
    path = _schema_sql_path()
    if not path.is_file():
        raise FileNotFoundError(f"schema SQL not found: {path}")
    sql = path.read_text(encoding="utf-8")
    statements = _parse_init_sql(sql)
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.transaction():
            for stmt in statements:
                conn.execute(stmt)
    logger.info("database schema applied from %s", path)


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


def replace_lot_spec_summary(conn, lot_id: str, payload: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO lot_spec_summaries (lot_id, payload, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (lot_id) DO UPDATE SET
              payload = EXCLUDED.payload,
              updated_at = NOW()
            """,
            (lot_id, Json(payload)),
        )
    conn.commit()


def get_lot_spec_summary(conn, lot_id: str) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT payload FROM lot_spec_summaries WHERE lot_id = %s",
            (lot_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    return row[0]


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


def health_db() -> tuple[bool, str | None]:
    try:
        conn = get_conn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        return True, None
    except Exception as e:
        logger.exception("database health check failed")
        return False, str(e)
