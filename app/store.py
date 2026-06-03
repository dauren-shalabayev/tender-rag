from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from app.config import DATABASE_URL

logger = logging.getLogger(__name__)


def _db_root() -> Path:
    override = os.environ.get("DB_ROOT", "").strip()
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "db"


def _schema_sql_path() -> Path:
    override = os.environ.get("SCHEMA_SQL_PATH", "").strip()
    if override:
        return Path(override)
    return _db_root() / "init.sql"


def _migrations_dir() -> Path:
    return _db_root() / "migrations"


def _parse_init_sql(script: str) -> list[str]:
    lines: list[str] = []
    for line in script.splitlines():
        s = line.strip()
        if s.startswith("--") or not s:
            continue
        lines.append(line)
    bulk = "\n".join(lines)
    return [p.strip() + ";" for p in bulk.split(";") if p.strip()]


def _apply_migrations(conn: psycopg.Connection) -> None:
    mig_dir = _migrations_dir()
    if not mig_dir.is_dir():
        return
    for path in sorted(mig_dir.glob("*.sql")):
        version = path.name
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM schema_migrations WHERE version = %s",
                (version,),
            )
            if cur.fetchone():
                continue
            sql = path.read_text(encoding="utf-8").strip()
            if sql:
                cur.execute(sql)
            cur.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)",
                (version,),
            )
        logger.info("migration applied: %s", version)


def apply_schema() -> None:
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
            _apply_migrations(conn)
    logger.info("database schema applied from %s", path)


def get_conn():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def replace_document_chunks(
    conn,
    doc_key: str,
    chunks: list[str],
    embeddings: list[list[float]],
    source_hint: str | None,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")
    if not chunks:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM kb_chunks WHERE doc_key = %s", (doc_key,))
        conn.commit()
        return

    with conn.cursor() as cur:
        cur.execute("DELETE FROM kb_chunks WHERE doc_key = %s", (doc_key,))
        for idx, (content, emb) in enumerate(zip(chunks, embeddings, strict=True)):
            cur.execute(
                """
                INSERT INTO kb_chunks (doc_key, chunk_index, content, embedding, source_hint)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (doc_key, idx, content, emb, source_hint),
            )
    conn.commit()


def kb_document_key(kb_id: str, document_id: str) -> str:
    kb = kb_id.strip().strip("/")
    doc = document_id.strip().strip("/")
    if not kb:
        raise ValueError("kb_id пуст")
    if not doc:
        raise ValueError("document_id пуст")
    if "/" in doc:
        raise ValueError("document_id не должен содержать /")
    return f"{kb}/{doc}"


def document_id_from_key(kb_id: str, doc_key: str) -> str:
    kb = kb_id.strip().strip("/")
    if doc_key == kb:
        return kb
    prefix = f"{kb}/"
    if doc_key.startswith(prefix):
        return doc_key[len(prefix) :]
    return doc_key


def search_kb_chunks(
    conn,
    kb_id: str,
    query_embedding: list[float],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    kb = kb_id.strip().strip("/")
    prefix = f"{kb}/"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT doc_key, content,
                   (1 - (embedding <=> %(q)s::vector))::float AS score
            FROM kb_chunks
            WHERE doc_key = %(exact)s OR doc_key LIKE %(pfx)s
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(lim)s
            """,
            {"q": query_embedding, "exact": kb, "pfx": prefix + "%", "lim": limit},
        )
        rows = cur.fetchall()

    return [
        {
            "document_id": document_id_from_key(kb, doc_key),
            "content": content,
            "score": float(score),
        }
        for doc_key, content, score in rows
    ]


def list_kb_documents(conn, kb_id: str) -> list[dict[str, Any]]:
    kb = kb_id.strip().strip("/")
    prefix = f"{kb}/"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT doc_key,
                   COUNT(*)::int AS chunk_count,
                   MAX(created_at) AS updated_at
            FROM kb_chunks
            WHERE doc_key = %(exact)s OR doc_key LIKE %(pfx)s
            GROUP BY doc_key
            ORDER BY MAX(created_at) DESC
            """,
            {"exact": kb, "pfx": prefix + "%"},
        )
        rows = cur.fetchall()

    return [
        {
            "document_id": document_id_from_key(kb, doc_key),
            "chunk_count": chunk_count,
            "updated_at": updated_at.isoformat() if updated_at else None,
        }
        for doc_key, chunk_count, updated_at in rows
    ]


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
