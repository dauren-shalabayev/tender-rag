#!/usr/bin/env python3
"""Загрузить PDF/DOCX из папки files в базу знаний (kb_id=default)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docx import Document
from pypdf import PdfReader

from app.chunking import chunk_text
from app.embeddings import embed_chunks
from app.store import get_conn, kb_document_key, replace_document_chunks

FILES_DIR = Path(os.environ.get("FILES_DIR", ROOT / "files"))
KB_ID = os.environ.get("KB_ID", "default")


def extract_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def extract_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def document_id_from_file(path: Path) -> str:
    safe = re.sub(r"[^\w\-.]+", "_", path.stem)
    return safe or "document"


def main() -> None:
    if not FILES_DIR.is_dir():
        print(f"Нет папки: {FILES_DIR}", file=sys.stderr)
        sys.exit(1)

    conn = get_conn()
    try:
        for path in sorted(FILES_DIR.iterdir()):
            if path.suffix.lower() not in {".pdf", ".docx"}:
                continue
            print(f"Загрузка {path.name} ...")
            text = extract_pdf(path) if path.suffix.lower() == ".pdf" else extract_docx(path)
            text = text.strip()
            if not text:
                print(f"  пропуск (пустой текст): {path.name}")
                continue
            doc_id = document_id_from_file(path)
            doc_key = kb_document_key(KB_ID, doc_id)
            chunks = chunk_text(text)
            if not chunks:
                continue
            vectors = embed_chunks(chunks)
            replace_document_chunks(
                conn,
                doc_key,
                chunks,
                vectors,
                source_hint=path.name,
            )
            print(f"  kb={KB_ID} document_id={doc_id}, чанков={len(chunks)}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
