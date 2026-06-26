#!/usr/bin/env python3
"""Загрузить страницы из Confluence в базу знаний."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.confluence_ingest import ingest_confluence, load_confluence_config


def main() -> None:
    kb_id = os.environ.get("KB_ID", "confluence")
    try:
        config = load_confluence_config()
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    print(f"Подключение к Confluence: {config.url} (space={config.space})")
    result = ingest_confluence(kb_id, space=config.space)
    print(f"Найдено {result.pages_total} страниц")

    for doc in result.indexed:
        print(
            f"  ✓ {doc.title} → kb={result.kb_id}, "
            f"document_id={doc.document_id}, чанков={doc.chunks}"
        )
    for skip in result.skipped:
        print(f"  пропуск {skip.title}: {skip.reason}")
    for err in result.errors:
        print(f"  ✗ {err.title}: {err.error}", file=sys.stderr)

    if result.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
