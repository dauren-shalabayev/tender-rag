"""Загрузка страниц Confluence в базу знаний."""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Literal

import requests
from requests.auth import HTTPBasicAuth

from app.chunking import chunk_text
from app.embeddings import embed_chunks
from app.store import get_conn, kb_document_key, replace_document_chunks

logger = logging.getLogger(__name__)


class HTMLToText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text:
            self.text.append(text)

    def get_text(self) -> str:
        return "\n".join(self.text)


def html_to_text(html: str) -> str:
    parser = HTMLToText()
    try:
        parser.feed(html)
        return parser.get_text()
    except Exception:
        return ""


def sanitize_doc_id(title: str) -> str:
    safe = re.sub(r"[^\w\-.]+", "_", title)
    return (safe[:120] or "document").strip("_") or "document"


def page_storage_text(page: dict[str, Any]) -> str:
    html = page.get("body", {}).get("storage", {}).get("value", "")
    return html_to_text(html)


@dataclass(frozen=True)
class ConfluenceConfig:
    url: str
    user: str
    token: str
    space: str


def confluence_configured() -> bool:
    try:
        load_confluence_config()
        return True
    except ValueError:
        return False


def load_confluence_config(*, space: str | None = None) -> ConfluenceConfig:
    url = os.environ.get("CONFLUENCE_URL", "").strip()
    user = os.environ.get("CONFLUENCE_USER", "").strip()
    token = os.environ.get("CONFLUENCE_TOKEN", "").strip()
    space_key = (space or os.environ.get("CONFLUENCE_SPACE", "")).strip()
    if not (url and user and token and space_key):
        raise ValueError(
            "Задайте CONFLUENCE_URL, CONFLUENCE_USER, CONFLUENCE_TOKEN "
            "и CONFLUENCE_SPACE (или передайте space в запросе)"
        )
    return ConfluenceConfig(url=url, user=user, token=token, space=space_key)


class ConfluenceClient:
    def __init__(self, config: ConfluenceConfig) -> None:
        self.base_url = config.url.rstrip("/")
        self.auth = HTTPBasicAuth(config.user, config.token)
        self.session = requests.Session()

    def get_pages(self, space_key: str, limit: int = 50) -> list[dict[str, Any]]:
        url = f"{self.base_url}/wiki/rest/api/content"
        params: dict[str, Any] = {
            "spaceKey": space_key,
            "type": "page",
            "limit": limit,
            "expand": "body.storage,version",
        }
        pages: list[dict[str, Any]] = []
        start = 0
        while True:
            params["start"] = start
            response = self.session.get(
                url, auth=self.auth, params=params, timeout=30
            )
            response.raise_for_status()
            data = response.json()
            pages.extend(data.get("results", []))
            if not data.get("_links", {}).get("next"):
                break
            start += limit
        return pages


@dataclass
class IngestedDocument:
    document_id: str
    title: str
    page_id: str
    chunks: int


@dataclass
class SkippedPage:
    title: str
    page_id: str
    reason: str


@dataclass
class IngestError:
    title: str
    page_id: str
    error: str


@dataclass
class IngestResult:
    kb_id: str
    space: str
    confluence_url: str
    pages_total: int
    indexed: list[IngestedDocument] = field(default_factory=list)
    skipped: list[SkippedPage] = field(default_factory=list)
    errors: list[IngestError] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kb_id": self.kb_id,
            "space": self.space,
            "confluence_url": self.confluence_url,
            "pages_total": self.pages_total,
            "indexed_count": len(self.indexed),
            "skipped_count": len(self.skipped),
            "error_count": len(self.errors),
            "indexed": [
                {
                    "document_id": d.document_id,
                    "title": d.title,
                    "page_id": d.page_id,
                    "chunks": d.chunks,
                }
                for d in self.indexed
            ],
            "skipped": [
                {"title": s.title, "page_id": s.page_id, "reason": s.reason}
                for s in self.skipped
            ],
            "errors": [
                {"title": e.title, "page_id": e.page_id, "error": e.error}
                for e in self.errors
            ],
        }


def ingest_confluence(kb_id: str, *, space: str | None = None) -> IngestResult:
    config = load_confluence_config(space=space)
    client = ConfluenceClient(config)
    pages = client.get_pages(config.space)

    result = IngestResult(
        kb_id=kb_id.strip().strip("/"),
        space=config.space,
        confluence_url=config.url,
        pages_total=len(pages),
    )

    conn = get_conn()
    try:
        for page in pages:
            page_id = str(page.get("id", ""))
            title = str(page.get("title", "unknown"))
            doc_id = sanitize_doc_id(title)
            try:
                text = page_storage_text(page).strip()
                if not text:
                    result.skipped.append(
                        SkippedPage(title=title, page_id=page_id, reason="пустой текст")
                    )
                    continue

                chunks = chunk_text(text)
                if not chunks:
                    result.skipped.append(
                        SkippedPage(title=title, page_id=page_id, reason="нет чанков")
                    )
                    continue

                doc_key = kb_document_key(result.kb_id, doc_id)
                vectors = embed_chunks(chunks)
                replace_document_chunks(
                    conn,
                    doc_key,
                    chunks,
                    vectors,
                    source_hint=f"confluence://{page_id}",
                )
                result.indexed.append(
                    IngestedDocument(
                        document_id=doc_id,
                        title=title,
                        page_id=page_id,
                        chunks=len(chunks),
                    )
                )
            except Exception as e:
                logger.exception("confluence ingest failed for page %s", page_id)
                result.errors.append(
                    IngestError(title=title, page_id=page_id, error=str(e))
                )
    finally:
        conn.close()

    return result


@dataclass
class IngestJobState:
    status: Literal["idle", "running", "completed", "failed"] = "idle"
    kb_id: str | None = None
    space: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


_lock = threading.Lock()
_state = IngestJobState()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_ingest_state() -> IngestJobState:
    with _lock:
        return IngestJobState(
            status=_state.status,
            kb_id=_state.kb_id,
            space=_state.space,
            started_at=_state.started_at,
            finished_at=_state.finished_at,
            result=_state.result,
            error=_state.error,
        )


def try_start_ingest(kb_id: str, space: str) -> bool:
    with _lock:
        if _state.status == "running":
            return False
        _state.status = "running"
        _state.kb_id = kb_id
        _state.space = space
        _state.started_at = _utc_now()
        _state.finished_at = None
        _state.result = None
        _state.error = None
        return True


def finish_ingest_success(result: IngestResult) -> None:
    with _lock:
        _state.status = "completed"
        _state.finished_at = _utc_now()
        _state.result = result.to_dict()
        _state.error = None


def finish_ingest_failure(error: str) -> None:
    with _lock:
        _state.status = "failed"
        _state.finished_at = _utc_now()
        _state.error = error
        _state.result = None


def run_ingest_job(kb_id: str, *, space: str | None = None) -> None:
    try:
        config = load_confluence_config(space=space)
        result = ingest_confluence(kb_id, space=config.space)
        finish_ingest_success(result)
    except Exception as e:
        logger.exception("confluence ingest job failed")
        finish_ingest_failure(str(e))
