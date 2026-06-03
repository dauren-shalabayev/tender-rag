from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.chunking import chunk_text
from app.config import CORS_ORIGINS
from app.document_extract import extract_text_from_bytes
from app.embeddings import embed_chunks, embed_query
from app.kb_chat import answer_kb_question
from app.store import (
    apply_schema,
    get_conn,
    health_db,
    kb_document_key,
    list_kb_documents,
    replace_document_chunks,
    search_kb_chunks,
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    apply_schema()
    yield


app = FastAPI(title="Knowledge Base RAG", version="0.2.0", lifespan=lifespan)

_cors_wildcard = CORS_ORIGINS == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=not _cors_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class KbChatBody(BaseModel):
    message: str = Field(..., description="Вопрос пользователя")
    history: list[ChatHistoryMessage] = Field(default_factory=list)
    top_chunks: int = Field(12, ge=1, le=40)


class KbChatSource(BaseModel):
    document_id: str
    score: float
    excerpt: str


class KbChatResponse(BaseModel):
    answer: str
    sources: list[KbChatSource]


class KbDocumentInfo(BaseModel):
    document_id: str
    chunk_count: int
    updated_at: str | None = None


def sanitize_document_id(name: str) -> str:
    base = Path(name).stem or "document"
    s = re.sub(r"[^\w.\-]+", "_", base, flags=re.UNICODE)
    return (s[:120] or "document").strip("_") or "document"


@app.get("/health")
def health() -> dict[str, Any]:
    db_ok, db_err = health_db()
    _key = os.environ.get("OPENAI_API_KEY", "")
    out: dict[str, Any] = {
        "ok": True,
        "database": db_ok,
        "openai_configured": bool(_key.strip()),
    }
    if db_err is not None:
        out["database_error"] = db_err
    return out


@app.get("/v1/kb/{kb_id}/documents", response_model=list[KbDocumentInfo])
def kb_list_documents(kb_id: str) -> list[KbDocumentInfo]:
    conn = get_conn()
    try:
        rows = list_kb_documents(conn, kb_id)
    finally:
        conn.close()
    return [KbDocumentInfo(**r) for r in rows]


@app.post("/v1/kb/{kb_id}/documents")
async def kb_upload_document(
    kb_id: str,
    file: Annotated[UploadFile, File(description="PDF или DOCX")],
    document_id: Annotated[
        str | None,
        Form(description="Идентификатор документа; по умолчанию — имя файла"),
    ] = None,
    source_hint: Annotated[str | None, Form()] = None,
) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="пустой файл")
    name = file.filename or "document.pdf"
    doc_id = (document_id or "").strip() or sanitize_document_id(name)
    try:
        doc_key = kb_document_key(kb_id, doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    try:
        text = extract_text_from_bytes(name, raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="не удалось извлечь текст (пустой файл или скан без OCR)",
        )

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="no chunks after normalization")
    vectors = embed_chunks(chunks)
    conn = get_conn()
    try:
        replace_document_chunks(conn, doc_key, chunks, vectors, source_hint)
    finally:
        conn.close()

    return JSONResponse(
        {
            "indexed": True,
            "kb_id": kb_id.strip().strip("/"),
            "document_id": doc_id,
            "chunks": len(chunks),
            "text_chars": len(text),
        },
        status_code=200,
    )


@app.post("/v1/kb/{kb_id}/chat", response_model=KbChatResponse)
def kb_chat(kb_id: str, body: KbChatBody) -> KbChatResponse:
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY не задан — добавьте ключ в .env",
        )
    msg = body.message.strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message пуст")

    q = embed_query(msg)
    conn = get_conn()
    try:
        hits = search_kb_chunks(conn, kb_id, q, limit=body.top_chunks)
    finally:
        conn.close()

    history = [h.model_dump() for h in body.history]
    try:
        answer = answer_kb_question(msg, hits, history=history)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI: {e!s}") from e

    sources: list[KbChatSource] = []
    seen: set[str] = set()
    for h in hits:
        doc = str(h.get("document_id", ""))
        if doc in seen:
            continue
        seen.add(doc)
        sources.append(
            KbChatSource(
                document_id=doc,
                score=float(h.get("score", 0)),
                excerpt=(h.get("content") or "")[:500],
            )
        )

    return KbChatResponse(answer=answer, sources=sources)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "knowledge-base-rag",
        "docs": "GET /v1/kb/{kb_id}/documents",
        "upload": "POST /v1/kb/{kb_id}/documents",
        "chat": "POST /v1/kb/{kb_id}/chat",
    }
