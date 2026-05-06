from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from app.chunking import chunk_text
from app.config import CORS_ORIGINS, get_company_profile
from app.document_extract import extract_text_from_bytes
from app.embeddings import embed_chunks, embed_profile
from app.openai_analyze import analyze_lot_without_index, analyze_match_context
from app.spec_summary import summarize_specification
from app.store import (
    apply_schema,
    get_conn,
    get_lot_spec_summary,
    health_db,
    match_profile,
    replace_lot_chunks,
    replace_lot_spec_summary,
)


def effective_profile(inline: str | None) -> str:
    p = (inline or "").strip()
    if p:
        return p
    d = get_company_profile()
    if not d:
        raise HTTPException(
            status_code=400,
            detail="Нет профиля: передайте поле profile в JSON или задайте COMPANY_PROFILE / COMPANY_PROFILE_FILE в .env",
        )
    return d


@asynccontextmanager
async def lifespan(_app: FastAPI):
    apply_schema()
    yield


app = FastAPI(title="Tender RAG", version="0.1.0", lifespan=lifespan)

_cors_wildcard = CORS_ORIGINS == ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=not _cors_wildcard,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SpecSummaryOut(BaseModel):
    model_config = ConfigDict(extra="ignore")

    overview: str = ""
    key_requirements: list[str] = Field(default_factory=list)
    deliverables: list[str] = Field(default_factory=list)
    terms_and_deadlines: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class IndexBody(BaseModel):
    text: str = Field(..., description="Full tender text to index (ТЗ, описание лота)")
    source_hint: str | None = Field(
        None, description="Optional tag: api version, document id, etc."
    )
    extract_spec_points: bool = Field(
        False,
        description="Извлечь основные пункты ТЗ через OpenAI и сохранить (нужен OPENAI_API_KEY)",
    )


class MatchBody(BaseModel):
    profile: str | None = Field(
        None,
        description="Кто вы и что умеете. Если пусто — берётся из .env (COMPANY_PROFILE или COMPANY_PROFILE_FILE)",
    )
    top_lots: int = Field(10, ge=1, le=100)
    chunk_hits: int = Field(60, ge=10, le=500)
    snippets_per_lot: int = Field(3, ge=1, le=10)


class MatchResult(BaseModel):
    lot_id: str
    score: float
    snippets: list[str]


class AnalyzeBody(BaseModel):
    profile: str | None = Field(
        None,
        description="Профиль компании; если пусто — из .env (COMPANY_PROFILE / COMPANY_PROFILE_FILE)",
    )
    incoming: str | None = Field(
        None,
        description="Что пришло с вашего API (JSON или текст про лот) — опционально",
    )
    top_lots: int = Field(10, ge=1, le=50)
    chunk_hits: int = Field(60, ge=10, le=500)
    snippets_per_lot: int = Field(3, ge=1, le=10)


class LotVerdict(BaseModel):
    lot_id: str
    fit: str = Field(..., description="подходит | сомнительно | не подходит")
    reason: str = Field(..., description="Коротко: почему так")


class AnalyzeResponse(BaseModel):
    summary: str = Field(..., description="Общий вывод 1–2 предложения")
    verdicts: list[LotVerdict] = Field(
        ...,
        description="По каждому лоту: подходит или нет и короткое обоснование",
    )
    matches: list[MatchResult]


class LotAnalyzeBody(BaseModel):
    profile: str | None = Field(
        None,
        description="Профиль компании; если пусто — из COMPANY_PROFILE / файла в .env",
    )
    lot_text: str = Field(
        ...,
        description="Текст лота целиком (или JSON строкой) — без предварительного /index",
    )


class LotAnalyzeResponse(BaseModel):
    summary: str
    fit: str
    reason: str
    checks: str | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    db_ok, db_err = health_db()
    _key = os.environ.get("OPENAI_API_KEY", "")
    out: dict[str, Any] = {
        "ok": True,
        "database": db_ok,
        "openai_configured": bool(_key.strip()),
        "openai_env": {
            "OPENAI_API_KEY_defined": "OPENAI_API_KEY" in os.environ,
            "OPENAI_API_KEY_length": len(_key),
        },
    }
    if db_err is not None:
        out["database_error"] = db_err
    return out


@app.post("/v1/lots/{lot_id}/index")
def index_lot(lot_id: str, body: IndexBody) -> Response:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty")
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="no chunks after normalization")
    vectors = embed_chunks(chunks)
    conn = get_conn()
    try:
        replace_lot_chunks(conn, lot_id, chunks, vectors, body.source_hint)
        if body.extract_spec_points:
            if not os.environ.get("OPENAI_API_KEY", "").strip():
                raise HTTPException(
                    status_code=503,
                    detail="extract_spec_points требует OPENAI_API_KEY в окружении",
                )
            try:
                payload = summarize_specification(text)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"OpenAI (spec summary): {e!s}",
                ) from e
            replace_lot_spec_summary(conn, lot_id, payload)
            return JSONResponse(
                {"indexed": True, "spec_summary": payload},
                status_code=200,
            )
    finally:
        conn.close()
    return Response(status_code=204)


@app.post("/v1/lots/{lot_id}/index-document")
async def index_document(
    lot_id: str,
    file: Annotated[UploadFile, File(description="PDF или DOCX со спецификацией / ТЗ")],
    source_hint: Annotated[
        str | None,
        Form(description="Опционально: тег источника"),
    ] = None,
    extract_spec_points: Annotated[
        bool,
        Form(
            description="Выжимка ТЗ через OpenAI (тратит токены; нужен OPENAI_API_KEY)"
        ),
    ] = False,
    include_extracted_text: Annotated[
        bool,
        Form(description="Вернуть полный извлечённый текст в JSON"),
    ] = True,
) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="пустой файл")
    name = file.filename or "document"
    try:
        text = extract_text_from_bytes(name, raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="не удалось извлечь текст из файла (пустой или только сканы без OCR)",
        )

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="no chunks after normalization")
    vectors = embed_chunks(chunks)
    conn = get_conn()
    try:
        replace_lot_chunks(conn, lot_id, chunks, vectors, source_hint)
        out: dict[str, Any] = {"indexed": True, "text_chars": len(text)}
        if include_extracted_text:
            out["extracted_text"] = text
        if extract_spec_points:
            if not os.environ.get("OPENAI_API_KEY", "").strip():
                raise HTTPException(
                    status_code=503,
                    detail="extract_spec_points требует OPENAI_API_KEY в окружении",
                )
            try:
                payload = summarize_specification(text)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            except Exception as e:
                raise HTTPException(
                    status_code=502,
                    detail=f"OpenAI (spec summary): {e!s}",
                ) from e
            replace_lot_spec_summary(conn, lot_id, payload)
            out["spec_summary"] = payload
    finally:
        conn.close()
    return JSONResponse(out, status_code=200)


@app.get("/v1/lots/{lot_id}/spec-summary", response_model=SpecSummaryOut)
def get_spec_summary(lot_id: str) -> SpecSummaryOut:
    conn = get_conn()
    try:
        row = get_lot_spec_summary(conn, lot_id)
    finally:
        conn.close()
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Выжимка не найдена: сначала индексируйте лот с extract_spec_points или POST index-document",
        )
    return SpecSummaryOut.model_validate(row)


@app.post("/v1/match", response_model=list[MatchResult])
def match(body: MatchBody) -> list[MatchResult]:
    profile = effective_profile(body.profile)
    q = embed_profile(profile)
    conn = get_conn()
    try:
        rows = match_profile(
            conn,
            q,
            top_lots=body.top_lots,
            chunk_hits=body.chunk_hits,
            snippets_per_lot=body.snippets_per_lot,
        )
    finally:
        conn.close()
    return [MatchResult(**r) for r in rows]


@app.post("/v1/match/analyze", response_model=AnalyzeResponse)
def match_analyze(body: AnalyzeBody) -> AnalyzeResponse:
    """Векторный поиск + текстовый разбор OpenAI по профилю, входящим данным и фрагментам из БД."""
    profile = effective_profile(body.profile)
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY не задан — добавьте ключ в окружение",
        )

    q = embed_profile(profile)
    conn = get_conn()
    try:
        rows = match_profile(
            conn,
            q,
            top_lots=body.top_lots,
            chunk_hits=body.chunk_hits,
            snippets_per_lot=body.snippets_per_lot,
        )
    finally:
        conn.close()

    matches = [MatchResult(**r) for r in rows]
    try:
        out = analyze_match_context(
            profile,
            [m.model_dump() for m in matches],
            body.incoming,
        )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI: {e!s}",
        ) from e

    verdicts_raw = out.get("verdicts") or []
    verdicts: list[LotVerdict] = []
    for v in verdicts_raw:
        if not isinstance(v, dict):
            continue
        verdicts.append(
            LotVerdict(
                lot_id=str(v.get("lot_id", "")).strip(),
                fit=str(v.get("fit", "сомнительно")).strip() or "сомнительно",
                reason=str(v.get("reason", "")).strip()
                or "Смотрите фрагменты в matches.",
            )
        )
    if not verdicts and matches:
        verdicts = [
            LotVerdict(
                lot_id=m.lot_id,
                fit="сомнительно",
                reason="Модель не вернула verdicts — ориентируйтесь на score и snippets.",
            )
            for m in matches
        ]

    return AnalyzeResponse(
        summary=out["summary"],
        verdicts=verdicts,
        matches=matches,
    )


@app.post("/v1/lot/analyze", response_model=LotAnalyzeResponse)
def lot_analyze(body: LotAnalyzeBody) -> LotAnalyzeResponse:
    """Один лот: профиль + текст лота → вердикт. Без pgvector и без POST /index (нужен OPENAI_API_KEY)."""
    if not os.environ.get("OPENAI_API_KEY", "").strip():
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY не задан",
        )
    profile = effective_profile(body.profile)
    try:
        out = analyze_lot_without_index(profile, body.lot_text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"OpenAI: {e!s}") from e
    return LotAnalyzeResponse(**out)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "tender-rag",
        "index": "POST /v1/lots/{lot_id}/index (поле extract_spec_points для выжимки ТЗ)",
        "index_document": "POST /v1/lots/{lot_id}/index-document — PDF/DOCX → индекс + опционально выжимка",
        "spec_summary": "GET /v1/lots/{lot_id}/spec-summary",
        "match": "POST /v1/match",
        "match_analyze": "POST /v1/match/analyze (нужен OPENAI_API_KEY)",
        "lot_analyze": "POST /v1/lot/analyze — лот без индекса, только OpenAI",
        "profile_default": "COMPANY_PROFILE или COMPANY_PROFILE_FILE в .env если не передаёте profile",
    }
