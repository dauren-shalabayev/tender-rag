from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.chunking import chunk_text
from app.config import get_company_profile
from app.embeddings import embed_chunks, embed_profile
from app.openai_analyze import analyze_lot_without_index, analyze_match_context
from app.store import apply_schema, get_conn, health_db, match_profile, replace_lot_chunks


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


class IndexBody(BaseModel):
    text: str = Field(..., description="Full tender text to index (ТЗ, описание лота)")
    source_hint: str | None = Field(
        None, description="Optional tag: api version, document id, etc."
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
    out: dict[str, Any] = {"ok": True, "database": db_ok}
    if db_err is not None:
        out["database_error"] = db_err
    return out


@app.post("/v1/lots/{lot_id}/index", status_code=204)
def index_lot(lot_id: str, body: IndexBody) -> None:
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
    finally:
        conn.close()


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
        "index": "POST /v1/lots/{lot_id}/index",
        "match": "POST /v1/match",
        "match_analyze": "POST /v1/match/analyze (нужен OPENAI_API_KEY)",
        "lot_analyze": "POST /v1/lot/analyze — лот без индекса, только OpenAI",
        "profile_default": "COMPANY_PROFILE или COMPANY_PROFILE_FILE в .env если не передаёте profile",
    }
