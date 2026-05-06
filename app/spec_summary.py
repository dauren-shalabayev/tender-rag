"""Сжатая выжимка техспецификации тендера через OpenAI (JSON)."""

from __future__ import annotations

import json
import os
from typing import Any

from app.config import OPENAI_CHAT_MODEL

MAX_SPEC_CHARS = 48_000

SYSTEM_SPEC = """Ты аналитик по госзакупкам и техзаданиям (Казахстан, русский язык).
Дан полный или фрагментированный текст документации лота / ТЗ / спецификации.

Задача: выделить проверяемые факты из текста. Не выдумывай требования — только то, что явно следует из текста или разумно следует из формулировок. Если чего-то нет в тексте, честно отрази это в open_questions.

Ответ строго JSON (без markdown), формат:
{
  "overview": "2–4 предложения: предмет закупки и контекст",
  "key_requirements": ["важное требование 1", "..."],
  "deliverables": ["что нужно поставить/сделать — пунктами"],
  "terms_and_deadlines": ["сроки, этапы, расписание — если есть в тексте"],
  "constraints": ["ограничения: лицензии, стандарты, объём, территория и т.д."],
  "open_questions": ["что уточнить у заказчика или проверить в полном комплекте документов"]
}

Массивы могут быть пустыми [], строки — короткие и конкретные."""


def _truncate(s: str, limit: int) -> str:
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[: limit - 30] + "\n… [текст обрезан для модели]"


def summarize_specification(spec_text: str) -> dict[str, Any]:
    """Структурированная выжимка ТЗ. Нужен OPENAI_API_KEY."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY не задан")

    text = _truncate(spec_text, MAX_SPEC_CHARS)
    if not text:
        raise ValueError("Пустой текст спецификации")

    model = OPENAI_CHAT_MODEL.strip()
    user = "### Текст документа (ТЗ / спецификация / описание лота)\n\n" + text

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.15,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_SPEC},
            {"role": "user", "content": user},
        ],
    )
    choice = resp.choices[0].message.content
    if not choice:
        raise RuntimeError("Пустой ответ от модели")

    data = json.loads(choice)
    return _normalize_payload(data)


def _normalize_payload(data: Any) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise RuntimeError("Модель вернула не объект JSON")

    def str_list(key: str) -> list[str]:
        v = data.get(key)
        if v is None:
            return []
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    overview = str(data.get("overview", "")).strip() or "—"
    return {
        "overview": overview,
        "key_requirements": str_list("key_requirements"),
        "deliverables": str_list("deliverables"),
        "terms_and_deadlines": str_list("terms_and_deadlines"),
        "constraints": str_list("constraints"),
        "open_questions": str_list("open_questions"),
    }
