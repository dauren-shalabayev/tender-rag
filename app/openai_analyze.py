"""Анализ профиля и найденных фрагментов через OpenAI Chat (структурированный ответ)."""

from __future__ import annotations

import json
import os
from typing import Any

from app.config import OPENAI_CHAT_MODEL

MAX_CONTEXT_CHARS = 24_000

SYSTEM_PROMPT = """Ты помощник по госзакупкам и ИТ-услугам (Казахстан).
На входе:
1) Профиль компании — что умеют и чем занимаются.
2) Опционально сырые данные с API о лоте.
3) Фрагменты из базы (чанки ТЗ), уже отобранные по семантической близости; у каждого лота есть lot_id и score.

Правила:
- Сопоставь профиль с содержанием фрагментов по каждому lot_id. Не придумывай факты — только то, что следует из текста.
- Не гарантируй победу или допуск к участию.

Ответ строго JSON (без markdown, без пояснений вне JSON) формата:
{
  "summary": "1–2 предложения: общий вывод по списку",
  "verdicts": [
    {
      "lot_id": "строка — точно как во входных данных",
      "fit": "подходит" | "сомнительно" | "не подходит",
      "reason": "1–2 коротких предложения: почему так (только по переданным фрагментам)"
    }
  ]
}

Для КАЖДОГО lot_id из списка совпадений должен быть ровно один объект в verdicts (в том же порядке или в произвольном, но все lot_id покрыты)."""


def _truncate(s: str, limit: int) -> str:
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[: limit - 20] + "\n… [обрезано]"


def build_user_prompt(
    profile: str,
    matches: list[dict[str, Any]],
    incoming: str | None,
) -> str:
    parts: list[str] = []
    parts.append("### Профиль компании\n" + profile.strip())
    if incoming and incoming.strip():
        parts.append(
            "### Данные с API / про лот (опционально)\n"
            + _truncate(incoming.strip(), 8000)
        )
    lines = ["### Совпадения из базы (lot_id, score, фрагменты)", ""]
    for m in matches:
        lid = m.get("lot_id", "")
        sc = m.get("score", 0)
        lines.append(f"lot_id: {lid}")
        lines.append(f"score: {sc:.4f}")
        for sn in m.get("snippets") or []:
            lines.append(_truncate(sn, 3500))
            lines.append("")
        lines.append("---")
    blob = "\n".join(lines)
    combined = "\n\n".join(parts) + "\n\n" + blob
    return _truncate(combined, MAX_CONTEXT_CHARS)


def analyze_match_context(
    profile: str,
    matches: list[dict[str, Any]],
    incoming: str | None = None,
) -> dict[str, Any]:
    """Возвращает dict с ключами summary (str) и verdicts (list of dict)."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY не задан")

    model = OPENAI_CHAT_MODEL.strip()
    user_prompt = build_user_prompt(profile, matches, incoming)
    lot_ids = [str(m.get("lot_id", "")) for m in matches]
    user_prompt += (
        "\n\n### Список lot_id для verdicts (все должны попасть в JSON):\n"
        + json.dumps(lot_ids, ensure_ascii=False)
    )

    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Установите пакет openai") from e

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    choice = resp.choices[0].message.content
    if not choice:
        raise RuntimeError("Пустой ответ от модели")

    try:
        data = json.loads(choice)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Модель вернула невалидный JSON: {e}") from e

    summary = (data.get("summary") or "").strip()
    verdicts_raw = data.get("verdicts")
    if not isinstance(verdicts_raw, list):
        raise RuntimeError("В ответе нет массива verdicts")

    verdicts: list[dict[str, str]] = []
    for v in verdicts_raw:
        if not isinstance(v, dict):
            continue
        verdicts.append(
            {
                "lot_id": str(v.get("lot_id", "")).strip(),
                "fit": str(v.get("fit", "сомнительно")).strip(),
                "reason": str(v.get("reason", "")).strip(),
            }
        )

    if not summary:
        summary = "Краткий вывод недоступен — проверьте фрагменты вручную."

    return {"summary": summary, "verdicts": verdicts}


SYSTEM_LOT_ONLY = """Ты помощник по госзакупкам и ИТ (Казахстан).
Даны: профиль компании и текст лота (описание, ТЗ, JSON — как есть).
Оцени, насколько лот соответствует возможностям компании ТОЛЬКО по этим текстам.
Не выдумывай факты. Не гарантируй победу или допуск.

Ответ строго JSON:
{
  "summary": "1–2 предложения общий вывод",
  "fit": "подходит" | "сомнительно" | "не подходит",
  "reason": "2–4 предложения: почему так",
  "checks": "что специалисту проверить в полных документах (короткий список через точку с запятой)"
}"""


def analyze_lot_without_index(company_profile: str, lot_text: str) -> dict[str, str]:
    """Разбор одного лота без векторного индекса (только OpenAI)."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY не задан")

    model = OPENAI_CHAT_MODEL.strip()
    lot_text = _truncate(lot_text.strip(), 18_000)
    if not lot_text:
        raise ValueError("lot_text пуст")

    user = (
        "### Профиль компании\n"
        + company_profile.strip()
        + "\n\n### Текст лота\n"
        + lot_text
    )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_LOT_ONLY},
            {"role": "user", "content": user},
        ],
    )
    choice = resp.choices[0].message.content
    if not choice:
        raise RuntimeError("Пустой ответ от модели")
    data = json.loads(choice)
    return {
        "summary": str(data.get("summary", "")).strip() or "—",
        "fit": str(data.get("fit", "сомнительно")).strip() or "сомнительно",
        "reason": str(data.get("reason", "")).strip() or "—",
        "checks": str(data.get("checks", "")).strip() or None,
    }
