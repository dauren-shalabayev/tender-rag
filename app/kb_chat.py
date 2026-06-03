"""RAG-чат по базе знаний: поиск чанков + ответ OpenAI."""

from __future__ import annotations

import os
from typing import Any

from app.config import OPENAI_CHAT_MODEL

MAX_CONTEXT_CHARS = 20_000
MAX_HISTORY_TURNS = 10

SYSTEM_PROMPT = """Ты ассистент по базе знаний.
Отвечай на вопрос пользователя, опираясь ТОЛЬКО на переданные фрагменты документов.
Если в фрагментах нет ответа — честно скажи, что в загруженных документах этого нет; не выдумывай.
Отвечай на том же языке, что и вопрос (если вопрос на русском — отвечай по-русски).
Будь конкретным и кратким, при необходимости перечисляй пункты."""


def _truncate(s: str, limit: int) -> str:
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[: limit - 20] + "\n… [обрезано]"


def build_context_block(chunks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        doc = c.get("document_id") or "?"
        score = c.get("score", 0)
        text = _truncate(str(c.get("content", "")), 4000)
        lines.append(f"[{i}] документ: {doc} (релевантность {score:.3f})")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def answer_kb_question(
    question: str,
    chunks: list[dict[str, Any]],
    *,
    history: list[dict[str, str]] | None = None,
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY не задан")

    question = question.strip()
    if not question:
        raise ValueError("message пуст")

    context = build_context_block(chunks)
    if not context:
        context = "(В базе знаний пока нет проиндексированных фрагментов.)"

    user_block = (
        "### Фрагменты из базы знаний\n"
        + _truncate(context, MAX_CONTEXT_CHARS)
        + "\n\n### Вопрос\n"
        + question
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if history:
        for h in history[-MAX_HISTORY_TURNS:]:
            role = (h.get("role") or "").strip().lower()
            content = (h.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_block})

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL.strip(),
        temperature=0.3,
        messages=messages,
    )
    choice = resp.choices[0].message.content
    if not choice:
        raise RuntimeError("Пустой ответ от модели")
    return choice.strip()
