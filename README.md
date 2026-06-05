# Knowledge Base RAG

Сервис базы знаний: загрузка PDF/DOCX и чат с ответами по документам (pgvector + OpenAI).

API: **`http://127.0.0.1:8083`**

## Запуск

```bash
docker compose up --build
```

Скопируйте `.env.example` → `.env` и укажите `OPENAI_API_KEY`.

При старте API автоматически применяются `db/init.sql` и миграции из `db/migrations/` (таблица `schema_migrations`).

Миграция `001_migrate_tender_to_kb.sql` переносит данные из старой таблицы `tender_chunks` в `kb_chunks`. Если миграция не помогла — сброс тома:

```bash
docker compose down -v
docker compose up --build
```

## Эндпоинты

Базовый URL: **`http://127.0.0.1:8083`**

`kb_id` — имя базы знаний (`default`, `company`, …). В одной базе может быть несколько документов.

Swagger: `http://127.0.0.1:8083/docs`

| Метод | Путь | OpenAI | Описание |
|-------|------|:------:|----------|
| GET | `/` | нет | Краткая справка по API |
| GET | `/health` | нет | Проверка сервиса и БД |
| POST | `/v1/kb/{kb_id}/documents` | нет | Загрузить PDF/DOCX |
| GET | `/v1/kb/{kb_id}/documents` | нет | Список документов в базе |
| POST | `/v1/kb/{kb_id}/chat` | да | Вопрос по базе знаний (RAG) |

---

### `GET /health`

```bash
curl http://127.0.0.1:8083/health
```

Ответ:

```json
{
  "ok": true,
  "database": true,
  "openai_configured": true
}
```

---

### `POST /v1/kb/{kb_id}/documents` — загрузка файла

`Content-Type: multipart/form-data`

| Поле | Обязательно | Описание |
|------|:-----------:|----------|
| `file` | да | PDF или DOCX |
| `document_id` | нет | ID документа; по умолчанию — имя файла без расширения |
| `source_hint` | нет | Произвольный тег |

```bash
curl -X POST "http://127.0.0.1:8083/v1/kb/default/documents" \
  -F "file=@./manual.pdf"
```

Ответ:

```json
{
  "indexed": true,
  "kb_id": "default",
  "document_id": "manual",
  "chunks": 42,
  "text_chars": 15000
}
```

Повторная загрузка с тем же `document_id` перезаписывает только этот документ.

---

### `GET /v1/kb/{kb_id}/documents` — список документов

```bash
curl http://127.0.0.1:8083/v1/kb/default/documents
```

Ответ:

```json
[
  {
    "document_id": "manual",
    "chunk_count": 42,
    "updated_at": "2026-06-03T14:00:00+00:00"
  }
]
```

---

### `POST /v1/kb/{kb_id}/chat` — запрос в чат

`Content-Type: application/json`

Тело:

```json
{
  "message": "О чём этот документ?",
  "history": [
    { "role": "user", "content": "Привет" },
    { "role": "assistant", "content": "Задайте вопрос по документам." }
  ],
  "top_chunks": 12
}
```

| Поле | Обязательно | Описание |
|------|:-----------:|----------|
| `message` | да | Вопрос пользователя |
| `history` | нет | История диалога (`user` / `assistant`) |
| `top_chunks` | нет | Сколько фрагментов подставить в контекст (1–40, по умолчанию 12) |

```bash
curl -X POST "http://127.0.0.1:8083/v1/kb/default/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "О чём этот документ?"}'
```

Ответ:

```json
{
  "answer": "Документ описывает …",
  "sources": [
    {
      "document_id": "manual",
      "score": 0.85,
      "excerpt": "фрагмент из документа…"
    }
  ]
}
```

Нужен `OPENAI_API_KEY` в `.env`.

---

## Фронт

Фронт на `localhost:8080` вызывает те же URL на `:8083` (CORS настроен в `docker-compose.yml`).

Подробнее: [docs/kb-chat.md](docs/kb-chat.md)
