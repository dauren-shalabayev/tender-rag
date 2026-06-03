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

## API

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Проверка сервиса |
| POST | `/v1/kb/{kb_id}/documents` | Загрузить PDF/DOCX |
| GET | `/v1/kb/{kb_id}/documents` | Список документов |
| POST | `/v1/kb/{kb_id}/chat` | Вопрос по базе знаний |

Подробнее: [docs/kb-chat.md](docs/kb-chat.md)

## Пример

```bash
curl -X POST "http://127.0.0.1:8083/v1/kb/default/documents" \
  -F "file=@./manual.pdf"

curl -X POST "http://127.0.0.1:8083/v1/kb/default/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "О чём этот документ?"}'
```

Фронт на `localhost:8080` вызывает те же эндпоинты (CORS настроен в compose).
