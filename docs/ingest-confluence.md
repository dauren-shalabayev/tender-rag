# Загрузка данных из Confluence

Скрипт `scripts/ingest_confluence.py` скачивает страницы из Confluence Cloud, конвертирует HTML в текст, режет на чанки, считает эмбеддинги и сохраняет в PostgreSQL (pgvector). После этого страницы доступны в чате RAG.

API: **`http://127.0.0.1:8083`**

## Что нужно заранее

1. Запущенный сервис: `docker compose up --build`
2. Файл `.env` в корне репозитория (переменные подхватываются контейнером `api` через `env_file`)
3. API-токен Confluence (не пароль от аккаунта)

## Переменные окружения

Добавьте в `.env`:

```env
CONFLUENCE_URL=https://your-domain.atlassian.net
CONFLUENCE_USER=you@example.com
CONFLUENCE_TOKEN=ATATT...
CONFLUENCE_SPACE=TEAM
KB_ID=default
```

| Переменная | Описание |
|------------|----------|
| `CONFLUENCE_URL` | URL инстанса без слэша в конце, например `https://company.atlassian.net` |
| `CONFLUENCE_USER` | Email учётной записи Atlassian |
| `CONFLUENCE_TOKEN` | API-токен из [Atlassian Account Security](https://id.atlassian.com/manage-profile/security/api-tokens) |
| `CONFLUENCE_SPACE` | Ключ space (виден в URL: `/wiki/spaces/MFS/...` → `MFS`) |
| `KB_ID` | Имя базы знаний в RAG (`default`, `confluence`, …). По умолчанию в скрипте — `confluence`, если переменная не задана |

`OPENAI_API_KEY` для ingest **не нужен** — используется только локальная embedding-модель (`EMBEDDING_MODEL`).

## Как получить API-токен

1. Войдите на [id.atlassian.com](https://id.atlassian.com/manage-profile/security/api-tokens)
2. **Create API token** → скопируйте значение
3. Вставьте в `CONFLUENCE_TOKEN` в `.env`

Авторизация: HTTP Basic Auth (`email` + `api_token`).

## Загрузка через API (с фронта)

Загрузка выполняется **в фоне** (может занять несколько минут). Сразу возвращается `202 Accepted`, статус опрашивается отдельным запросом.

### Запустить ingest

```http
POST /v1/kb/{kb_id}/ingest/confluence
Content-Type: application/json
```

Тело (опционально):

```json
{
  "space": "MFS"
}
```

Если `space` не передан — берётся `CONFLUENCE_SPACE` из `.env`.

Ответ `202`:

```json
{
  "status": "started",
  "kb_id": "default",
  "space": "MFS"
}
```

Ошибки:
- `503` — не заданы переменные Confluence в `.env`
- `409` — загрузка уже выполняется

### Статус ingest

```http
GET /v1/kb/{kb_id}/ingest/confluence/status
```

Пока идёт загрузка:

```json
{
  "status": "running",
  "kb_id": "default",
  "space": "MFS",
  "started_at": "2026-06-26T06:00:00+00:00",
  "finished_at": null,
  "result": null,
  "error": null
}
```

После завершения:

```json
{
  "status": "completed",
  "kb_id": "default",
  "space": "MFS",
  "started_at": "...",
  "finished_at": "...",
  "result": {
    "pages_total": 5,
    "indexed_count": 5,
    "skipped_count": 0,
    "error_count": 0,
    "indexed": [
      {
        "document_id": "Владельцы_процессов",
        "title": "Владельцы процессов",
        "page_id": "66055",
        "chunks": 1
      }
    ]
  },
  "error": null
}
```

### Пример с фронта (fetch)

```javascript
const API = "http://127.0.0.1:8083";
const KB_ID = "default";

export async function ingestConfluence(space) {
  const res = await fetch(`${API}/v1/kb/${KB_ID}/ingest/confluence`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(space ? { space } : {}),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function pollIngestStatus() {
  const res = await fetch(`${API}/v1/kb/${KB_ID}/ingest/confluence/status`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Запуск + опрос каждые 3 сек
export async function syncConfluence() {
  await ingestConfluence();
  for (;;) {
    const s = await pollIngestStatus();
    if (s.status === "completed") return s.result;
    if (s.status === "failed") throw new Error(s.error || "ingest failed");
    await new Promise((r) => setTimeout(r, 3000));
  }
}
```

`GET /health` возвращает `confluence_configured: true/false` — можно проверить до нажатия кнопки.

## Запуск через CLI (скрипт)

В отдельном терминале, пока работает `docker compose up`:

```bash
docker compose exec api python scripts/ingest_confluence.py
```

Ожидаемый вывод:

```
Подключение к Confluence: https://....atlassian.net (space=MFS)
Найдено 5 страниц
Загрузка: Название страницы (ID: 12345)...
  ✓ kb=default, document_id=Название_страницы, чанков=3
...
```

Первый запуск может занять несколько минут: скачивается embedding-модель `intfloat/multilingual-e5-small`.

### Если скрипта нет в контейнере

Пересоберите образ (скрипт копируется при `docker build`):

```bash
docker compose up --build
```

## Как это работает

1. Скрипт запрашивает все страницы (`type=page`) из указанного space через Confluence REST API
2. HTML тела страницы конвертируется в простой текст
3. Текст режется на чанки (`app/chunking.py`)
4. Для каждого чанка считается вектор (`app/embeddings.py`)
5. Чанки сохраняются в таблицу `kb_chunks` с ключом `{KB_ID}/{document_id}`

`document_id` формируется из заголовка страницы (спецсимволы → `_`).

`source_hint` для каждого документа: `confluence://{page_id}`.

## Проверка

Список проиндексированных документов:

```bash
curl http://127.0.0.1:8083/v1/kb/default/documents
```

Подставьте свой `KB_ID`, если задавали другой.

Тест чата (нужен `OPENAI_API_KEY`):

```bash
curl -X POST "http://127.0.0.1:8083/v1/kb/default/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "О чём страница про владельцев процессов?"}'
```

## Повторная загрузка

Повторный запуск **перезаписывает** документ с тем же `document_id` (тот же заголовок после санитизации). Новые страницы в space добавляются, старые обновляются.

Чтобы подтянуть изменения в Confluence, просто снова выполните:

```bash
docker compose exec api python scripts/ingest_confluence.py
```

## Очистка перед повторной загрузкой

Удалить всю базу знаний `default`:

```bash
docker compose exec db psql -U rag -d rag -c \
  "DELETE FROM kb_chunks WHERE doc_key = 'default' OR doc_key LIKE 'default/%';"
```

Удалить один документ:

```bash
docker compose exec db psql -U rag -d rag -c \
  "DELETE FROM kb_chunks WHERE doc_key = 'default/Имя_документа';"
```

## Ограничения

- Загружаются только **страницы** (`type=page`), не вложения, не комментарии
- HTML конвертируется в простой текст без таблиц и сложной вёрстки
- Страницы без извлекаемого текста пропускаются
- Одновременно может выполняться только одна фоновая загрузка

## Частые ошибки

| Симптом | Причина | Решение |
|---------|---------|---------|
| `Требуемые переменные окружения` | Не заданы `CONFLUENCE_*` | Проверьте `.env`, перезапустите `docker compose up` |
| `401 Unauthorized` | Неверный email или токен | Пересоздайте API-токен, проверьте `CONFLUENCE_USER` |
| `404` на `/wiki/rest/api/...` | Неверный `CONFLUENCE_URL` | URL должен быть вида `https://xxx.atlassian.net` без `/wiki` |
| `can't open file ... ingest_confluence.py` | Старый образ Docker | `docker compose up --build` |
| Пустой список страниц | Неверный `CONFLUENCE_SPACE` | Проверьте ключ space в URL Confluence |
| `409` при POST ingest | Уже идёт загрузка | Дождитесь `status: completed` или перезапустите API |

## См. также

- [kb-chat.md](kb-chat.md) — загрузка PDF/DOCX и чат с фронта
- [README.md](../README.md) — общий запуск сервиса
