# Tender RAG

Семантический поиск по текстам лотов (pgvector) и опциональный разбор через OpenAI. API по умолчанию: **`http://127.0.0.1:8083`**.

## Запуск

```bash
docker compose up --build
```

Переменные окружения — см. `.env.example` (`OPENAI_*`, `COMPANY_PROFILE` / `COMPANY_PROFILE_FILE` для профиля по умолчанию).

## Проверка

```bash
curl -s http://127.0.0.1:8083/health
```

## Документация по сценариям

Оглавление отдельных страниц: [docs/README.md](docs/README.md).  
Краткая сводка **2 / 3 / 4** одним файлом: [README.variants.md](README.variants.md).

| Документ | Что внутри |
|----------|------------|
| [docs/indexing.md](docs/indexing.md) | **`POST /v1/lots/{lot_id}/index`** — залить лот в базу (нужно для вариантов 2 и 4) |
| [docs/match.md](docs/match.md) | **Вариант 2** — `POST /v1/match`: только поиск по базе, без OpenAI |
| [docs/lot-analyze.md](docs/lot-analyze.md) | **Вариант 3** — `POST /v1/lot/analyze`: один лот по тексту, вердикт «подходит / нет» |
| [docs/match-analyze.md](docs/match-analyze.md) | **Вариант 4** — `POST /v1/match/analyze`: топ по базе + разбор OpenAI |

## Сводка

Перед **2** и **4** для новых лотов нужен **`POST /v1/lots/{lot_id}/index`**. **3** индекс не требует.

| | Нужен `/index` | OpenAI | Один лот текстом в запросе |
|--|:--:|:--:|:--|
| **2** `match` | да | нет | нет |
| **3** `lot/analyze` | нет | да | да (`lot_text`) |
| **4** `match/analyze` | да для попадания в топ из БД | да | опционально (`incoming`) |

**Итог:** **2** — умный поиск по базе; **3** — разбор одного лота без базы (карточка тендера); **4** — топ по базе и словесный разбор.
