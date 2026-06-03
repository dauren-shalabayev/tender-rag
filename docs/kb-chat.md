# База знаний и чат

API: **`http://127.0.0.1:8083`**  
Фронт: **`http://localhost:8080`**

`kb_id` — имя базы (`default`, `company`, …). В одной базе несколько документов.

## Загрузить документ

```http
POST /v1/kb/{kb_id}/documents
Content-Type: multipart/form-data
```

| Поле | Описание |
|------|----------|
| `file` | PDF или DOCX |
| `document_id` | опционально, иначе из имени файла |

```bash
curl -X POST "http://127.0.0.1:8083/v1/kb/default/documents" \
  -F "file=@manual.pdf"
```

## Список документов

```http
GET /v1/kb/{kb_id}/documents
```

## Чат

```http
POST /v1/kb/{kb_id}/chat
Content-Type: application/json
```

```json
{
  "message": "Как настроить интеграцию?",
  "history": [
    { "role": "user", "content": "Привет" },
    { "role": "assistant", "content": "Задайте вопрос по документам." }
  ],
  "top_chunks": 12
}
```

Ответ: `answer`, `sources[]` (`document_id`, `score`, `excerpt`).

## Фронт (fetch)

```javascript
const API = "http://127.0.0.1:8083";
const KB_ID = "default";

export async function uploadDoc(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API}/v1/kb/${KB_ID}/documents`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function sendChat(message, history = []) {
  const res = await fetch(`${API}/v1/kb/${KB_ID}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history, top_chunks: 12 }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
```
