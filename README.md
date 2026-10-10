# Vessel Assistant

FastAPI chat API for vessel data with Gemini tool calling, durable PostgreSQL
conversation history, and rolling conversation summaries.

## Requirements

- Python 3.13 or newer
- Docker Compose
- A Gemini API key
- The supplied CSV files in `data/`

## Configure and run

1. Copy `.env.example` to `.env`, set `GEMINI_API_KEY` and `GEMINI_MODEL`, then
   adjust the PostgreSQL password if needed.
2. Start PostgreSQL with PostGIS:

   ```sh
   docker compose up -d --build postgres
   ```

3. Install dependencies and load the supplied data:

   ```sh
   uv sync
   uv run python scripts/load_csv.py --data-dir data
   ```

4. Start the API:

   ```sh
   uv run uvicorn src.app.main:app --reload
   ```

Interactive API documentation is available at `http://localhost:8000/docs`.

`CHAT_CONTEXT_TURNS` sets how many recent user turns are sent verbatim. When
older turns accumulate, Gemini updates a concise summary stored on that
conversation; all original messages remain in PostgreSQL.

## Conversation API

Create a conversation:

```sh
curl.exe -X POST http://localhost:8000/conversations \
  -H "Content-Type: application/json" \
  -d '{"title":"Vessel investigation"}'
```

Use the returned `session_id` as `{id}` below. Read `token`, `tool_call`,
`done`, and `error` events as they arrive:

```sh
curl.exe -N -X POST http://localhost:8000/conversations/{id}/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Cho tôi thông tin về tàu KOTA GAYA."}'
```

Other endpoints:

- `GET /conversations` lists conversations.
- `GET /conversations/{id}/messages` reads persisted user, assistant, and tool messages.
- `DELETE /conversations/{id}` deletes the conversation and its associated history and summary.

Conversation messages and summaries are scoped by `session_id`. Requests for
the same conversation are serialized with a PostgreSQL advisory lock, while
separate conversations can run concurrently.
