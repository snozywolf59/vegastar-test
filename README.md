# Vessel Assistant

FastAPI chat API for vessel data with Gemini tool calling, durable PostgreSQL
conversation history, and pgvector-backed semantic conversation memory.

## Requirements

- Python 3.13 or newer
- Docker Compose
- A Gemini API key
- The supplied CSV files in `data/`

## Configure and run

1. Copy `.env.example` to `.env`, set `GEMINI_API_KEY`, `GEMINI_MODEL`, and
   `GEMINI_EMBEDDING_MODEL`, then adjust the PostgreSQL password if needed.
2. Build and start PostgreSQL, load CSV data once, and start the backend:

   ```sh
   docker compose up -d --build
   ```

   The `loader` service runs `scripts/load_csv.py` against the directory set by
   `DATA_PATH` before the
   API starts. It skips reloading when any destination table already contains
   data, so normal restarts do not truncate and reload the dataset.

The chat UI is available at `http://localhost:8000`, and interactive API
documentation is available at `http://localhost:8000/docs`. Set `DATA_PATH` in
`.env` to the directory containing all four CSV files; relative paths are
resolved from the Compose project directory.

`CHAT_CONTEXT_TURNS` sets how many recent user turns are sent verbatim. Older
turns are summarized in batches, embedded, and stored in the conversation's
`session_memories` rows. The agent can call `search_conversation_memory` when a
follow-up needs older context; retrieved memories are always scoped to that
conversation. Original messages remain in PostgreSQL.
Set `GEMINI_EMBEDDING_MODEL` to a Gemini embedding model identifier.
`MEMORY_TOP_K` and `MEMORY_MIN_SIMILARITY` control retrieval size and relevance.

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
