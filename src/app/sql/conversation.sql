-- name: advisory_lock
SELECT pg_advisory_lock(hashtextextended(%s, 0))

-- name: get_summary_state
SELECT summary, summary_through_sequence FROM sessions WHERE session_id = %s

-- name: update_summary_cursor
UPDATE sessions SET summary_through_sequence = %s, updated_at = NOW()
WHERE session_id = %s

-- name: memory_exists
SELECT EXISTS (SELECT 1 FROM session_memories
               WHERE session_id = %s AND memory_key = %s)

-- name: store_memory
INSERT INTO session_memories
    (session_id, memory_key, memory_text, embedding, sequence_start,
     sequence_end, embedding_model)
VALUES (%s, %s, %s, %s::vector, %s, %s, %s)
ON CONFLICT (session_id, memory_key)
DO UPDATE SET memory_text = EXCLUDED.memory_text,
              embedding = EXCLUDED.embedding,
              sequence_start = EXCLUDED.sequence_start,
              sequence_end = EXCLUDED.sequence_end,
              embedding_model = EXCLUDED.embedding_model,
              updated_at = NOW()

-- name: initialize_vector_extension
CREATE EXTENSION IF NOT EXISTS vector

-- name: initialize_summary_column
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT ''

-- name: initialize_summary_cursor_column
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary_through_sequence BIGINT NOT NULL DEFAULT 0

-- name: initialize_memories_table
CREATE TABLE IF NOT EXISTS session_memories (
    memory_id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    memory_key TEXT NOT NULL,
    memory_text TEXT NOT NULL,
    embedding vector NOT NULL,
    sequence_start BIGINT NOT NULL,
    sequence_end BIGINT NOT NULL,
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, memory_key)
)

-- name: initialize_memory_session_index
CREATE INDEX IF NOT EXISTS idx_session_memories_session
ON session_memories (session_id)

-- name: search_memories
WITH scoped_memories AS MATERIALIZED (
    SELECT memory_text, sequence_start, sequence_end, embedding
    FROM session_memories
    WHERE session_id = %s AND embedding_model = %s
)
SELECT memory_text, sequence_start, sequence_end,
       1 - (embedding::vector({dimensions}) <=> %s::vector({dimensions})) AS similarity
FROM scoped_memories
WHERE 1 - (embedding::vector({dimensions}) <=> %s::vector({dimensions})) >= %s
ORDER BY embedding::vector({dimensions}) <=> %s::vector({dimensions})
LIMIT %s

-- name: initialize_memory_embedding_index
CREATE INDEX IF NOT EXISTS idx_session_memories_embedding
ON session_memories USING hnsw
((embedding::vector({dimensions})) vector_cosine_ops)

-- name: create_session
INSERT INTO sessions (title) VALUES (%s)
RETURNING session_id, title, status, created_at, updated_at

-- name: list_sessions
SELECT s.session_id, s.title, s.status, s.created_at, s.updated_at,
       count(m.message_id) AS message_count
FROM sessions s LEFT JOIN messages m ON m.session_id = s.session_id
GROUP BY s.session_id
ORDER BY s.updated_at DESC, s.session_id
LIMIT %s OFFSET %s

-- name: get_session
SELECT session_id, title, status, created_at, updated_at
FROM sessions WHERE session_id = %s

-- name: delete_session
DELETE FROM sessions WHERE session_id = %s

-- name: get_messages
SELECT message_id, sequence_no, role, content, tool_name,
       tool_call_id, tool_calls, metadata, created_at
FROM messages WHERE session_id = %s ORDER BY sequence_no

-- name: load_recent_history
WITH context_start AS (
    SELECT COALESCE((SELECT sequence_no FROM messages
        WHERE session_id = %s AND role = 'user'
        ORDER BY sequence_no DESC OFFSET %s LIMIT 1), 0) AS sequence_no
)
SELECT messages.message_id, messages.sequence_no, messages.role,
       messages.content, messages.tool_name, messages.tool_call_id,
       messages.tool_calls, messages.metadata, messages.created_at
FROM messages, context_start
WHERE session_id = %s AND messages.sequence_no >= context_start.sequence_no
ORDER BY messages.sequence_no

-- name: get_history_with_sequences
SELECT sequence_no, role, content, tool_calls
FROM messages WHERE session_id = %s 
    AND role IN ('user', 'assistant')
    AND content <> '[]'
ORDER BY sequence_no

-- name: lock_session_row
SELECT session_id FROM sessions WHERE session_id = %s FOR UPDATE

-- name: latest_message_sequence
SELECT COALESCE(MAX(sequence_no), 0) FROM messages WHERE session_id = %s

-- name: insert_message
INSERT INTO messages
    (session_id, sequence_no, role, content, tool_name,
     tool_call_id, tool_calls, metadata)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
RETURNING message_id

-- name: touch_session
UPDATE sessions SET updated_at = NOW() WHERE session_id = %s
