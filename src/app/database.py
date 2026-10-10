"""Persistent conversation history and summary repository."""

import json
import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any
from uuid import UUID

import psycopg2
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from psycopg2.extras import Json, RealDictCursor, register_uuid

register_uuid()

class ConversationRepository:
    """Read and write conversation messages and rolling summaries."""

    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg2.extensions.connection:
        return psycopg2.connect(self._database_url, connect_timeout=5)

    @contextmanager
    def _connection(self) -> Iterator[psycopg2.extensions.connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @asynccontextmanager
    async def conversation_lock(self, conversation_id: UUID) -> AsyncIterator[None]:
        """Serialize concurrent requests for one conversation across workers."""
        connection = await asyncio.to_thread(self._acquire_lock, conversation_id)
        try:
            yield
        finally:
            await asyncio.shield(asyncio.to_thread(connection.close))

    def _acquire_lock(self, conversation_id: UUID) -> psycopg2.extensions.connection:
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
                    (str(conversation_id),),
                )
            return connection
        except Exception:
            connection.close()
            raise

    def initialize(self) -> None:
        """Ensure sessions have a place to store their rolling summary."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT ''"
            )
            cursor.execute(
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary_through_sequence BIGINT NOT NULL DEFAULT 0"
            )

    def get_summary_state(self, conversation_id: UUID) -> tuple[str, int]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT summary, summary_through_sequence FROM sessions WHERE session_id = %s",
                (conversation_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise LookupError("Conversation not found.")
            return row[0] or "", int(row[1])

    def update_summary(self, conversation_id: UUID, summary: str, through_sequence: int) -> None:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "UPDATE sessions SET summary = %s, summary_through_sequence = %s, "
                "updated_at = NOW() WHERE session_id = %s",
                (summary, through_sequence, conversation_id),
            )

    def create_conversation(self, title: str | None) -> dict[str, Any]:
        with self._connection() as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "INSERT INTO sessions (title) VALUES (%s) "
                "RETURNING session_id, title, status, created_at, updated_at",
                (title.strip() if title and title.strip() else None,),
            )
            return dict(cursor.fetchone())

    def list_conversations(self, limit: int, offset: int) -> list[dict[str, Any]]:
        with self._connection() as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """SELECT s.session_id, s.title, s.status, s.created_at, s.updated_at,
                          count(m.message_id) AS message_count
                   FROM sessions s LEFT JOIN messages m ON m.session_id = s.session_id
                   GROUP BY s.session_id
                   ORDER BY s.updated_at DESC, s.session_id
                   LIMIT %s OFFSET %s""",
                (limit, offset),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_conversation(self, conversation_id: UUID) -> dict[str, Any] | None:
        with self._connection() as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT session_id, title, status, created_at, updated_at "
                "FROM sessions WHERE session_id = %s",
                (conversation_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_conversation(self, conversation_id: UUID) -> bool:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM sessions WHERE session_id = %s", (conversation_id,))
            return cursor.rowcount > 0

    def get_messages(self, conversation_id: UUID) -> list[dict[str, Any]]:
        with self._connection() as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """SELECT message_id, sequence_no, role, content, tool_name,
                          tool_call_id, tool_calls, metadata, created_at
                   FROM messages WHERE session_id = %s
                   ORDER BY sequence_no""",
                (conversation_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def load_recent_history(
        self,
        conversation_id: UUID,
        context_turns: int,
    ) -> list[BaseMessage]:
        with self._connection() as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """WITH context_start AS (
                       SELECT COALESCE((
                           SELECT sequence_no FROM messages
                           WHERE session_id = %s AND role = 'user'
                           ORDER BY sequence_no DESC OFFSET %s LIMIT 1
                       ), 0) AS sequence_no
                   )
                   SELECT messages.message_id, messages.sequence_no, messages.role,
                          messages.content, messages.tool_name, messages.tool_call_id,
                          messages.tool_calls, messages.metadata, messages.created_at
                   FROM messages, context_start
                   WHERE session_id = %s AND messages.sequence_no >= context_start.sequence_no
                   ORDER BY messages.sequence_no""",
                (conversation_id, context_turns - 1, conversation_id),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        history = [self._to_langchain_message(row) for row in rows]
        return history

    def load_history_with_sequences(
        self,
        conversation_id: UUID,
    ) -> list[tuple[int, BaseMessage]]:
        with self._connection() as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """SELECT sequence_no, role, content, tool_name, tool_call_id, tool_calls
                   FROM messages WHERE session_id = %s ORDER BY sequence_no""",
                (conversation_id,),
            )
            rows = [dict(row) for row in cursor.fetchall()]
        return [
            (int(row["sequence_no"]), self._to_langchain_message(row))
            for row in rows
        ]

    def append_user_message(self, conversation_id: UUID, content: str) -> int:
        return self.append_messages(conversation_id, [HumanMessage(content=content)])[0]

    def append_messages(
        self,
        conversation_id: UUID,
        messages: list[BaseMessage],
    ) -> list[int]:
        """Append LangChain messages atomically and return their database IDs."""
        inserted_ids: list[int] = []
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT session_id FROM sessions WHERE session_id = %s FOR UPDATE",
                (conversation_id,),
            )
            if cursor.fetchone() is None:
                raise LookupError("Conversation not found.")
            cursor.execute(
                "SELECT COALESCE(MAX(sequence_no), 0) FROM messages WHERE session_id = %s",
                (conversation_id,),
            )
            sequence = int(cursor.fetchone()[0])
            for message in messages:
                converted = self._from_langchain_message(message)
                if converted is None:
                    continue
                role, content, tool_name, tool_call_id, tool_calls = converted
                sequence += 1
                cursor.execute(
                    """INSERT INTO messages
                       (session_id, sequence_no, role, content, tool_name,
                        tool_call_id, tool_calls)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)
                       RETURNING message_id""",
                    (conversation_id, sequence, role, content, tool_name,
                     tool_call_id, Json(tool_calls) if tool_calls is not None else None),
                )
                inserted_ids.append(int(cursor.fetchone()[0]))
            cursor.execute(
                "UPDATE sessions SET updated_at = NOW() WHERE session_id = %s",
                (conversation_id,),
            )
        return inserted_ids

    @staticmethod
    def _message_content(content: object) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, default=str)

    @classmethod
    def _from_langchain_message(
        cls,
        message: BaseMessage,
    ) -> tuple[str, str, str | None, str | None, list[dict[str, Any]] | None] | None:
        if isinstance(message, HumanMessage):
            return "user", cls._message_content(message.content), None, None, None
        if isinstance(message, AIMessage):
            return (
                "assistant", cls._message_content(message.content), None, None,
                message.tool_calls or None,
            )
        if isinstance(message, ToolMessage):
            return (
                "tool", cls._message_content(message.content), message.name,
                message.tool_call_id, None,
            )
        return None

    @staticmethod
    def _to_langchain_message(row: dict[str, Any]) -> BaseMessage:
        content = row["content"] or ""
        if row["role"] == "user":
            return HumanMessage(content=content)
        if row["role"] == "assistant":
            return AIMessage(content=content, tool_calls=row["tool_calls"] or [])
        if row["role"] == "tool":
            return ToolMessage(
                content=content,
                tool_call_id=row["tool_call_id"],
                name=row["tool_name"],
            )
        raise ValueError(f"Unsupported stored message role: {row['role']}")
