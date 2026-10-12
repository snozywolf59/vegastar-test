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
from src.app.resources import load_sql_section

register_uuid()

class ConversationRepository:
    """Read and write conversation messages and vector memories."""

    def __init__(self, database_url: str, embedding_dimensions: int) -> None:
        self._database_url = database_url
        self._embedding_dimensions = embedding_dimensions

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
                cursor.execute(load_sql_section("conversation.sql", "advisory_lock"), (str(conversation_id),))
            return connection
        except Exception:
            connection.close()
            raise

    def initialize(self) -> None:
        """Ensure vector memory storage and its similarity index exist."""
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(load_sql_section("conversation.sql", "initialize_vector_extension"))
            cursor.execute(load_sql_section("conversation.sql", "initialize_summary_column"))
            cursor.execute(load_sql_section("conversation.sql", "initialize_summary_cursor_column"))
            cursor.execute(load_sql_section("conversation.sql", "initialize_memories_table"))
            cursor.execute(
                load_sql_section("conversation.sql", "initialize_memory_embedding_index").format(
                    dimensions=self._embedding_dimensions
                )
            )
            cursor.execute(
                load_sql_section("conversation.sql", "initialize_memory_session_index")
            )

    def get_summary_state(self, conversation_id: UUID) -> tuple[str, int]:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                load_sql_section("conversation.sql", "get_summary_state"),
                (conversation_id,),
            )
            row = cursor.fetchone()
            if row is None:
                raise LookupError("Conversation not found.")
            return row[0] or "", int(row[1])

    def update_summary_cursor(self, conversation_id: UUID, through_sequence: int) -> None:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                load_sql_section("conversation.sql", "update_summary_cursor"),
                (through_sequence, conversation_id),
            )

    def memory_exists(self, conversation_id: UUID, memory_key: str) -> bool:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                load_sql_section("conversation.sql", "memory_exists"),
                (conversation_id, memory_key),
            )
            return bool(cursor.fetchone()[0])

    def store_memory(
        self,
        conversation_id: UUID,
        memory_key: str,
        memory_text: str,
        embedding: list[float],
        sequence_start: int,
        sequence_end: int,
        embedding_model: str,
    ) -> None:
        vector = self._vector_literal(embedding)
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                load_sql_section("conversation.sql", "store_memory"),
                (conversation_id, memory_key, memory_text, vector,
                 sequence_start, sequence_end, embedding_model),
            )

    def search_memories(
        self,
        conversation_id: UUID,
        embedding: list[float],
        limit: int,
        minimum_similarity: float,
        embedding_model: str,
    ) -> list[dict[str, Any]]:
        vector = self._vector_literal(embedding)
        query = load_sql_section("conversation.sql", "search_memories").format(
            dimensions=self._embedding_dimensions
        )
        with self._connection() as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                query,
                (conversation_id, embedding_model, vector, vector, minimum_similarity, vector, limit),
            )
            return [dict(row) for row in cursor.fetchall()]

    @staticmethod
    def _vector_literal(values: list[float]) -> str:
        return "[" + ",".join(repr(float(value)) for value in values) + "]"

    def create_conversation(self, title: str | None) -> dict[str, Any]:
        with self._connection() as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                load_sql_section("conversation.sql", "create_session"),
                (title.strip() if title and title.strip() else None,),
            )
            return dict(cursor.fetchone())

    def list_conversations(self, limit: int, offset: int) -> list[dict[str, Any]]:
        with self._connection() as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                load_sql_section("conversation.sql", "list_sessions"),
                (limit, offset),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_conversation(self, conversation_id: UUID) -> dict[str, Any] | None:
        with self._connection() as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                load_sql_section("conversation.sql", "get_session"),
                (conversation_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_conversation(self, conversation_id: UUID) -> bool:
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(load_sql_section("conversation.sql", "delete_session"), (conversation_id,))
            return cursor.rowcount > 0

    def get_messages(self, conversation_id: UUID) -> list[dict[str, Any]]:
        with self._connection() as connection, connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                load_sql_section("conversation.sql", "get_messages"),
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
                load_sql_section("conversation.sql", "load_recent_history"),
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
                load_sql_section("conversation.sql", "get_history_with_sequences"),
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
        final_assistant_metadata: dict[str, Any] | None = None,
    ) -> list[int]:
        """Append LangChain messages atomically and return their database IDs."""
        inserted_ids: list[int] = []
        with self._connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                load_sql_section("conversation.sql", "lock_session_row"),
                (conversation_id,),
            )
            if cursor.fetchone() is None:
                raise LookupError("Conversation not found.")
            cursor.execute(
                load_sql_section("conversation.sql", "latest_message_sequence"),
                (conversation_id,),
            )
            sequence = int(cursor.fetchone()[0])
            for index, message in enumerate(messages):
                converted = self._from_langchain_message(message)
                if converted is None:
                    continue
                role, content, tool_name, tool_call_id, tool_calls = converted
                metadata = (
                    final_assistant_metadata or {}
                    if index == len(messages) - 1 and isinstance(message, AIMessage)
                    else {}
                )
                sequence += 1
                cursor.execute(
                    load_sql_section("conversation.sql", "insert_message"),
                    (conversation_id, sequence, role, content, tool_name,
                     tool_call_id, Json(tool_calls) if tool_calls is not None else None,
                     Json(metadata)),
                )
                inserted_ids.append(int(cursor.fetchone()[0]))
            cursor.execute(
                load_sql_section("conversation.sql", "touch_session"),
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
