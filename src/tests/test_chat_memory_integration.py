import json
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from src.app.api.conversations import router


class FakeEmbeddings:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def aembed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


class FakeSummarizer:
    def __init__(self) -> None:
        self.calls: list[list[BaseMessage]] = []

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.calls.append(messages)
        return AIMessage(
            content="The user previously discussed vessel OLD VESSEL.",
            usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
        )


class FakeRepository:
    def __init__(self, conversation_id: UUID, history: list[tuple[int, BaseMessage]] | None = None) -> None:
        self.conversation_id = conversation_id
        self.history = history or []
        self.user_messages: list[str] = []
        self.appended_batches: list[tuple[list[BaseMessage], dict[str, Any] | None]] = []
        self.memories: list[dict[str, Any]] = []
        self.summary_cursor = 0

    @asynccontextmanager
    async def conversation_lock(self, conversation_id: UUID):
        if conversation_id != self.conversation_id:
            raise AssertionError("Unexpected conversation")
        yield

    def get_conversation(self, conversation_id: UUID) -> dict[str, Any] | None:
        if conversation_id != self.conversation_id:
            return None
        return {"session_id": conversation_id, "status": "active"}

    def get_summary_state(self, conversation_id: UUID) -> tuple[str, int]:
        return "", self.summary_cursor

    def load_history_with_sequences(self, conversation_id: UUID) -> list[tuple[int, BaseMessage]]:
        return list(self.history)

    def append_user_message(self, conversation_id: UUID, content: str) -> int:
        self.user_messages.append(content)
        return len(self.user_messages)

    def memory_exists(self, conversation_id: UUID, memory_key: str) -> bool:
        return any(memory["memory_key"] == memory_key for memory in self.memories)

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
        self.memories.append({
            "memory_key": memory_key,
            "memory_text": memory_text,
            "embedding": embedding,
            "sequence_start": sequence_start,
            "sequence_end": sequence_end,
            "embedding_model": embedding_model,
        })

    def update_summary_cursor(self, conversation_id: UUID, through_sequence: int) -> None:
        self.summary_cursor = through_sequence

    def search_memories(
        self,
        conversation_id: UUID,
        embedding: list[float],
        limit: int,
        minimum_similarity: float,
        embedding_model: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "memory_text": memory["memory_text"],
                "sequence_start": memory["sequence_start"],
                "sequence_end": memory["sequence_end"],
                "similarity": 0.92,
            }
            for memory in self.memories[-limit:]
        ]

    def append_messages(
        self,
        conversation_id: UUID,
        messages: list[BaseMessage],
        metadata: dict[str, Any] | None = None,
    ) -> list[int]:
        self.appended_batches.append((messages, metadata))
        return list(range(1, len(messages) + 1))


class FakeAgent:
    def __init__(self, use_memory: bool = False) -> None:
        self.use_memory = use_memory
        self.received_history: list[BaseMessage] = []

    async def astream(self, question, history=(), extra_tools=()):
        self.received_history = list(history)
        messages: list[BaseMessage] = [*history, HumanMessage(content=question)]

        if self.use_memory:
            memory_call = {"name": "search_conversation_memory", "args": {"query": "OLD VESSEL"}, "id": "memory-1"}
            messages.append(AIMessage(content="", tool_calls=[memory_call]))
            yield {"type": "tool_start", "tool": memory_call["name"], "args": memory_call["args"]}
            memory_result = await extra_tools[0].ainvoke(memory_call["args"])
            messages.append(ToolMessage(
                content=memory_result,
                tool_call_id=memory_call["id"],
                name=memory_call["name"],
            ))
            yield {
                "type": "tool_result",
                "tool": memory_call["name"],
                "content": memory_result,
                "duration_ms": 1.5,
            }

        answer = "Final answer using conversation memory." if self.use_memory else "Final answer."
        messages.append(AIMessage(content=answer))
        yield {"type": "token", "content": answer}
        yield {
            "type": "done",
            "answer": answer,
            "messages": messages,
            "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "model_calls": 1},
        }


def create_test_client(repository: FakeRepository, agent: FakeAgent):
    embeddings = FakeEmbeddings()
    summarizer = FakeSummarizer()
    settings = SimpleNamespace(
        chat_context_turns=2,
        memory_top_k=5,
        memory_min_similarity=0.25,
        gemini_embedding_model="fake-embedding",
    )
    services = SimpleNamespace(
        repository=repository,
        agent=agent,
        embeddings=embeddings,
        summarizer=summarizer,
        settings=settings,
    )
    app = FastAPI()
    app.include_router(router)
    app.state.services = services
    return TestClient(app), embeddings, summarizer


def parse_sse(response_text: str) -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = []
    for block in response_text.strip().split("\n\n"):
        lines = block.splitlines()
        event = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        events.append((event, data))
    return events


class ChatIntegrationTests(unittest.TestCase):
    def test_chat_streams_and_persists_final_answer_with_metadata(self) -> None:
        conversation_id = uuid4()
        repository = FakeRepository(conversation_id)
        client, _, _ = create_test_client(repository, FakeAgent())

        response = client.post(
            f"/conversations/{conversation_id}/chat",
            json={"message": "Where is the vessel?"},
        )

        self.assertEqual(response.status_code, 200)
        events = parse_sse(response.text)
        self.assertEqual([event for event, _ in events], ["token", "done"])
        self.assertEqual(repository.user_messages, ["Where is the vessel?"])
        messages, metadata = repository.appended_batches[-1]
        self.assertEqual(messages[-1].content, "Final answer.")
        self.assertEqual(metadata["token_usage"]["total_tokens"], 120)
        self.assertIn("timing", metadata)
        self.assertIn("total_duration_ms", metadata["timing"])

    def test_overflow_is_summarized_retrieved_and_saved_with_final_answer(self) -> None:
        conversation_id = uuid4()
        history: list[tuple[int, BaseMessage]] = [
            (1, HumanMessage(content="Remember OLD VESSEL.")),
            (2, AIMessage(content="I will remember it.")),
            (3, HumanMessage(content="Another question.")),
            (4, AIMessage(content="Another answer.")),
            (5, HumanMessage(content="Recent question.")),
            (6, AIMessage(content="Recent answer.")),
        ]
        repository = FakeRepository(conversation_id, history)
        agent = FakeAgent(use_memory=True)
        client, embeddings, summarizer = create_test_client(repository, agent)

        response = client.post(
            f"/conversations/{conversation_id}/chat",
            json={"message": "What was the old vessel?"},
        )

        self.assertEqual(response.status_code, 200)
        events = parse_sse(response.text)
        self.assertIn("tool_call", [event for event, _ in events])
        self.assertEqual(len(summarizer.calls), 1)
        self.assertEqual(repository.summary_cursor, 4)
        self.assertEqual(repository.memories[0]["memory_key"], "summary:1:4")
        self.assertEqual(len(embeddings.queries), 2)
        messages, metadata = repository.appended_batches[-1]
        self.assertEqual(messages[-1].content, "Final answer using conversation memory.")
        self.assertTrue(any(
            isinstance(message, ToolMessage) and message.name == "search_conversation_memory"
            for message in messages
        ))
        self.assertEqual(metadata["tool_inputs"][0]["name"], "search_conversation_memory")
        self.assertEqual(metadata["timing"]["tool_calls"][0]["name"], "search_conversation_memory")


if __name__ == "__main__":
    unittest.main()
