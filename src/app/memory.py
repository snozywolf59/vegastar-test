"""Conversation-scoped semantic-memory retrieval tool."""

import asyncio
from uuid import UUID

from langchain_core.embeddings import Embeddings
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from src.app.database import ConversationRepository


class ConversationMemoryQuery(BaseModel):
    query: str = Field(
        min_length=2,
        description="Earlier conversation detail or reference to retrieve, phrased as a focused search query.",
    )


def create_conversation_memory_tool(
    repository: ConversationRepository,
    embeddings: Embeddings,
    conversation_id: UUID,
    top_k: int,
    minimum_similarity: float,
    embedding_model: str,
) -> BaseTool:
    """Create a retrieval tool scoped to one conversation."""

    async def retrieve_conversation_memory(query: str) -> str:
        vector = await embeddings.aembed_query(query)
        memories = await asyncio.to_thread(
            repository.search_memories,
            conversation_id,
            vector,
            top_k,
            minimum_similarity,
            embedding_model,
        )
        if not memories:
            return "No relevant older conversation memory was found."
        return "\n\n---\n\n".join(
            f"Older conversation (messages {item['sequence_start']}–{item['sequence_end']}, "
            f"similarity {float(item['similarity']):.3f}):\n{item['memory_text']}"
            for item in memories
        )

    return StructuredTool.from_function(
        coroutine=retrieve_conversation_memory,
        name="search_conversation_memory",
        description=(
            "Search older summarized turns from this conversation only. Call when a follow-up "
            "depends on a vessel, company, date, choice, or other detail not present in recent "
            "messages. Do not use this tool for current vessel/AIS facts."
        ),
        args_schema=ConversationMemoryQuery,
    )
