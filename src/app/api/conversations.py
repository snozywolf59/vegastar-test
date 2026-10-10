"""Conversation management and SSE chat endpoints."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.app.services import AppServices

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


class CreateConversationRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)


def _services(request: Request) -> AppServices:
    return request.app.state.services


def _sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("", status_code=201)
async def create_conversation(
    payload: CreateConversationRequest,
    request: Request,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _services(request).repository.create_conversation,
        payload.title,
    )


@router.get("")
async def list_conversations(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    conversations = await asyncio.to_thread(
        _services(request).repository.list_conversations,
        limit,
        offset,
    )
    return {"count": len(conversations), "conversations": conversations}


@router.get("/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: UUID,
    request: Request,
) -> dict[str, Any]:
    repository = _services(request).repository
    conversation = await asyncio.to_thread(repository.get_conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    messages = await asyncio.to_thread(repository.get_messages, conversation_id)
    return {"conversation_id": conversation_id, "messages": messages}


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: UUID,
    request: Request,
) -> dict[str, Any]:
    deleted = await asyncio.to_thread(
        _services(request).repository.delete_conversation,
        conversation_id,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"conversation_id": conversation_id, "deleted": True}


@router.post("/{conversation_id}/chat")
async def chat(
    conversation_id: UUID,
    payload: ChatRequest,
    request: Request,
) -> StreamingResponse:
    services = _services(request)

    async def stream() -> AsyncIterator[str]:
        repository = services.repository
        async with repository.conversation_lock(conversation_id):
            conversation = await asyncio.to_thread(
                repository.get_conversation,
                conversation_id,
            )
            if conversation is None:
                yield _sse("error", {"code": "conversation_not_found"})
                return

            history = await asyncio.to_thread(
                repository.load_recent_history,
                conversation_id,
                services.settings.chat_context_turns,
            )
            await asyncio.to_thread(
                repository.append_user_message,
                conversation_id,
                payload.message,
            )

            user_turns = sum(isinstance(message, HumanMessage) for message in history)
            if user_turns >= services.settings.chat_context_turns:
                try:
                    query_embedding = await services.embeddings.aembed_query(payload.message)
                    memories = await asyncio.to_thread(
                        repository.search_memories,
                        conversation_id,
                        query_embedding,
                        services.settings.memory_top_k,
                        services.settings.memory_min_similarity,
                    )
                    if memories:
                        memory_message = SystemMessage(
                            content=(
                                "Relevant facts recalled from this conversation only. "
                                "Use them to resolve follow-ups, and distinguish them "
                                "from facts retrieved from vessel tools:\n"
                                + "\n---\n".join(memories)
                            )
                        )
                        history = [memory_message, *history]
                except Exception:
                    logger.exception("Conversation memory retrieval failed.")
                    yield _sse(
                        "error",
                        {"code": "memory_retrieval_failed", "message": "Long-term memory is temporarily unavailable."},
                    )

            try:
                async for agent_event in services.agent.astream(payload.message, history):
                    event_type = agent_event.get("type")
                    if event_type == "token":
                        yield _sse("token", {"text": agent_event.get("content", "")})
                    elif event_type == "tool_start":
                        yield _sse(
                            "tool_call",
                            {
                                "name": agent_event.get("tool"),
                                "arguments": agent_event.get("args", {}),
                            },
                        )
                    elif event_type == "error":
                        yield _sse(
                            "error",
                            {
                                "code": agent_event.get("code", "tool_error"),
                                "tool": agent_event.get("tool"),
                                "message": agent_event.get("message", "Tool execution failed."),
                            },
                        )
                    elif event_type == "done":
                        messages = agent_event.get("messages", [])
                        new_messages = _messages_after_current_question(
                            messages,
                            payload.message,
                        )
                        inserted_ids = await asyncio.to_thread(
                            repository.append_messages,
                            conversation_id,
                            new_messages,
                        )
                        answer = str(agent_event.get("answer", ""))
                        memory_text = f"User: {payload.message}\nAssistant: {answer}"
                        try:
                            vectors = await services.embeddings.aembed_documents([memory_text])
                            if vectors:
                                await asyncio.to_thread(
                                    repository.store_memory,
                                    conversation_id,
                                    memory_text,
                                    vectors[0],
                                    inserted_ids[-1] if inserted_ids else None,
                                )
                        except Exception:
                            logger.exception("Conversation memory write failed.")
                            yield _sse(
                                "error",
                                {"code": "memory_write_failed", "message": "The answer was saved, but long-term memory could not be updated."},
                            )
                        yield _sse(
                            "done",
                            {"conversation_id": conversation_id, "answer": answer},
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Chat generation failed for conversation %s.", conversation_id)
                await asyncio.to_thread(
                    repository.append_messages,
                    conversation_id,
                    [AIMessage(content="The request could not be completed.")],
                )
                yield _sse(
                    "error",
                    {"code": "chat_failed", "message": "The assistant could not complete this request."},
                )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _messages_after_current_question(
    messages: list[BaseMessage],
    question: str,
) -> list[BaseMessage]:
    """Return only generated assistant/tool messages for the current turn."""
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage) and message.content == question:
            return messages[index + 1:]
    raise RuntimeError("The agent response did not contain the current user message.")
