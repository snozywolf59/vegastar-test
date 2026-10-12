"""Conversation management and SSE chat endpoints."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from src.agents.vessel_agent import SYSTEM_PROMPT
from src.app.memory import create_conversation_memory_tool
from src.app.resources import load_prompt
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

            summary, summary_through = await asyncio.to_thread(
                repository.get_summary_state,
                conversation_id,
            )
            sequenced_history = await asyncio.to_thread(
                repository.load_history_with_sequences,
                conversation_id,
            )
            old_messages, recent_messages = _split_history(
                [message for _, message in sequenced_history],
                services.settings.chat_context_turns - 1,
            )
            old_count = len(old_messages)
            new_old_messages = [
                message
                for sequence, message in sequenced_history[:old_count]
                if sequence > summary_through
            ]
            history = recent_messages
            summary_usage: dict[str, Any] | None = None
            await asyncio.to_thread(
                repository.append_user_message,
                conversation_id,
                payload.message,
            )

            streamed_map_keys: set[str] = set()
            try:
                if summary and summary_through:
                    legacy_memory_key = f"legacy-summary:{summary_through}"
                    exists = await asyncio.to_thread(
                        repository.memory_exists,
                        conversation_id,
                        legacy_memory_key,
                    )
                    if not exists:
                        legacy_embedding = await services.embeddings.aembed_query(summary)
                        await asyncio.to_thread(
                            repository.store_memory,
                            conversation_id,
                            legacy_memory_key,
                            summary,
                            legacy_embedding,
                            1,
                            summary_through,
                            services.settings.gemini_embedding_model,
                        )
            except Exception:
                logger.exception("Legacy conversation summary migration failed.")
                yield _sse(
                    "error",
                    {"code": "memory_migration_failed", "message": "Could not index the existing conversation summary."},
                )
                return

            if new_old_messages:
                try:
                    summary_result = await services.summarizer.ainvoke([
                        SystemMessage(content=load_prompt("conversation_summary.md")),
                        HumanMessage(content=_format_messages(new_old_messages)),
                    ])
                    memory_text = _content_text(summary_result.content).strip()
                    if not memory_text:
                        raise ValueError("The conversation summary was empty.")
                    summary_usage = _normalize_token_usage(
                        summary_result.usage_metadata,
                        model_calls=1,
                    )
                    memory_embedding = await services.embeddings.aembed_query(memory_text)
                    sequence_start = min(
                        sequence
                        for sequence, message in sequenced_history[:old_count]
                        if sequence > summary_through
                    )
                    sequence_end = max(
                        sequence
                        for sequence, message in sequenced_history[:old_count]
                        if sequence > summary_through
                    )
                    await asyncio.to_thread(
                        repository.store_memory,
                        conversation_id,
                        f"summary:{sequence_start}:{sequence_end}",
                        memory_text,
                        memory_embedding,
                        sequence_start,
                        sequence_end,
                        services.settings.gemini_embedding_model,
                    )
                    await asyncio.to_thread(
                        repository.update_summary_cursor,
                        conversation_id,
                        sequence_end,
                    )
                except Exception:
                    logger.exception("Conversation summarization failed.")
                    yield _sse("error", {"code": "summary_failed", "message": "Could not update conversation summary."})
                    return
            try:
                memory_tool = create_conversation_memory_tool(
                    repository,
                    services.embeddings,
                    conversation_id,
                    services.settings.memory_top_k,
                    services.settings.memory_min_similarity,
                    services.settings.gemini_embedding_model,
                )
                async for agent_event in services.agent.astream(
                    payload.message,
                    history,
                    extra_tools=[memory_tool],
                ):
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
                    elif event_type == "tool_result":
                        map_data = _tool_result_to_geojson(
                            str(agent_event.get("tool", "")),
                            agent_event.get("content", ""),
                        )
                        if map_data is not None:
                            map_key = json.dumps(map_data, sort_keys=True, default=str)
                            if map_key not in streamed_map_keys:
                                streamed_map_keys.add(map_key)
                                yield _sse("map_data", map_data)
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
                        input_metadata = _build_llm_input_metadata(
                            payload.message,
                            history,
                            new_messages,
                        )
                        token_usage = _merge_token_usage(
                            summary_usage,
                            agent_event.get("usage"),
                        )
                        input_metadata["token_usage"] = token_usage
                        input_metadata["map_data"] = _map_data_from_messages(new_messages)
                        await asyncio.to_thread(
                            repository.append_messages,
                            conversation_id,
                            new_messages,
                            input_metadata,
                        )
                        answer = str(agent_event.get("answer", ""))
                        yield _sse(
                            "done",
                            {
                                "conversation_id": conversation_id,
                                "answer": answer,
                                "token_usage": token_usage,
                            },
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


def _split_history(messages: list[BaseMessage], recent_turns: int) -> tuple[list[BaseMessage], list[BaseMessage]]:
    if recent_turns <= 0:
        return messages, []
    user_indices = [index for index, message in enumerate(messages) if isinstance(message, HumanMessage)]
    if len(user_indices) <= recent_turns:
        return [], messages
    split_at = user_indices[-recent_turns]
    return messages[:split_at], messages[split_at:]


def _format_messages(messages: list[BaseMessage]) -> str:
    formatted: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == "search_conversation_memory":
            continue
        content = _content_text(message.content)
        if content:
            formatted.append(f"{message.type}: {content}")
        if isinstance(message, AIMessage) and message.tool_calls:
            formatted.append(
                "assistant tool calls: "
                + json.dumps(message.tool_calls, ensure_ascii=False, default=str)
            )
    return "\n".join(formatted)


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return str(content)


def _tool_result_to_geojson(tool_name: str, content: object) -> dict[str, Any] | None:
    """Convert selected AIS tool results into frontend-safe GeoJSON."""
    if tool_name not in {
        "get_vessel_position_near_time",
        "get_vessel_last_position",
        "get_vessel_route",
        "get_vessels_journeys",
    }:
        return None

    try:
        result = json.loads(content) if isinstance(content, str) else content
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(result, dict) or result.get("error"):
        return None

    if tool_name in {"get_vessel_position_near_time", "get_vessel_last_position"}:
        position = result.get("position")
        if not isinstance(position, dict):
            return None
        latitude, longitude = position.get("lat"), position.get("lon")
        if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
            return None
        geojson = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
            "properties": {
                key: value for key, value in position.items() if key not in {"lat", "lon"}
            },
        }
    elif tool_name == "get_vessel_route":
        geometry = result.get("geojson")
        if not _is_geojson_geometry(geometry):
            return None
        geojson = {
            "type": "Feature",
            "geometry": geometry,
            "properties": {key: value for key, value in result.items() if key != "geojson"},
        }
    else:
        journeys = result.get("journeys")
        if not isinstance(journeys, list):
            return None
        features = [
            {
                "type": "Feature",
                "geometry": journey["geojson"],
                "properties": {
                    key: value for key, value in journey.items() if key != "geojson"
                },
            }
            for journey in journeys
            if isinstance(journey, dict) and _is_geojson_geometry(journey.get("geojson"))
        ]
        if not features:
            return None
        geojson = {"type": "FeatureCollection", "features": features}

    return {"tool": tool_name, "geojson": geojson}


def _map_data_from_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Extract unique map payloads from tool messages before persistence."""
    map_data: list[dict[str, Any]] = []
    seen: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage) or not message.name:
            continue
        item = _tool_result_to_geojson(message.name, _content_text(message.content))
        if item is None:
            continue
        key = json.dumps(item, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            map_data.append(item)
    return map_data


def _is_geojson_geometry(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("type"), str)
        and isinstance(value.get("coordinates"), list)
    )


def _build_llm_input_metadata(
    prompt: str,
    history: list[BaseMessage],
    turn_messages: list[BaseMessage],
) -> dict[str, Any]:
    return {
        "system_prompt": SYSTEM_PROMPT,
        "prompt": prompt,
        "context": [
            {"role": message.type, "content": _content_text(message.content)}
            for message in history
        ],
        "tool_inputs": [
            {"name": call["name"], "arguments": call["args"]}
            for message in turn_messages
            if isinstance(message, AIMessage)
            for call in message.tool_calls
        ],
        "tool_outputs": [
            {"name": message.name, "content": _content_text(message.content)}
            for message in turn_messages
            if isinstance(message, ToolMessage)
        ],
    }


def _normalize_token_usage(
    usage: dict[str, Any] | None,
    model_calls: int,
) -> dict[str, int | None]:
    return {
        "input_tokens": int(usage["input_tokens"]) if usage and usage.get("input_tokens") is not None else None,
        "output_tokens": int(usage["output_tokens"]) if usage and usage.get("output_tokens") is not None else None,
        "total_tokens": int(usage["total_tokens"]) if usage and usage.get("total_tokens") is not None else None,
        "model_calls": model_calls,
    }


def _merge_token_usage(
    *usages: dict[str, int | None] | None,
) -> dict[str, int | None] | None:
    available = [usage for usage in usages if usage is not None]
    if not available:
        return None
    result: dict[str, int | None] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        known_values = [usage[key] for usage in available if usage.get(key) is not None]
        result[key] = sum(known_values) if known_values else None
    result["model_calls"] = sum(int(usage.get("model_calls") or 0) for usage in available)
    return result
