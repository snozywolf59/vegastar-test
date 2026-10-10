"""Tool-calling LangChain agent for maritime vessel questions."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Sequence, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessageChunk
)
from langchain_core.tools import BaseTool

from src.agents.tools import VESSEL_TOOLS


SYSTEM_PROMPT = """You are a maritime vessel data assistant. Answer the user in the
language they used. Use the available tools for vessel, company, AIS position,
route, and dark-gap facts. Never invent vessel names, identifiers, positions,
times, distances, ownership, or statistics. If a tool returns no data, say that
the supplied dataset has no matching data. Vessel names may be misspelled or
ambiguous: use search results and ask the user to choose when more than one
candidate could match. Resolve follow-up references from the conversation.
Interpret unqualified dates and times in UTC when the data context indicates
UTC. Explain that a requested-time position is the nearest reported AIS point
and include its time offset. Do not claim interpolated positions. For routes,
summarize the tool's returned statistics; do not reproduce large GeoJSON unless
the user asks for map data. Treat tool output as the sole source for factual
claims about the supplied dataset."""


@dataclass
class AgentResponse:
    """Final answer and updated messages for a conversation turn."""

    answer: str
    messages: list[BaseMessage]


class VesselAgent:
    """Run a chat model with the vessel tools and caller-managed history.

    The supplied model must support LangChain tool calling. Pass the returned
    messages into the next call for follow-up questions and persist them in the
    conversation store when durable history is required.
    """

    def __init__(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool] = VESSEL_TOOLS,
        max_tool_rounds: int = 3,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1.")
        self._model = model.bind_tools(list(tools))
        self._tools = {item.name: item for item in tools}
        self._max_tool_rounds = max_tool_rounds

    def invoke(
        self,
        question: str,
        history: Sequence[BaseMessage] = (),
    ) -> AgentResponse:
        """Answer one question, returning history extended through the answer."""
        messages = [SystemMessage(content=SYSTEM_PROMPT), *history, HumanMessage(content=question)]

        for round_index in range(self._max_tool_rounds + 1):
            response = self._model.invoke(messages)
            if not isinstance(response, AIMessage):
                raise TypeError("The chat model returned a non-assistant message.")
            messages.append(response)
            if not response.tool_calls:
                return AgentResponse(
                    answer=self._message_text(response),
                    messages=messages[1:],
                )
            if round_index == self._max_tool_rounds:
                break

            for call in response.tool_calls:
                tool_name = call["name"]
                selected_tool = self._tools.get(tool_name)
                if selected_tool is None:
                    result = f"Tool error: unknown tool '{tool_name}'."
                else:
                    try:
                        result = selected_tool.invoke(call["args"])
                    except Exception as error:
                        result = f"Tool error ({tool_name}): {error}"
                messages.append(
                    ToolMessage(
                        content=self._message_text(result),
                        tool_call_id=call["id"],
                        name=tool_name,
                    )
                )

        raise RuntimeError(
            f"The agent exceeded its limit of {self._max_tool_rounds} tool rounds."
        )
    
    async def astream(
        self,
        question: str,
        history: Sequence[BaseMessage] = (),
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream agent events, including tokens and tool execution results."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            *history,
            HumanMessage(content=question),
        ]

        for round_index in range(self._max_tool_rounds + 1):
            response_chunk: AIMessageChunk | None = None

            # Stream model output chunk by chunk.
            async for chunk in self._model.astream(messages):
                if response_chunk is None:
                    response_chunk = chunk
                else:
                    response_chunk += chunk

                text = self._message_text(chunk.content)
                if text:
                    yield {
                        "type": "token",
                        "content": text,
                    }

            if response_chunk is None:
                raise RuntimeError("The model returned no response.")

            messages.append(response_chunk)

            # No tool calls: the assistant has finished its answer.
            if not response_chunk.tool_calls:
                yield {
                    "type": "done",
                    "answer": self._message_text(response_chunk.content),
                    "messages": messages[1:],
                }
                return

            if round_index == self._max_tool_rounds:
                raise RuntimeError(
                    f"The agent exceeded its limit of "
                    f"{self._max_tool_rounds} tool rounds."
                )

            # Execute tool calls after aggregating their streamed chunks.
            for call in response_chunk.tool_calls:
                tool_name = call["name"]

                yield {
                    "type": "tool_start",
                    "tool": tool_name,
                    "args": call["args"],
                }

                selected_tool = self._tools.get(tool_name)

                if selected_tool is None:
                    result = f"Tool error: unknown tool '{tool_name}'."
                else:
                    try:
                        result = await selected_tool.ainvoke(call["args"])
                    except Exception as error:
                        result = f"Tool error ({tool_name}): {error}"
                        yield {
                            "type": "error",
                            "code": "tool_error",
                            "tool": tool_name,
                            "message": str(error),
                        }

                result_text = self._message_text(result)

                messages.append(
                    ToolMessage(
                        content=result_text,
                        tool_call_id=call["id"],
                        name=tool_name,
                    )
                )

                yield {
                    "type": "tool_result",
                    "tool": tool_name,
                    "content": result_text,
                }

    @staticmethod
    def _message_text(value: object) -> str:
        """Convert model or tool content into text suitable for message history."""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return "".join(
                part.get("text", "")
                for part in value
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            )
        return str(value)


def create_vessel_agent(model: BaseChatModel, max_tool_rounds:int) -> VesselAgent:
    """Create an agent using a configured LangChain chat model."""
    return VesselAgent(model=model, max_tool_rounds=max_tool_rounds)
