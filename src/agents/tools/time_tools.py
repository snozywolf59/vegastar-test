"""LangChain tools for current date and time."""

from datetime import datetime, timezone

from langchain_core.tools import tool

from src.agents.tools._common import to_json


@tool
def get_current_time() -> str:
    """Get the current UTC time as an ISO-8601 timestamp with timezone."""
    current_time = datetime.now(timezone.utc)
    return to_json({
        "timezone": "UTC",
        "iso_time": current_time.isoformat().replace("+00:00", "Z"),
    })


TIME_TOOLS = [get_current_time]
