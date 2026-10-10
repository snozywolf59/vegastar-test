"""Environment-backed settings for the API and conversation summaries."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    gemini_api_key: str
    gemini_model: str
    chat_context_turns: int
    max_tool_rounds: int

    @classmethod
    def from_env(cls) -> "Settings":
        """Read settings and fail early for missing or invalid required values."""
        required = ("DATABASE_URL", "GEMINI_API_KEY", "GEMINI_MODEL")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

        return cls(
            database_url=os.environ["DATABASE_URL"],
            gemini_api_key=os.environ["GEMINI_API_KEY"],
            gemini_model=os.environ["GEMINI_MODEL"],
            chat_context_turns=max(1, int(os.getenv("CHAT_CONTEXT_TURNS", "8"))),
            max_tool_rounds=max(1, int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "6"))),
        )
