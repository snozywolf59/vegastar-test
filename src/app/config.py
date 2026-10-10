"""Environment-backed settings for the API and conversation memory."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    gemini_api_key: str
    gemini_model: str
    gemini_embedding_model: str
    embedding_dimensions: int
    chat_context_turns: int
    memory_top_k: int
    memory_min_similarity: float
    max_tool_rounds: int

    @classmethod
    def from_env(cls) -> "Settings":
        """Read settings and fail early for missing or invalid required values."""
        required = ("DATABASE_URL", "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_EMBEDDING_MODEL")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

        dimensions = int(os.getenv("GEMINI_EMBEDDING_DIMENSIONS", "768"))
        if not 1 <= dimensions <= 2000:
            raise ValueError("GEMINI_EMBEDDING_DIMENSIONS must be between 1 and 2000.")
        similarity = float(os.getenv("MEMORY_MIN_SIMILARITY", "0.25"))
        if not 0.0 <= similarity <= 1.0:
            raise ValueError("MEMORY_MIN_SIMILARITY must be between 0 and 1.")

        return cls(
            database_url=os.environ["DATABASE_URL"],
            gemini_api_key=os.environ["GEMINI_API_KEY"],
            gemini_model=os.environ["GEMINI_MODEL"],
            gemini_embedding_model=os.environ["GEMINI_EMBEDDING_MODEL"],
            embedding_dimensions=dimensions,
            chat_context_turns=max(1, int(os.getenv("CHAT_CONTEXT_TURNS", "8"))),
            memory_top_k=max(1, int(os.getenv("MEMORY_TOP_K", "5"))),
            memory_min_similarity=similarity,
            max_tool_rounds=max(1, int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "6"))),
        )
