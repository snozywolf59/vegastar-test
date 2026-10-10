"""Application services for Gemini chat and PostgreSQL conversation memory."""

from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from src.agents.vessel_agent import VesselAgent, create_vessel_agent
from src.app.config import Settings
from src.app.database import ConversationRepository


@dataclass(frozen=True)
class AppServices:
    settings: Settings
    repository: ConversationRepository
    embeddings: Embeddings
    agent: VesselAgent


def create_services(settings: Settings) -> AppServices:
    """Configure Gemini chat, Gemini embeddings, and PostgreSQL storage."""
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )

    chat_model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        streaming=True,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.gemini_embedding_model,
        google_api_key=settings.gemini_api_key,
        output_dimensionality=settings.embedding_dimensions,
    )
    return AppServices(
        settings=settings,
        repository=ConversationRepository(
            settings.database_url,
            settings.embedding_dimensions,
        ),
        embeddings=embeddings,
        agent=create_vessel_agent(
            chat_model,
            settings.max_tool_rounds
        ),
    )
