"""Application services for Gemini chat and PostgreSQL conversation memory."""

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel

from src.agents.vessel_agent import VesselAgent, create_vessel_agent
from src.app.config import Settings
from src.app.database import ConversationRepository


@dataclass(frozen=True)
class AppServices:
    settings: Settings
    repository: ConversationRepository
    summarizer: BaseChatModel
    agent: VesselAgent


def create_services(settings: Settings) -> AppServices:
    """Configure Gemini chat and PostgreSQL conversation storage."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    chat_model = ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.gemini_api_key,
        streaming=True,
    )
    return AppServices(
        settings=settings,
        repository=ConversationRepository(settings.database_url),
        summarizer=chat_model,
        agent=create_vessel_agent(
            chat_model,
            settings.max_tool_rounds
        ),
    )
