"""FastAPI application entry point."""

import asyncio
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI

from src.app.api.conversations import router as conversations_router
from src.app.config import Settings
from src.app.services import create_services

load_dotenv()

async def lifespan(application: FastAPI) -> AsyncGenerator[None]:
    settings = Settings.from_env()
    services = create_services(settings)
    await asyncio.to_thread(services.repository.initialize)
    application.state.services = services
    yield


app = FastAPI(
    title="Vessel Assistant API",
    lifespan=lifespan,
)
app.include_router(conversations_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
