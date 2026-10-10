"""FastAPI application entry point."""

import asyncio
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse

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


@app.get("/", include_in_schema=False)
async def chat_ui() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
