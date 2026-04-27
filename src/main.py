"""FastAPI application factory and entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import candidates, health, search
from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global orchestrator instance (initialised on startup)
# ---------------------------------------------------------------------------

_orchestrator = None


def get_orchestrator():
    """Return the application-level :class:`Orchestrator` instance."""
    if _orchestrator is None:
        raise RuntimeError("Orchestrator has not been initialised. Did app startup run?")
    return _orchestrator


def _create_llm():
    """Create the LangChain LLM instance.

    If OPENAI_API_KEY is not set the service will still start but LLM-powered
    features (job analysis, candidate evaluation) will return defaults.
    """
    if not settings.openai_api_key:
        logger.warning(
            "OPENAI_API_KEY is not set. LLM features will use fallback defaults. "
            "Set OPENAI_API_KEY in your environment or .env file to enable full functionality."
        )
        from src.utils.fake_llm import FakeLLM

        return FakeLLM()

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.openai_temperature,
        api_key=settings.openai_api_key,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise and tear-down application resources."""
    global _orchestrator  # noqa: PLW0603

    logger.info("Starting up %s v%s", settings.app_name, settings.app_version)

    from src.agents.orchestrator import Orchestrator
    from src.mcp.client import MCPClient

    llm = _create_llm()
    mcp_client = MCPClient()
    _orchestrator = Orchestrator(llm=llm, mcp_client=mcp_client)

    logger.info("Orchestrator initialised successfully")
    yield

    logger.info("Shutting down %s", settings.app_name)
    _orchestrator = None


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI-native talent search service with multi-agent orchestration, "
            "LangChain pipelines, and MCP-enabled tool calling."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(search.router, prefix="/api/v1")
    app.include_router(candidates.router, prefix="/api/v1")

    return app


app = create_app()


def run() -> None:
    """CLI entry point: start the Uvicorn server."""
    uvicorn.run(
        "src.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    run()
