"""Health-check endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.config import settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    service: str


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        service=settings.app_name,
    )


@router.get("/", response_model=HealthResponse, include_in_schema=False)
async def root() -> HealthResponse:
    return await health()
