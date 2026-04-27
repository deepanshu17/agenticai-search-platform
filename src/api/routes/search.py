"""Talent search endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from src.models.search import SearchRequest, SearchResponse, SearchStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


def _get_orchestrator():  # type: ignore[return]
    """Dependency that returns the app-level Orchestrator instance."""
    from src.main import get_orchestrator

    return get_orchestrator()


@router.post(
    "",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a full multi-agent talent search",
    description=(
        "Submit a job description and receive a ranked list of matching candidates "
        "with evaluation scores, skill gap analysis, and personalised outreach drafts."
    ),
)
async def run_search(
    request: SearchRequest,
    orchestrator=Depends(_get_orchestrator),
) -> SearchResponse:
    """Execute a multi-agent talent search pipeline."""
    if not request.job_description_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="job_description_text must not be empty",
        )

    logger.info("POST /search top_k=%d", request.top_k)
    response = await orchestrator.search(request)

    if response.status == SearchStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=response.summary or "Search failed",
        )
    return response
