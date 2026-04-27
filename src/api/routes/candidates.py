"""Candidate management endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from src.mcp.server import _MOCK_CANDIDATES, _tool_search_candidates
from src.models.candidate import Candidate, SeniorityLevel, Skill, WorkExperience
from src.models.search import CandidateSearchRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/candidates", tags=["candidates"])


def _raw_to_candidate(raw: dict) -> Candidate:
    from src.models.candidate import Education

    skills = [Skill(name=s) for s in raw.get("skills", [])]
    experience = []
    if raw.get("current_company"):
        experience.append(
            WorkExperience(
                company=raw["current_company"],
                title=raw.get("headline", ""),
                start_date="2020-01",
                is_current=True,
                description=raw.get("summary", ""),
            )
        )
    education = []
    if raw.get("education"):
        education.append(
            Education(
                institution=raw["education"],
                degree="Bachelor's",
                field_of_study="Computer Science",
            )
        )
    try:
        seniority = SeniorityLevel(raw.get("seniority", "mid"))
    except ValueError:
        seniority = SeniorityLevel.MID

    return Candidate(
        id=raw.get("id", "unknown"),
        name=raw.get("name", "Unknown"),
        headline=raw.get("headline", ""),
        location=raw.get("location", ""),
        summary=raw.get("summary", ""),
        skills=skills,
        experience=experience,
        education=education,
        seniority_level=seniority,
        total_years_experience=float(raw.get("years_experience", 0)),
        source="database",
    )


@router.get("", response_model=list[Candidate], summary="List candidates")
async def list_candidates(limit: int = 20) -> list[Candidate]:
    """Return all candidates in the database (paginated)."""
    return [_raw_to_candidate(c) for c in _MOCK_CANDIDATES[:limit]]


@router.get("/{candidate_id}", response_model=Candidate, summary="Get candidate by ID")
async def get_candidate(candidate_id: str) -> Candidate:
    """Retrieve a single candidate profile by ID."""
    raw = next((c for c in _MOCK_CANDIDATES if c["id"] == candidate_id), None)
    if raw is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate '{candidate_id}' not found",
        )
    return _raw_to_candidate(raw)


@router.post("/search", response_model=list[Candidate], summary="Search candidates")
async def search_candidates(request: CandidateSearchRequest) -> list[Candidate]:
    """Search candidates using skills, location, seniority, and experience filters."""
    result = await _tool_search_candidates(
        {
            "skills": request.skills,
            "location": request.location,
            "seniority": request.seniority_levels[0].value if request.seniority_levels else None,
            "min_years_experience": request.min_years_experience,
            "limit": request.limit,
        }
    )
    return [_raw_to_candidate(c) for c in result.get("candidates", [])]
