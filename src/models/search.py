"""Search request and response models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.models.candidate import RankedCandidate, SeniorityLevel
from src.models.job import JobDescription


class SearchStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class SearchFilter(BaseModel):
    locations: list[str] = Field(default_factory=list)
    seniority_levels: list[SeniorityLevel] = Field(default_factory=list)
    min_years_experience: float | None = None
    max_years_experience: float | None = None
    required_skills: list[str] = Field(default_factory=list)
    exclude_companies: list[str] = Field(default_factory=list)
    min_salary: float | None = None
    max_salary: float | None = None
    remote_only: bool = False


class SearchRequest(BaseModel):
    job_description_text: str = Field(description="Raw job description text to analyze and search against")
    filters: SearchFilter = Field(default_factory=SearchFilter)
    top_k: int = Field(default=10, ge=1, le=100)
    include_outreach_drafts: bool = Field(default=True)


class AgentTrace(BaseModel):
    agent_name: str
    action: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0


class SearchResponse(BaseModel):
    search_id: str
    status: SearchStatus
    job_description: JobDescription | None = None
    ranked_candidates: list[RankedCandidate] = Field(default_factory=list)
    total_candidates_found: int = 0
    agent_traces: list[AgentTrace] = Field(default_factory=list)
    summary: str = ""
    duration_ms: float = 0.0


class CandidateSearchRequest(BaseModel):
    skills: list[str] = Field(default_factory=list)
    location: str | None = None
    min_years_experience: float | None = None
    max_years_experience: float | None = None
    seniority_levels: list[SeniorityLevel] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    limit: int = Field(default=20, ge=1, le=100)
