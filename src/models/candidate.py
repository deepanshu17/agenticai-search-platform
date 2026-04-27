"""Candidate data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SeniorityLevel(str, Enum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    STAFF = "staff"
    PRINCIPAL = "principal"
    DIRECTOR = "director"
    VP = "vp"
    C_LEVEL = "c_level"


class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"


class WorkLocation(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ON_SITE = "on_site"


class Skill(BaseModel):
    name: str
    years_of_experience: float = Field(default=0.0, ge=0)
    proficiency: str = Field(default="intermediate")
    last_used: str | None = None


class WorkExperience(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: str | None = None
    is_current: bool = False
    description: str = ""
    skills_used: list[str] = Field(default_factory=list)


class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    graduation_year: int | None = None


class Candidate(BaseModel):
    id: str
    name: str
    email: str | None = None
    phone: str | None = None
    location: str = ""
    headline: str = ""
    summary: str = ""
    skills: list[Skill] = Field(default_factory=list)
    experience: list[WorkExperience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    seniority_level: SeniorityLevel = SeniorityLevel.MID
    total_years_experience: float = Field(default=0.0, ge=0)
    preferred_employment_type: EmploymentType | None = None
    preferred_work_location: WorkLocation | None = None
    expected_salary_min: float | None = None
    expected_salary_max: float | None = None
    profile_url: str | None = None
    source: str = "unknown"
    last_active: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateScore(BaseModel):
    candidate_id: str
    overall_score: float = Field(ge=0.0, le=100.0)
    skills_match_score: float = Field(ge=0.0, le=100.0)
    experience_score: float = Field(ge=0.0, le=100.0)
    education_score: float = Field(ge=0.0, le=100.0)
    culture_fit_score: float = Field(ge=0.0, le=100.0)
    explanation: str = ""
    strengths: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class RankedCandidate(BaseModel):
    rank: int
    candidate: Candidate
    score: CandidateScore
    outreach_draft: str | None = None
