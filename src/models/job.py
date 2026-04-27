"""Job description and position data models."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.candidate import EmploymentType, SeniorityLevel, WorkLocation


class SalaryRange(BaseModel):
    min: float = Field(ge=0)
    max: float = Field(ge=0)
    currency: str = "USD"
    period: str = "annual"


class JobRequirement(BaseModel):
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_years_experience: float = Field(default=0.0, ge=0)
    max_years_experience: float | None = None
    required_education: str | None = None
    certifications: list[str] = Field(default_factory=list)


class JobDescription(BaseModel):
    id: str
    title: str
    company: str
    department: str = ""
    description: str
    requirements: JobRequirement = Field(default_factory=JobRequirement)
    seniority_level: SeniorityLevel = SeniorityLevel.MID
    employment_type: EmploymentType = EmploymentType.FULL_TIME
    work_location: WorkLocation = WorkLocation.HYBRID
    location: str = ""
    salary_range: SalaryRange | None = None
    benefits: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    parsed_requirements: dict = Field(default_factory=dict)
