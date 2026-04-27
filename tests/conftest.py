"""Shared pytest fixtures and configuration."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.models.candidate import (
    Candidate,
    CandidateScore,
    Education,
    SeniorityLevel,
    Skill,
    WorkExperience,
)
from src.models.job import JobDescription, JobRequirement, SeniorityLevel as JobSeniority
from src.models.search import SearchFilter, SearchRequest
from src.utils.fake_llm import FakeLLM


# ---------------------------------------------------------------------------
# Reusable fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def sample_candidate() -> Candidate:
    return Candidate(
        id="cand-001",
        name="Alice Chen",
        headline="Senior ML Engineer",
        location="San Francisco, CA",
        summary="ML engineer focused on LLMs.",
        skills=[
            Skill(name="Python", years_of_experience=7.0),
            Skill(name="LangChain", years_of_experience=2.0),
            Skill(name="PyTorch", years_of_experience=4.0),
        ],
        experience=[
            WorkExperience(
                company="OpenAI",
                title="Senior ML Engineer",
                start_date="2021-01",
                is_current=True,
                description="Building LLM-based products.",
            )
        ],
        education=[
            Education(
                institution="Stanford University",
                degree="MS",
                field_of_study="Computer Science",
                graduation_year=2017,
            )
        ],
        seniority_level=SeniorityLevel.SENIOR,
        total_years_experience=7.0,
    )


@pytest.fixture
def sample_job() -> JobDescription:
    return JobDescription(
        id="job-test-001",
        title="Senior ML Engineer",
        company="TechCorp",
        description="We are looking for a senior ML engineer to build LLM-based products.",
        requirements=JobRequirement(
            required_skills=["Python", "PyTorch", "LangChain"],
            preferred_skills=["Kubernetes", "MLflow"],
            min_years_experience=5.0,
        ),
        seniority_level=SeniorityLevel.SENIOR,
    )


@pytest.fixture
def sample_search_request() -> SearchRequest:
    return SearchRequest(
        job_description_text=(
            "We are looking for a Senior ML Engineer to join our AI team. "
            "Required: Python, PyTorch, LangChain. 5+ years experience."
        ),
        filters=SearchFilter(),
        top_k=5,
        include_outreach_drafts=True,
    )


@pytest.fixture
def sample_score(sample_candidate: Candidate) -> CandidateScore:
    return CandidateScore(
        candidate_id=sample_candidate.id,
        overall_score=85.0,
        skills_match_score=90.0,
        experience_score=80.0,
        education_score=75.0,
        culture_fit_score=80.0,
        explanation="Strong match overall.",
        strengths=["Python expertise", "LLM experience"],
        gaps=["Kubernetes exposure"],
    )
