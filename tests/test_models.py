"""Tests for Pydantic data models."""

from __future__ import annotations

import pytest

from src.models.candidate import (
    Candidate,
    CandidateScore,
    Education,
    EmploymentType,
    RankedCandidate,
    SeniorityLevel,
    Skill,
    WorkExperience,
    WorkLocation,
)
from src.models.job import JobDescription, JobRequirement, SalaryRange
from src.models.search import (
    AgentTrace,
    CandidateSearchRequest,
    SearchFilter,
    SearchRequest,
    SearchResponse,
    SearchStatus,
)


class TestSkillModel:
    def test_defaults(self):
        skill = Skill(name="Python")
        assert skill.name == "Python"
        assert skill.years_of_experience == 0.0
        assert skill.proficiency == "intermediate"
        assert skill.last_used is None

    def test_with_values(self):
        skill = Skill(name="Go", years_of_experience=3.5, proficiency="advanced", last_used="2024")
        assert skill.years_of_experience == 3.5
        assert skill.proficiency == "advanced"

    def test_negative_experience_rejected(self):
        with pytest.raises(Exception):
            Skill(name="Python", years_of_experience=-1.0)


class TestCandidateModel:
    def test_minimal_candidate(self, sample_candidate: Candidate):
        assert sample_candidate.id == "cand-001"
        assert sample_candidate.name == "Alice Chen"
        assert len(sample_candidate.skills) == 3

    def test_seniority_level(self, sample_candidate: Candidate):
        assert sample_candidate.seniority_level == SeniorityLevel.SENIOR

    def test_candidate_score_bounds(self, sample_score: CandidateScore):
        assert 0 <= sample_score.overall_score <= 100
        assert 0 <= sample_score.skills_match_score <= 100

    def test_ranked_candidate(self, sample_candidate: Candidate, sample_score: CandidateScore):
        ranked = RankedCandidate(rank=1, candidate=sample_candidate, score=sample_score)
        assert ranked.rank == 1
        assert ranked.outreach_draft is None


class TestJobModel:
    def test_job_defaults(self, sample_job: JobDescription):
        assert sample_job.title == "Senior ML Engineer"
        assert sample_job.company == "TechCorp"
        assert len(sample_job.requirements.required_skills) == 3

    def test_salary_range(self):
        sr = SalaryRange(min=100_000, max=150_000, currency="USD")
        assert sr.min == 100_000
        assert sr.max == 150_000
        assert sr.currency == "USD"

    def test_negative_salary_rejected(self):
        with pytest.raises(Exception):
            SalaryRange(min=-1, max=100_000)


class TestSearchModels:
    def test_search_request_defaults(self, sample_search_request: SearchRequest):
        assert sample_search_request.top_k == 5
        assert sample_search_request.include_outreach_drafts is True

    def test_top_k_bounds(self):
        with pytest.raises(Exception):
            SearchRequest(job_description_text="test", top_k=0)
        with pytest.raises(Exception):
            SearchRequest(job_description_text="test", top_k=101)

    def test_search_filter_defaults(self):
        f = SearchFilter()
        assert f.locations == []
        assert f.remote_only is False

    def test_search_response_model(self, sample_job: JobDescription):
        resp = SearchResponse(
            search_id="abc-123",
            status=SearchStatus.COMPLETED,
            job_description=sample_job,
            total_candidates_found=5,
            summary="All good",
        )
        assert resp.search_id == "abc-123"
        assert resp.status == SearchStatus.COMPLETED

    def test_agent_trace(self):
        trace = AgentTrace(
            agent_name="test_agent",
            action="test_action",
            input={"key": "val"},
            output={"result": 42},
            duration_ms=123.4,
        )
        assert trace.agent_name == "test_agent"
        assert trace.duration_ms == 123.4

    def test_candidate_search_request(self):
        req = CandidateSearchRequest(skills=["Python", "Go"], limit=5)
        assert req.limit == 5
        assert "Python" in req.skills
