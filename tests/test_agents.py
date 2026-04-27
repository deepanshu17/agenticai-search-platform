"""Tests for multi-agent system."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.orchestrator import Orchestrator
from src.agents.outreach_agent import OutreachAgent
from src.agents.ranking_agent import RankingAgent
from src.agents.search_agent import SearchAgent, _mcp_to_candidate
from src.agents.skills_agent import SkillsAgent
from src.mcp.client import MCPClient
from src.models.candidate import Candidate, CandidateScore, RankedCandidate, SeniorityLevel
from src.models.search import SearchStatus
from src.pipelines.candidate_evaluator import CandidateEvaluatorPipeline
from src.pipelines.match_scorer import MatchScorerPipeline
from src.utils.fake_llm import FakeLLM


class TestSearchAgent:
    @pytest.fixture
    def agent(self) -> SearchAgent:
        return SearchAgent(MCPClient())

    @pytest.mark.asyncio
    async def test_run_returns_candidates(self, agent: SearchAgent, sample_job):
        result = await agent.run({"job": sample_job, "limit": 5})
        assert "candidates" in result
        assert isinstance(result["candidates"], list)
        assert len(result["candidates"]) > 0

    @pytest.mark.asyncio
    async def test_run_total_found(self, agent: SearchAgent, sample_job):
        result = await agent.run({"job": sample_job})
        assert "total_found" in result
        assert result["total_found"] >= len(result["candidates"])

    def test_mcp_to_candidate_mapping(self):
        raw = {
            "id": "cand-001",
            "name": "Alice Chen",
            "headline": "ML Engineer",
            "location": "SF",
            "skills": ["Python", "PyTorch"],
            "years_experience": 7.0,
            "seniority": "senior",
            "current_company": "OpenAI",
            "education": "MS Stanford",
            "summary": "AI engineer",
        }
        cand = _mcp_to_candidate(raw)
        assert cand.id == "cand-001"
        assert len(cand.skills) == 2
        assert cand.seniority_level == SeniorityLevel.SENIOR
        assert cand.total_years_experience == 7.0

    def test_mcp_to_candidate_invalid_seniority(self):
        raw = {"id": "x", "name": "X", "seniority": "invalid"}
        cand = _mcp_to_candidate(raw)
        assert cand.seniority_level == SeniorityLevel.MID  # fallback

    @pytest.mark.asyncio
    async def test_run_traced(self, agent: SearchAgent, sample_job):
        result, trace = await agent.run_traced("search", {"job": sample_job, "limit": 3})
        assert trace.agent_name == "search_agent"
        assert trace.duration_ms >= 0


class TestSkillsAgent:
    @pytest.fixture
    def agent(self, fake_llm: FakeLLM) -> SkillsAgent:
        return SkillsAgent(MCPClient(), CandidateEvaluatorPipeline(fake_llm))

    @pytest.mark.asyncio
    async def test_run_returns_scores(self, agent: SkillsAgent, sample_candidate, sample_job):
        result = await agent.run({"candidates": [sample_candidate], "job": sample_job})
        assert "scores" in result
        assert len(result["scores"]) == 1
        assert result["scores"][0].candidate_id == sample_candidate.id

    @pytest.mark.asyncio
    async def test_score_bounds(self, agent: SkillsAgent, sample_candidate, sample_job):
        result = await agent.run({"candidates": [sample_candidate], "job": sample_job})
        score = result["scores"][0]
        assert 0 <= score.overall_score <= 100
        assert 0 <= score.skills_match_score <= 100

    @pytest.mark.asyncio
    async def test_multiple_candidates(self, agent: SkillsAgent, sample_job, fake_llm):
        cands = [
            Candidate(id=f"c{i}", name=f"C{i}", seniority_level=SeniorityLevel.MID)
            for i in range(3)
        ]
        result = await agent.run({"candidates": cands, "job": sample_job})
        assert len(result["scores"]) == 3


class TestRankingAgent:
    @pytest.fixture
    def agent(self, fake_llm: FakeLLM) -> RankingAgent:
        return RankingAgent(MCPClient(), MatchScorerPipeline(fake_llm))

    @pytest.mark.asyncio
    async def test_run_returns_ranked(
        self, agent: RankingAgent, sample_candidate, sample_score, sample_job
    ):
        result = await agent.run(
            {"candidates": [sample_candidate], "scores": [sample_score], "job": sample_job}
        )
        assert "ranked_candidates" in result
        rc = result["ranked_candidates"][0]
        assert rc.rank == 1
        assert rc.candidate.id == sample_candidate.id

    @pytest.mark.asyncio
    async def test_market_data_fetched(
        self, agent: RankingAgent, sample_candidate, sample_score, sample_job
    ):
        result = await agent.run(
            {"candidates": [sample_candidate], "scores": [sample_score], "job": sample_job}
        )
        assert "market_data" in result
        assert "salary_range" in result["market_data"]


class TestOutreachAgent:
    @pytest.fixture
    def agent(self) -> OutreachAgent:
        return OutreachAgent(MCPClient())

    @pytest.mark.asyncio
    async def test_drafts_outreach(
        self, agent: OutreachAgent, sample_candidate, sample_score, sample_job
    ):
        rc = RankedCandidate(rank=1, candidate=sample_candidate, score=sample_score)
        result = await agent.run(
            {"ranked_candidates": [rc], "job": sample_job, "top_k": 1}
        )
        updated = result["ranked_candidates"][0]
        assert updated.outreach_draft is not None
        assert len(updated.outreach_draft) > 0

    @pytest.mark.asyncio
    async def test_only_drafts_top_k(
        self, agent: OutreachAgent, sample_candidate, sample_score, sample_job
    ):
        ranked = [
            RankedCandidate(
                rank=i + 1,
                candidate=Candidate(id=f"c{i}", name=f"C{i}", seniority_level=SeniorityLevel.MID),
                score=CandidateScore(
                    candidate_id=f"c{i}",
                    overall_score=80.0 - i,
                    skills_match_score=80.0,
                    experience_score=80.0,
                    education_score=80.0,
                    culture_fit_score=80.0,
                ),
            )
            for i in range(5)
        ]
        result = await agent.run(
            {"ranked_candidates": ranked, "job": sample_job, "top_k": 2}
        )
        updated = result["ranked_candidates"]
        # Only first 2 should have outreach drafts
        assert updated[0].outreach_draft is not None
        assert updated[1].outreach_draft is not None
        # Remaining should not have drafts
        for rc in updated[2:]:
            assert rc.outreach_draft is None


class TestOrchestrator:
    @pytest.fixture
    def orchestrator(self, fake_llm: FakeLLM) -> Orchestrator:
        return Orchestrator(llm=fake_llm)

    @pytest.mark.asyncio
    async def test_search_returns_response(
        self, orchestrator: Orchestrator, sample_search_request
    ):
        response = await orchestrator.search(sample_search_request)
        assert response.search_id
        assert response.status == SearchStatus.COMPLETED
        assert response.job_description is not None
        assert len(response.ranked_candidates) > 0

    @pytest.mark.asyncio
    async def test_search_has_agent_traces(
        self, orchestrator: Orchestrator, sample_search_request
    ):
        response = await orchestrator.search(sample_search_request)
        assert len(response.agent_traces) >= 3  # job_analyzer, search, skills, ranking

    @pytest.mark.asyncio
    async def test_search_top_k_respected(self, orchestrator: Orchestrator):
        from src.models.search import SearchFilter, SearchRequest

        req = SearchRequest(
            job_description_text="Python developer with 2 years experience",
            top_k=2,
        )
        response = await orchestrator.search(req)
        assert len(response.ranked_candidates) <= 2

    @pytest.mark.asyncio
    async def test_search_summary_populated(
        self, orchestrator: Orchestrator, sample_search_request
    ):
        response = await orchestrator.search(sample_search_request)
        assert response.summary
        assert len(response.summary) > 0

    @pytest.mark.asyncio
    async def test_search_duration_tracked(
        self, orchestrator: Orchestrator, sample_search_request
    ):
        response = await orchestrator.search(sample_search_request)
        assert response.duration_ms > 0
