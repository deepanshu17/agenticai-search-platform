"""Tests for LangChain pipelines (using FakeLLM)."""

from __future__ import annotations

import pytest

from src.models.candidate import SeniorityLevel
from src.models.job import JobDescription, JobRequirement
from src.pipelines.candidate_evaluator import CandidateEvaluatorPipeline
from src.pipelines.job_analyzer import JobAnalyzerPipeline
from src.pipelines.match_scorer import MatchScorerPipeline
from src.utils.fake_llm import FakeLLM


class TestJobAnalyzerPipeline:
    @pytest.fixture
    def pipeline(self, fake_llm: FakeLLM) -> JobAnalyzerPipeline:
        return JobAnalyzerPipeline(fake_llm)

    @pytest.mark.asyncio
    async def test_returns_job_description(self, pipeline: JobAnalyzerPipeline):
        jd_text = (
            "Senior ML Engineer at TechCorp. Required: Python, PyTorch. 5+ years."
        )
        result = await pipeline.analyze(jd_text, job_id="test-001")
        assert result.id == "test-001"
        assert result.title  # non-empty
        assert isinstance(result.requirements.required_skills, list)

    @pytest.mark.asyncio
    async def test_preserves_original_text(self, pipeline: JobAnalyzerPipeline):
        jd_text = "Full-stack engineer needed."
        result = await pipeline.analyze(jd_text)
        assert result.description == jd_text

    @pytest.mark.asyncio
    async def test_default_job_id(self, pipeline: JobAnalyzerPipeline):
        result = await pipeline.analyze("Some job description")
        assert result.id == "job-001"

    @pytest.mark.asyncio
    async def test_handles_llm_failure_gracefully(self):
        """Even if LLM fails, pipeline should return a valid JobDescription."""
        from unittest.mock import AsyncMock, MagicMock

        bad_llm = MagicMock()
        bad_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))
        # Patch the chain to fail
        pipeline = JobAnalyzerPipeline.__new__(JobAnalyzerPipeline)
        pipeline._llm = bad_llm
        # Build a broken chain
        pipeline._chain = MagicMock()
        pipeline._chain.ainvoke = AsyncMock(side_effect=Exception("chain error"))

        result = await pipeline.analyze("some text", job_id="err-001")
        assert result.id == "err-001"
        assert result.description == "some text"


class TestCandidateEvaluatorPipeline:
    @pytest.fixture
    def pipeline(self, fake_llm: FakeLLM) -> CandidateEvaluatorPipeline:
        return CandidateEvaluatorPipeline(fake_llm)

    @pytest.mark.asyncio
    async def test_evaluate_returns_score(
        self, pipeline: CandidateEvaluatorPipeline, sample_candidate, sample_job
    ):
        score = await pipeline.evaluate(sample_candidate, sample_job)
        assert score.candidate_id == sample_candidate.id
        assert 0 <= score.overall_score <= 100
        assert 0 <= score.skills_match_score <= 100

    @pytest.mark.asyncio
    async def test_evaluate_batch(
        self, pipeline: CandidateEvaluatorPipeline, sample_candidate, sample_job
    ):
        candidates = [sample_candidate]
        scores = await pipeline.evaluate_batch(candidates, sample_job)
        assert len(scores) == 1
        assert scores[0].candidate_id == sample_candidate.id

    @pytest.mark.asyncio
    async def test_evaluate_handles_failure(self, sample_candidate, sample_job):
        from unittest.mock import AsyncMock, MagicMock

        pipeline = CandidateEvaluatorPipeline.__new__(CandidateEvaluatorPipeline)
        pipeline._chain = MagicMock()
        pipeline._chain.ainvoke = AsyncMock(side_effect=Exception("LLM failed"))

        score = await pipeline.evaluate(sample_candidate, sample_job)
        # Should return default 50.0 scores
        assert score.overall_score == 50.0
        assert score.candidate_id == sample_candidate.id


class TestMatchScorerPipeline:
    @pytest.fixture
    def pipeline(self, fake_llm: FakeLLM) -> MatchScorerPipeline:
        return MatchScorerPipeline(fake_llm)

    @pytest.mark.asyncio
    async def test_rank_returns_ranked_list(
        self, pipeline: MatchScorerPipeline, sample_candidate, sample_score, sample_job
    ):
        ranked = await pipeline.rank([sample_candidate], [sample_score], sample_job)
        assert len(ranked) == 1
        assert ranked[0].rank == 1
        assert ranked[0].candidate.id == sample_candidate.id

    @pytest.mark.asyncio
    async def test_rank_sorted_by_score(
        self, pipeline: MatchScorerPipeline, sample_job, fake_llm
    ):
        from src.models.candidate import Candidate, CandidateScore, SeniorityLevel

        cand_a = Candidate(id="a", name="A", seniority_level=SeniorityLevel.MID)
        cand_b = Candidate(id="b", name="B", seniority_level=SeniorityLevel.MID)
        score_a = CandidateScore(
            candidate_id="a",
            overall_score=90.0,
            skills_match_score=90.0,
            experience_score=90.0,
            education_score=90.0,
            culture_fit_score=90.0,
        )
        score_b = CandidateScore(
            candidate_id="b",
            overall_score=60.0,
            skills_match_score=60.0,
            experience_score=60.0,
            education_score=60.0,
            culture_fit_score=60.0,
        )
        ranked = await pipeline.rank([cand_b, cand_a], [score_b, score_a], sample_job)
        assert ranked[0].candidate.id == "a"  # Higher score first
        assert ranked[1].candidate.id == "b"

    @pytest.mark.asyncio
    async def test_rank_length_mismatch_raises(
        self, pipeline: MatchScorerPipeline, sample_candidate, sample_job
    ):
        with pytest.raises(ValueError):
            await pipeline.rank([sample_candidate], [], sample_job)
