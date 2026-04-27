"""Skills assessment agent — evaluates candidate skills using MCP + LangChain pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.agents.base import BaseAgent
from src.mcp.client import MCPClient
from src.models.candidate import Candidate, CandidateScore
from src.models.job import JobDescription
from src.pipelines.candidate_evaluator import CandidateEvaluatorPipeline

logger = logging.getLogger(__name__)


class SkillsAgent(BaseAgent):
    """Agent that combines MCP skill-match analysis with LangChain evaluation."""

    agent_name = "skills_agent"

    def __init__(self, mcp_client: MCPClient, evaluator_pipeline: CandidateEvaluatorPipeline) -> None:
        self._mcp = mcp_client
        self._evaluator = evaluator_pipeline

    async def _enrich_score_with_mcp(
        self, candidate: Candidate, job: JobDescription
    ) -> dict[str, Any]:
        """Use the MCP analyze_skill_match tool to get objective skill overlap data."""
        try:
            return await self._mcp.call_tool(
                "analyze_skill_match",
                {
                    "candidate_id": candidate.id,
                    "required_skills": job.requirements.required_skills,
                    "preferred_skills": job.requirements.preferred_skills,
                },
            )
        except Exception:
            logger.exception("MCP skill analysis failed for candidate %s", candidate.id)
            return {}

    async def _evaluate_one(
        self, candidate: Candidate, job: JobDescription
    ) -> tuple[CandidateScore, dict[str, Any]]:
        """Run LangChain evaluation and MCP skill match in parallel."""
        llm_score_task = asyncio.create_task(self._evaluator.evaluate(candidate, job))
        mcp_data_task = asyncio.create_task(self._enrich_score_with_mcp(candidate, job))

        llm_score, mcp_data = await asyncio.gather(llm_score_task, mcp_data_task)

        # If MCP provides an objective skills score, override the LLM skills score
        mcp_skills_score = mcp_data.get("combined_score")
        if mcp_skills_score is not None:
            llm_score.skills_match_score = float(mcp_skills_score)
            # Recalculate overall as weighted average
            llm_score.overall_score = round(
                llm_score.skills_match_score * 0.40
                + llm_score.experience_score * 0.35
                + llm_score.education_score * 0.15
                + llm_score.culture_fit_score * 0.10,
                1,
            )

        return llm_score, mcp_data

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Evaluate all candidates against the job description.

        Expected keys in ``input_data``:
        - ``candidates``: list of :class:`Candidate`
        - ``job``: :class:`JobDescription`

        Returns a dict with ``scores`` (list of :class:`CandidateScore`).
        """
        candidates: list[Candidate] = input_data["candidates"]
        job: JobDescription = input_data["job"]

        logger.info("SkillsAgent: evaluating %d candidates for job=%s", len(candidates), job.id)

        tasks = [self._evaluate_one(c, job) for c in candidates]
        results = await asyncio.gather(*tasks)

        scores = [score for score, _ in results]
        logger.info("SkillsAgent: evaluation complete")
        return {"scores": scores}
