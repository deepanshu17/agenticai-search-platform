"""Ranking agent — sorts candidates and adds market context."""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent
from src.mcp.client import MCPClient
from src.models.candidate import Candidate, CandidateScore, RankedCandidate
from src.models.job import JobDescription
from src.pipelines.match_scorer import MatchScorerPipeline

logger = logging.getLogger(__name__)


class RankingAgent(BaseAgent):
    """Agent that ranks candidates and enriches results with market salary data."""

    agent_name = "ranking_agent"

    def __init__(self, mcp_client: MCPClient, scorer_pipeline: MatchScorerPipeline) -> None:
        self._mcp = mcp_client
        self._scorer = scorer_pipeline

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Rank scored candidates and attach market salary context.

        Expected keys in ``input_data``:
        - ``candidates``: list of :class:`Candidate`
        - ``scores``: list of :class:`CandidateScore`
        - ``job``: :class:`JobDescription`

        Returns a dict with ``ranked_candidates`` and ``market_data``.
        """
        candidates: list[Candidate] = input_data["candidates"]
        scores: list[CandidateScore] = input_data["scores"]
        job: JobDescription = input_data["job"]

        logger.info("RankingAgent: ranking %d candidates for job=%s", len(candidates), job.id)

        ranked = await self._scorer.rank(candidates, scores, job)

        # Fetch market salary data via MCP
        market_data: dict[str, Any] = {}
        try:
            market_data = await self._mcp.call_tool(
                "get_market_salary_data",
                {
                    "job_title": job.title,
                    "location": job.location or "United States",
                    "seniority": job.seniority_level.value,
                },
            )
        except Exception:
            logger.exception("Failed to fetch market salary data")

        logger.info("RankingAgent: ranking complete, top candidate=%s", ranked[0].candidate.id if ranked else "none")
        return {"ranked_candidates": ranked, "market_data": market_data}
