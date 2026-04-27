"""Multi-agent orchestrator — coordinates all recruiting agents."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from src.agents.base import BaseAgent
from src.agents.outreach_agent import OutreachAgent
from src.agents.ranking_agent import RankingAgent
from src.agents.search_agent import SearchAgent
from src.agents.skills_agent import SkillsAgent
from src.mcp.client import MCPClient
from src.models.candidate import RankedCandidate
from src.models.search import AgentTrace, SearchRequest, SearchResponse, SearchStatus
from src.pipelines.candidate_evaluator import CandidateEvaluatorPipeline
from src.pipelines.job_analyzer import JobAnalyzerPipeline
from src.pipelines.match_scorer import MatchScorerPipeline

logger = logging.getLogger(__name__)


class Orchestrator:
    """Coordinates the full multi-agent talent search pipeline.

    Pipeline stages
    ---------------
    1. **JobAnalyzerPipeline** – parse the raw JD text into a structured :class:`JobDescription`.
    2. **SearchAgent** – use MCP tools to retrieve matching candidates.
    3. **SkillsAgent** – evaluate each candidate (LLM + MCP skill analysis).
    4. **RankingAgent** – rank candidates and attach market salary data.
    5. **OutreachAgent** – draft personalised outreach for the top-K candidates.
    """

    def __init__(
        self,
        llm: Any,
        mcp_client: MCPClient | None = None,
    ) -> None:
        self._llm = llm
        self._mcp = mcp_client or MCPClient()

        # Pipelines
        self._job_analyzer = JobAnalyzerPipeline(llm)
        self._evaluator = CandidateEvaluatorPipeline(llm)
        self._scorer = MatchScorerPipeline(llm)

        # Agents
        self._search_agent = SearchAgent(self._mcp)
        self._skills_agent = SkillsAgent(self._mcp, self._evaluator)
        self._ranking_agent = RankingAgent(self._mcp, self._scorer)
        self._outreach_agent = OutreachAgent(self._mcp)

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Execute the full multi-agent search for a given :class:`SearchRequest`.

        Args:
            request: A search request containing a raw job description and filters.

        Returns:
            A :class:`SearchResponse` with ranked candidates and agent traces.
        """
        search_id = str(uuid.uuid4())
        traces: list[AgentTrace] = []
        wall_start = time.monotonic()

        logger.info("Orchestrator: starting search (id=%s)", search_id)

        try:
            # -------------------------------------------------------------------
            # Stage 1: Analyse job description
            # -------------------------------------------------------------------
            t0 = time.monotonic()
            job = await self._job_analyzer.analyze(
                request.job_description_text, job_id=search_id
            )
            traces.append(
                AgentTrace(
                    agent_name="job_analyzer_pipeline",
                    action="analyze_job_description",
                    input={"text_length": len(request.job_description_text)},
                    output={
                        "title": job.title,
                        "required_skills": job.requirements.required_skills,
                        "seniority_level": job.seniority_level.value,
                    },
                    duration_ms=(time.monotonic() - t0) * 1000,
                )
            )
            logger.info("Orchestrator: job analyzed – title=%s", job.title)

            # -------------------------------------------------------------------
            # Stage 2: Search for candidates
            # -------------------------------------------------------------------
            _, search_trace = await self._search_agent.run_traced(
                "search_candidates",
                {"job": job, "limit": request.filters.__class__.__name__ and 20},
            )
            search_result = await self._search_agent.run({"job": job, "limit": 20})
            candidates = search_result["candidates"]
            traces.append(search_trace)
            logger.info("Orchestrator: %d candidates found", len(candidates))

            if not candidates:
                return SearchResponse(
                    search_id=search_id,
                    status=SearchStatus.COMPLETED,
                    job_description=job,
                    ranked_candidates=[],
                    total_candidates_found=0,
                    agent_traces=traces,
                    summary="No candidates found matching the job description.",
                    duration_ms=(time.monotonic() - wall_start) * 1000,
                )

            # -------------------------------------------------------------------
            # Stage 3: Skills assessment
            # -------------------------------------------------------------------
            _, skills_trace = await self._skills_agent.run_traced(
                "evaluate_candidates",
                {"candidates": candidates, "job": job},
            )
            skills_result = await self._skills_agent.run({"candidates": candidates, "job": job})
            scores = skills_result["scores"]
            traces.append(skills_trace)

            # -------------------------------------------------------------------
            # Stage 4: Rank candidates
            # -------------------------------------------------------------------
            _, rank_trace = await self._ranking_agent.run_traced(
                "rank_candidates",
                {"candidates": candidates, "scores": scores, "job": job},
            )
            ranking_result = await self._ranking_agent.run(
                {"candidates": candidates, "scores": scores, "job": job}
            )
            ranked: list[RankedCandidate] = ranking_result["ranked_candidates"]
            market_data = ranking_result.get("market_data", {})
            traces.append(rank_trace)

            # Trim to requested top_k
            ranked = ranked[: request.top_k]

            # -------------------------------------------------------------------
            # Stage 5: Draft outreach messages (optional)
            # -------------------------------------------------------------------
            if request.include_outreach_drafts:
                _, outreach_trace = await self._outreach_agent.run_traced(
                    "draft_outreach",
                    {"ranked_candidates": ranked, "job": job, "top_k": min(5, request.top_k)},
                )
                outreach_result = await self._outreach_agent.run(
                    {"ranked_candidates": ranked, "job": job, "top_k": min(5, request.top_k)}
                )
                ranked = outreach_result["ranked_candidates"]
                traces.append(outreach_trace)

            # -------------------------------------------------------------------
            # Build response summary
            # -------------------------------------------------------------------
            top_names = ", ".join(rc.candidate.name for rc in ranked[:3])
            salary_info = ""
            if market_data.get("salary_range"):
                sr = market_data["salary_range"]
                salary_info = (
                    f" Market salary range: ${sr.get('min', 0):,.0f}–${sr.get('max', 0):,.0f}."
                )

            summary = (
                f"Found {len(candidates)} candidates; top {len(ranked)} after filtering. "
                f"Top candidates: {top_names}.{salary_info}"
            )

            total_ms = (time.monotonic() - wall_start) * 1000
            logger.info("Orchestrator: search complete in %.0fms", total_ms)

            return SearchResponse(
                search_id=search_id,
                status=SearchStatus.COMPLETED,
                job_description=job,
                ranked_candidates=ranked,
                total_candidates_found=len(candidates),
                agent_traces=traces,
                summary=summary,
                duration_ms=total_ms,
            )

        except Exception:
            logger.exception("Orchestrator: search failed (id=%s)", search_id)
            return SearchResponse(
                search_id=search_id,
                status=SearchStatus.FAILED,
                agent_traces=traces,
                summary="Search failed due to an internal error.",
                duration_ms=(time.monotonic() - wall_start) * 1000,
            )
