"""Outreach agent — drafts personalised recruiting messages via MCP."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.agents.base import BaseAgent
from src.mcp.client import MCPClient
from src.models.candidate import RankedCandidate
from src.models.job import JobDescription

logger = logging.getLogger(__name__)


class OutreachAgent(BaseAgent):
    """Agent that drafts personalised outreach emails for top-ranked candidates."""

    agent_name = "outreach_agent"

    def __init__(self, mcp_client: MCPClient) -> None:
        self._mcp = mcp_client

    async def _draft_one(
        self,
        ranked_candidate: RankedCandidate,
        job: JobDescription,
        recruiter_name: str,
    ) -> str:
        try:
            selling_points = []
            if job.salary_range:
                selling_points.append(
                    f"Competitive salary: ${job.salary_range.min:,.0f}–${job.salary_range.max:,.0f}"
                )
            if job.requirements.required_skills:
                sp = ", ".join(job.requirements.required_skills[:3])
                selling_points.append(f"Work with cutting-edge tech: {sp}")
            if job.benefits:
                selling_points.append(job.benefits[0])

            result = await self._mcp.call_tool(
                "draft_outreach_message",
                {
                    "candidate_id": ranked_candidate.candidate.id,
                    "job_title": job.title,
                    "company_name": job.company,
                    "recruiter_name": recruiter_name,
                    "key_selling_points": selling_points[:3],
                },
            )
            return result.get("body", "")
        except Exception:
            logger.exception(
                "Outreach draft failed for candidate %s", ranked_candidate.candidate.id
            )
            return ""

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Draft outreach messages for top candidates.

        Expected keys in ``input_data``:
        - ``ranked_candidates``: list of :class:`RankedCandidate`
        - ``job``: :class:`JobDescription`
        - ``top_k``: int – how many candidates to draft messages for (default 5)
        - ``recruiter_name``: str (optional)

        Returns a dict with the same ``ranked_candidates`` list, each
        enriched with ``outreach_draft``.
        """
        ranked: list[RankedCandidate] = input_data["ranked_candidates"]
        job: JobDescription = input_data["job"]
        top_k: int = input_data.get("top_k", 5)
        recruiter_name: str = input_data.get("recruiter_name", "The Recruiting Team")

        to_draft = ranked[:top_k]
        logger.info(
            "OutreachAgent: drafting messages for top-%d candidates (job=%s)",
            len(to_draft),
            job.id,
        )

        drafts = await asyncio.gather(
            *[self._draft_one(rc, job, recruiter_name) for rc in to_draft]
        )

        for rc, draft in zip(to_draft, drafts):
            rc.outreach_draft = draft or None

        return {"ranked_candidates": ranked}
