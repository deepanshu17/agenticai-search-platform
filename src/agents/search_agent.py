"""Search agent — finds candidate profiles using MCP tools."""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent
from src.mcp.client import MCPClient
from src.models.candidate import Candidate, Education, Skill, WorkExperience
from src.models.job import JobDescription

logger = logging.getLogger(__name__)


def _mcp_to_candidate(raw: dict[str, Any]) -> Candidate:
    """Convert a raw MCP search result dict into a :class:`Candidate`."""
    skills = [Skill(name=s) for s in raw.get("skills", [])]
    exp_years = float(raw.get("years_experience", 0))

    experience: list[WorkExperience] = []
    if raw.get("current_company"):
        experience.append(
            WorkExperience(
                company=raw["current_company"],
                title=raw.get("headline", ""),
                start_date="2020-01",
                is_current=True,
                description=raw.get("summary", ""),
            )
        )

    education: list[Education] = []
    if raw.get("education"):
        education.append(
            Education(
                institution=raw["education"],
                degree="Bachelor's",
                field_of_study="Computer Science",
            )
        )

    from src.models.candidate import SeniorityLevel

    try:
        seniority = SeniorityLevel(raw.get("seniority", "mid"))
    except ValueError:
        seniority = SeniorityLevel.MID

    return Candidate(
        id=raw.get("id", "unknown"),
        name=raw.get("name", "Unknown"),
        headline=raw.get("headline", ""),
        location=raw.get("location", ""),
        summary=raw.get("summary", ""),
        skills=skills,
        experience=experience,
        education=education,
        seniority_level=seniority,
        total_years_experience=exp_years,
        source="mcp_search",
    )


class SearchAgent(BaseAgent):
    """Agent that queries MCP tools to find matching candidates."""

    agent_name = "search_agent"

    def __init__(self, mcp_client: MCPClient) -> None:
        self._mcp = mcp_client

    async def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Search for candidates using the MCP tool layer.

        Expected keys in ``input_data``:
        - ``job``: :class:`JobDescription`
        - ``limit``: int (optional, default 20)

        Returns a dict with ``candidates`` (list of :class:`Candidate`).
        """
        job: JobDescription = input_data["job"]
        limit: int = input_data.get("limit", 20)

        logger.info("SearchAgent: searching for candidates for job=%s", job.id)

        result = await self._mcp.call_tool(
            "search_candidates",
            {
                "skills": job.requirements.required_skills + job.requirements.preferred_skills,
                "seniority": job.seniority_level.value,
                "min_years_experience": job.requirements.min_years_experience,
                "limit": limit,
            },
        )

        raw_candidates: list[dict] = result.get("candidates", [])
        candidates = [_mcp_to_candidate(raw) for raw in raw_candidates]

        logger.info("SearchAgent: found %d candidates", len(candidates))
        return {"candidates": candidates, "total_found": result.get("total_found", len(candidates))}
