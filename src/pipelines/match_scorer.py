"""LangChain pipeline for scoring and ranking candidates."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.models.candidate import Candidate, CandidateScore, RankedCandidate
from src.models.job import JobDescription

logger = logging.getLogger(__name__)

RANK_SYSTEM_PROMPT = """\
You are a talent acquisition director. Given a ranked list of scored candidates and a job description, \
provide a brief ranking rationale for each candidate and suggest the top candidates to prioritise.

Return a JSON object with this exact schema:
{{
  "ranking_rationale": "<Overall explanation of how candidates were ranked>",
  "candidate_notes": [
    {{
      "candidate_id": "<id>",
      "rank_note": "<1 sentence note about this candidate's ranking>",
      "priority": "<high|medium|low>"
    }}
  ]
}}
"""

RANK_HUMAN_PROMPT = """\
Job: {job_title} ({seniority_level}) at {company}
Required Skills: {required_skills}

Scored Candidates (sorted by overall_score desc):
{scored_candidates}
"""


def _format_scored_candidates(pairs: list[tuple[Candidate, CandidateScore]]) -> str:
    lines = []
    for candidate, score in pairs:
        lines.append(
            f"- ID={candidate.id} Name={candidate.name} "
            f"Overall={score.overall_score:.1f} Skills={score.skills_match_score:.1f} "
            f"Exp={score.experience_score:.1f}"
        )
    return "\n".join(lines)


class MatchScorerPipeline:
    """Pipeline that takes scored candidates and produces a final ranked list."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self._chain = self._build_chain()

    def _build_chain(self) -> Any:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", RANK_SYSTEM_PROMPT),
                ("human", RANK_HUMAN_PROMPT),
            ]
        )
        return prompt | self._llm | JsonOutputParser()

    async def rank(
        self,
        candidates: list[Candidate],
        scores: list[CandidateScore],
        job: JobDescription,
    ) -> list[RankedCandidate]:
        """Rank candidates by their scores, optionally enriched with LLM rationale.

        Args:
            candidates: Candidate objects.
            scores: Matching CandidateScore objects (same order as candidates).
            job: The target job description.

        Returns:
            List of :class:`RankedCandidate` sorted from highest to lowest score.
        """
        if len(candidates) != len(scores):
            raise ValueError("candidates and scores must have the same length")

        # Sort by overall score descending
        pairs = sorted(zip(candidates, scores), key=lambda p: p[1].overall_score, reverse=True)

        # Build ranked list without LLM first (deterministic)
        ranked: list[RankedCandidate] = [
            RankedCandidate(rank=idx + 1, candidate=cand, score=score)
            for idx, (cand, score) in enumerate(pairs)
        ]

        # Optionally enrich with LLM rationale
        try:
            raw: dict[str, Any] = await self._chain.ainvoke(
                {
                    "job_title": job.title,
                    "seniority_level": job.seniority_level.value,
                    "company": job.company,
                    "required_skills": ", ".join(job.requirements.required_skills),
                    "scored_candidates": _format_scored_candidates(pairs),
                }
            )
            notes_by_id: dict[str, dict] = {
                n["candidate_id"]: n for n in raw.get("candidate_notes", [])
            }
            for rc in ranked:
                note = notes_by_id.get(rc.candidate.id, {})
                if note.get("rank_note"):
                    # Append rank note to the score explanation
                    rc.score.explanation = (
                        rc.score.explanation + f" [{note['rank_note']}]"
                        if rc.score.explanation
                        else note["rank_note"]
                    )
        except Exception:
            logger.exception("LLM ranking rationale failed; using score-based ranking only")

        return ranked
