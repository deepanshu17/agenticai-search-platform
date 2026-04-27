"""LangChain pipeline for evaluating candidates against a job description."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.models.candidate import Candidate, CandidateScore
from src.models.job import JobDescription

logger = logging.getLogger(__name__)

CANDIDATE_EVAL_SYSTEM_PROMPT = """\
You are a senior technical recruiter who objectively evaluates candidate profiles. \
Given a job description and a candidate profile, score the candidate across multiple dimensions.

Return a JSON object with this exact schema:
{{
  "overall_score": <0-100 float>,
  "skills_match_score": <0-100 float>,
  "experience_score": <0-100 float>,
  "education_score": <0-100 float>,
  "culture_fit_score": <0-100 float>,
  "explanation": "<2-3 sentence explanation of the overall assessment>",
  "strengths": ["<strength1>", "<strength2>"],
  "gaps": ["<gap1>", "<gap2>"]
}}

Scoring guidelines:
- skills_match_score: How well do their skills match required + preferred skills?
- experience_score: Is their experience level and domain appropriate?
- education_score: Does their education meet requirements? (if no requirement, score 75)
- culture_fit_score: Based on career trajectory, communication signals, and alignment.
- overall_score: Weighted average (skills 40%, experience 35%, education 15%, culture 10%).
"""

CANDIDATE_EVAL_HUMAN_PROMPT = """\
Job Description:
Title: {job_title}
Required Skills: {required_skills}
Preferred Skills: {preferred_skills}
Min Years Experience: {min_years_experience}
Seniority Level: {seniority_level}

Candidate Profile:
Name: {candidate_name}
Headline: {candidate_headline}
Total Years Experience: {total_years_experience}
Seniority: {candidate_seniority}
Skills: {candidate_skills}
Recent Experience: {recent_experience}
Education: {candidate_education}
"""


def _format_skills(candidate: Candidate) -> str:
    return ", ".join(s.name for s in candidate.skills[:20]) if candidate.skills else "Not listed"


def _format_recent_experience(candidate: Candidate, max_entries: int = 3) -> str:
    if not candidate.experience:
        return "No experience listed"
    parts = []
    for exp in candidate.experience[:max_entries]:
        current_tag = " (current)" if exp.is_current else ""
        parts.append(f"{exp.title} at {exp.company}{current_tag}: {exp.description[:100]}")
    return " | ".join(parts)


def _format_education(candidate: Candidate) -> str:
    if not candidate.education:
        return "Not listed"
    return "; ".join(
        f"{e.degree} in {e.field_of_study} from {e.institution}" for e in candidate.education
    )


class CandidateEvaluatorPipeline:
    """LangChain pipeline that scores a candidate against a job description."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self._chain = self._build_chain()

    def _build_chain(self) -> Any:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CANDIDATE_EVAL_SYSTEM_PROMPT),
                ("human", CANDIDATE_EVAL_HUMAN_PROMPT),
            ]
        )
        return prompt | self._llm | JsonOutputParser()

    async def evaluate(self, candidate: Candidate, job: JobDescription) -> CandidateScore:
        """Evaluate a single candidate against a job description.

        Args:
            candidate: The candidate to evaluate.
            job: The target job description.

        Returns:
            A :class:`CandidateScore` with dimension scores and explanation.
        """
        logger.debug("Evaluating candidate %s for job %s", candidate.id, job.id)
        try:
            raw: dict[str, Any] = await self._chain.ainvoke(
                {
                    "job_title": job.title,
                    "required_skills": ", ".join(job.requirements.required_skills),
                    "preferred_skills": ", ".join(job.requirements.preferred_skills),
                    "min_years_experience": job.requirements.min_years_experience,
                    "seniority_level": job.seniority_level.value,
                    "candidate_name": candidate.name,
                    "candidate_headline": candidate.headline,
                    "total_years_experience": candidate.total_years_experience,
                    "candidate_seniority": candidate.seniority_level.value,
                    "candidate_skills": _format_skills(candidate),
                    "recent_experience": _format_recent_experience(candidate),
                    "candidate_education": _format_education(candidate),
                }
            )
        except Exception:
            logger.exception("LLM evaluation failed for candidate %s; using defaults", candidate.id)
            raw = {}

        return CandidateScore(
            candidate_id=candidate.id,
            overall_score=float(raw.get("overall_score", 50.0)),
            skills_match_score=float(raw.get("skills_match_score", 50.0)),
            experience_score=float(raw.get("experience_score", 50.0)),
            education_score=float(raw.get("education_score", 50.0)),
            culture_fit_score=float(raw.get("culture_fit_score", 50.0)),
            explanation=raw.get("explanation", ""),
            strengths=raw.get("strengths", []),
            gaps=raw.get("gaps", []),
        )

    async def evaluate_batch(
        self, candidates: list[Candidate], job: JobDescription
    ) -> list[CandidateScore]:
        """Evaluate multiple candidates concurrently.

        Args:
            candidates: List of candidates to evaluate.
            job: The target job description.

        Returns:
            List of :class:`CandidateScore` objects in the same order.
        """
        import asyncio

        tasks = [self.evaluate(c, job) for c in candidates]
        return list(await asyncio.gather(*tasks))
