"""Fake LLM for use when OPENAI_API_KEY is not set.

Returns deterministic JSON responses that satisfy the pipeline output parsers,
allowing the service to run in demo/test mode without real API credentials.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


_JOB_ANALYSIS_DEFAULT = {
    "title": "Software Engineer",
    "company": "Unknown",
    "department": "Engineering",
    "seniority_level": "mid",
    "required_skills": ["Python", "SQL"],
    "preferred_skills": ["Docker", "Kubernetes"],
    "min_years_experience": 3.0,
    "max_years_experience": None,
    "required_education": None,
    "certifications": [],
    "work_location": "hybrid",
    "location": "Not specified",
    "employment_type": "full_time",
    "salary_min": None,
    "salary_max": None,
    "salary_currency": "USD",
    "benefits": [],
    "keywords": ["engineering", "software"],
    "summary": "A software engineering role requiring Python and SQL skills.",
}

_CANDIDATE_EVAL_DEFAULT = {
    "overall_score": 72.0,
    "skills_match_score": 75.0,
    "experience_score": 70.0,
    "education_score": 75.0,
    "culture_fit_score": 65.0,
    "explanation": "Candidate meets most requirements with a strong skills background.",
    "strengths": ["Relevant technical skills", "Appropriate experience level"],
    "gaps": ["Limited preferred skill coverage"],
}

_RANK_DEFAULT = {
    "ranking_rationale": "Candidates ranked by overall fit score.",
    "candidate_notes": [],
}


class FakeLLM(BaseChatModel):
    """A deterministic fake LLM that returns valid JSON for each pipeline stage."""

    model_name: str = Field(default="fake-llm")

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response_json = self._pick_response(messages)
        message = AIMessage(content=json.dumps(response_json))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _pick_response(self, messages: List[BaseMessage]) -> dict:
        """Choose a response template based on the system prompt content."""
        system_content = ""
        for m in messages:
            if hasattr(m, "content") and "recruiter" in str(m.content).lower():
                if "score" in str(m.content).lower() or "evaluat" in str(m.content).lower():
                    return _CANDIDATE_EVAL_DEFAULT
                if "rank" in str(m.content).lower() or "prioritis" in str(m.content).lower():
                    return _RANK_DEFAULT
            if hasattr(m, "content") and "job description" in str(m.content).lower():
                return _JOB_ANALYSIS_DEFAULT

        # Default fallback
        return _JOB_ANALYSIS_DEFAULT
