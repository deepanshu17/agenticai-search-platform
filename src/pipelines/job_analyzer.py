"""LangChain pipeline for parsing and analyzing job descriptions."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from pydantic import BaseModel, Field

from src.models.job import JobDescription, JobRequirement, SalaryRange, SeniorityLevel

logger = logging.getLogger(__name__)

JOB_ANALYSIS_SYSTEM_PROMPT = """\
You are an expert recruiter and talent acquisition specialist. \
Analyze the provided job description and extract structured information.

Return a JSON object with this exact schema:
{{
  "title": "<job title>",
  "company": "<company name or 'Unknown'>",
  "department": "<department or team>",
  "seniority_level": "<one of: intern|junior|mid|senior|staff|principal|director|vp|c_level>",
  "required_skills": ["<skill1>", "<skill2>"],
  "preferred_skills": ["<skill1>", "<skill2>"],
  "min_years_experience": <number>,
  "max_years_experience": <number or null>,
  "required_education": "<education requirement or null>",
  "certifications": ["<cert1>"],
  "work_location": "<one of: remote|hybrid|on_site>",
  "location": "<city, state/country or 'Not specified'>",
  "employment_type": "<one of: full_time|part_time|contract|freelance>",
  "salary_min": <number or null>,
  "salary_max": <number or null>,
  "salary_currency": "USD",
  "benefits": ["<benefit1>"],
  "keywords": ["<keyword1>"],
  "summary": "<2-3 sentence summary of the role>"
}}
"""

JOB_ANALYSIS_HUMAN_PROMPT = "Job Description:\n\n{job_description_text}"


class ParsedJobFields(BaseModel):
    title: str = "Unknown Title"
    company: str = "Unknown"
    department: str = ""
    seniority_level: str = "mid"
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_years_experience: float = 0.0
    max_years_experience: float | None = None
    required_education: str | None = None
    certifications: list[str] = Field(default_factory=list)
    work_location: str = "hybrid"
    location: str = ""
    employment_type: str = "full_time"
    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str = "USD"
    benefits: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    summary: str = ""


def _build_job_description(fields: dict[str, Any], original_text: str, job_id: str) -> JobDescription:
    """Convert raw LLM output dict into a JobDescription model."""
    parsed = ParsedJobFields.model_validate(fields)

    salary_range: SalaryRange | None = None
    if parsed.salary_min is not None or parsed.salary_max is not None:
        salary_range = SalaryRange(
            min=parsed.salary_min or 0,
            max=parsed.salary_max or 0,
            currency=parsed.salary_currency,
        )

    seniority: SeniorityLevel
    try:
        seniority = SeniorityLevel(parsed.seniority_level)
    except ValueError:
        seniority = SeniorityLevel.MID

    return JobDescription(
        id=job_id,
        title=parsed.title,
        company=parsed.company,
        department=parsed.department,
        description=original_text,
        requirements=JobRequirement(
            required_skills=parsed.required_skills,
            preferred_skills=parsed.preferred_skills,
            min_years_experience=parsed.min_years_experience,
            max_years_experience=parsed.max_years_experience,
            required_education=parsed.required_education,
            certifications=parsed.certifications,
        ),
        seniority_level=seniority,
        work_location=parsed.work_location,
        location=parsed.location,
        salary_range=salary_range,
        benefits=parsed.benefits,
        keywords=parsed.keywords,
        parsed_requirements={"summary": parsed.summary},
    )


class JobAnalyzerPipeline:
    """LangChain LCEL pipeline that parses raw JD text into a structured JobDescription."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self._chain = self._build_chain()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chain(self) -> Any:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", JOB_ANALYSIS_SYSTEM_PROMPT),
                ("human", JOB_ANALYSIS_HUMAN_PROMPT),
            ]
        )
        parser = JsonOutputParser()
        return prompt | self._llm | parser

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(self, job_description_text: str, job_id: str = "job-001") -> JobDescription:
        """Parse and analyze a raw job description string.

        Args:
            job_description_text: The raw job description text.
            job_id: Unique identifier for this job.

        Returns:
            A fully populated :class:`JobDescription` instance.
        """
        logger.info("Analyzing job description (id=%s)", job_id)
        try:
            raw: dict[str, Any] = await self._chain.ainvoke(
                {"job_description_text": job_description_text}
            )
        except Exception:
            logger.exception("LLM call failed during job analysis; using defaults")
            raw = {}

        return _build_job_description(raw, job_description_text, job_id)
