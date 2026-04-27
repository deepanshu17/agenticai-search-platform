"""MCP server exposing recruiting tools for AI agents.

The server exposes the following tools:
- search_candidates      – fuzzy search a mock candidate database
- get_candidate_profile  – retrieve full profile by ID
- analyze_skill_match    – compute skill overlap between candidate and JD
- get_market_salary_data – return market salary benchmarks for a role/location
- draft_outreach_message – generate a personalised outreach email stub
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import string
import uuid
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock candidate database (deterministic seed for reproducibility)
# ---------------------------------------------------------------------------

_MOCK_CANDIDATES: list[dict[str, Any]] = [
    {
        "id": "cand-001",
        "name": "Alice Chen",
        "headline": "Senior ML Engineer | PyTorch, LLMs, MLflow",
        "location": "San Francisco, CA",
        "skills": ["Python", "PyTorch", "LangChain", "MLflow", "Kubernetes", "SQL"],
        "years_experience": 7.0,
        "seniority": "senior",
        "current_company": "OpenAI",
        "education": "MS Computer Science, Stanford",
        "summary": "ML engineer focused on large language models and production ML systems.",
    },
    {
        "id": "cand-002",
        "name": "Brian Kapoor",
        "headline": "Staff Software Engineer | Distributed Systems, Go, Rust",
        "location": "New York, NY",
        "skills": ["Go", "Rust", "Kubernetes", "gRPC", "PostgreSQL", "Redis"],
        "years_experience": 10.0,
        "seniority": "staff",
        "current_company": "Stripe",
        "education": "BS Computer Science, MIT",
        "summary": "Systems engineer with deep expertise in high-throughput distributed architectures.",
    },
    {
        "id": "cand-003",
        "name": "Carla Martinez",
        "headline": "Full-Stack Engineer | React, Node.js, TypeScript",
        "location": "Austin, TX",
        "skills": ["TypeScript", "React", "Node.js", "GraphQL", "PostgreSQL", "Docker"],
        "years_experience": 4.5,
        "seniority": "mid",
        "current_company": "Shopify",
        "education": "BS Software Engineering, UT Austin",
        "summary": "Product-focused engineer who ships polished full-stack features at scale.",
    },
    {
        "id": "cand-004",
        "name": "David Park",
        "headline": "Data Scientist | NLP, Python, Spark",
        "location": "Seattle, WA",
        "skills": ["Python", "Spark", "NLP", "scikit-learn", "SQL", "Airflow"],
        "years_experience": 6.0,
        "seniority": "senior",
        "current_company": "Amazon",
        "education": "PhD Statistics, University of Washington",
        "summary": "Data scientist specialising in NLP pipelines and large-scale data processing.",
    },
    {
        "id": "cand-005",
        "name": "Eva Johansson",
        "headline": "DevOps / Platform Engineer | AWS, Terraform, Kubernetes",
        "location": "Remote",
        "skills": ["AWS", "Terraform", "Kubernetes", "Python", "Helm", "GitHub Actions"],
        "years_experience": 8.0,
        "seniority": "senior",
        "current_company": "HashiCorp",
        "education": "BS Information Systems, KTH Royal Institute",
        "summary": "Platform engineer who automates everything and champions zero-downtime deployments.",
    },
    {
        "id": "cand-006",
        "name": "Frank Liu",
        "headline": "AI/ML Engineer | Transformers, RLHF, Fine-tuning",
        "location": "San Francisco, CA",
        "skills": ["Python", "Transformers", "RLHF", "LangChain", "FastAPI", "CUDA"],
        "years_experience": 5.0,
        "seniority": "mid",
        "current_company": "Anthropic",
        "education": "MS AI, Carnegie Mellon",
        "summary": "AI engineer specialising in fine-tuning and aligning large language models.",
    },
    {
        "id": "cand-007",
        "name": "Grace O'Brien",
        "headline": "Backend Engineer | Java, Spring Boot, Microservices",
        "location": "Chicago, IL",
        "skills": ["Java", "Spring Boot", "Kafka", "PostgreSQL", "Docker", "AWS"],
        "years_experience": 9.0,
        "seniority": "senior",
        "current_company": "JPMorgan Chase",
        "education": "BS Computer Science, UIUC",
        "summary": "Seasoned backend engineer building reliable microservices for fintech at scale.",
    },
    {
        "id": "cand-008",
        "name": "Hiroshi Tanaka",
        "headline": "Junior SWE | Python, React, learning ML",
        "location": "Boston, MA",
        "skills": ["Python", "React", "JavaScript", "SQL", "Git"],
        "years_experience": 1.5,
        "seniority": "junior",
        "current_company": "Startup Inc.",
        "education": "BS Computer Science, Northeastern",
        "summary": "Early-career engineer eager to apply ML in real-world products.",
    },
]


def _skill_score(candidate_skills: list[str], target_skills: list[str]) -> float:
    """Return a 0-100 overlap score between two skill sets."""
    if not target_skills:
        return 75.0
    cs = {s.lower() for s in candidate_skills}
    ts = {s.lower() for s in target_skills}
    overlap = len(cs & ts)
    return round(min(100.0, (overlap / len(ts)) * 100), 1)


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

server = Server("talent-search-mcp")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_candidates",
            description=(
                "Search the candidate database using skills, location, seniority, and keywords. "
                "Returns a ranked list of matching candidates."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Skills to search for",
                    },
                    "location": {
                        "type": "string",
                        "description": "Preferred location or 'Remote'",
                    },
                    "seniority": {
                        "type": "string",
                        "description": "Desired seniority level (junior/mid/senior/staff/principal)",
                    },
                    "min_years_experience": {
                        "type": "number",
                        "description": "Minimum years of experience required",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="get_candidate_profile",
            description="Retrieve the full profile for a candidate by their ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "candidate_id": {
                        "type": "string",
                        "description": "The unique candidate ID",
                    }
                },
                "required": ["candidate_id"],
            },
        ),
        types.Tool(
            name="analyze_skill_match",
            description=(
                "Compute skill-overlap percentage between a candidate's skills "
                "and a list of required / preferred skills."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "required_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "preferred_skills": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
                "required": ["candidate_id", "required_skills"],
            },
        ),
        types.Tool(
            name="get_market_salary_data",
            description="Return market salary benchmarks for a given role and location.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_title": {"type": "string"},
                    "location": {"type": "string", "default": "United States"},
                    "seniority": {"type": "string", "default": "mid"},
                },
                "required": ["job_title"],
            },
        ),
        types.Tool(
            name="draft_outreach_message",
            description="Generate a personalised outreach email for a candidate.",
            inputSchema={
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "job_title": {"type": "string"},
                    "company_name": {"type": "string"},
                    "recruiter_name": {"type": "string", "default": "The Recruiting Team"},
                    "key_selling_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Top 2-3 selling points for the role",
                    },
                },
                "required": ["candidate_id", "job_title", "company_name"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict[str, Any]
) -> list[types.TextContent]:
    result = await _dispatch(name, arguments)
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


async def _dispatch(name: str, args: dict[str, Any]) -> Any:
    handlers = {
        "search_candidates": _tool_search_candidates,
        "get_candidate_profile": _tool_get_candidate_profile,
        "analyze_skill_match": _tool_analyze_skill_match,
        "get_market_salary_data": _tool_get_market_salary_data,
        "draft_outreach_message": _tool_draft_outreach_message,
    }
    handler = handlers.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    return await handler(args)


async def _tool_search_candidates(args: dict[str, Any]) -> dict[str, Any]:
    skills: list[str] = args.get("skills", [])
    location: str | None = args.get("location")
    seniority: str | None = args.get("seniority")
    min_exp: float = float(args.get("min_years_experience") or 0)
    limit: int = int(args.get("limit", 10))

    results = []
    for cand in _MOCK_CANDIDATES:
        if min_exp and cand["years_experience"] < min_exp:
            continue
        if seniority and cand["seniority"] != seniority.lower():
            # Be lenient: allow ±1 level
            seniority_order = ["intern", "junior", "mid", "senior", "staff", "principal"]
            try:
                idx_target = seniority_order.index(seniority.lower())
                idx_cand = seniority_order.index(cand["seniority"])
                if abs(idx_target - idx_cand) > 1:
                    continue
            except ValueError:
                pass
        if location and location.lower() not in ("remote", "any"):
            if (
                location.lower() not in cand["location"].lower()
                and cand["location"].lower() != "remote"
            ):
                continue

        score = _skill_score(cand["skills"], skills)
        results.append({**cand, "match_score": score})

    results.sort(key=lambda c: c["match_score"], reverse=True)
    return {"candidates": results[:limit], "total_found": len(results)}


async def _tool_get_candidate_profile(args: dict[str, Any]) -> dict[str, Any]:
    cid = args.get("candidate_id", "")
    for cand in _MOCK_CANDIDATES:
        if cand["id"] == cid:
            return {"candidate": cand}
    return {"error": f"Candidate '{cid}' not found"}


async def _tool_analyze_skill_match(args: dict[str, Any]) -> dict[str, Any]:
    cid = args.get("candidate_id", "")
    required = args.get("required_skills", [])
    preferred = args.get("preferred_skills", [])

    cand = next((c for c in _MOCK_CANDIDATES if c["id"] == cid), None)
    if cand is None:
        return {"error": f"Candidate '{cid}' not found"}

    c_skills = {s.lower() for s in cand["skills"]}
    req_set = {s.lower() for s in required}
    pref_set = {s.lower() for s in preferred}

    matched_required = list(c_skills & req_set)
    missing_required = list(req_set - c_skills)
    matched_preferred = list(c_skills & pref_set)

    req_score = _skill_score(cand["skills"], required)
    pref_score = _skill_score(cand["skills"], preferred) if preferred else 75.0
    combined = round(req_score * 0.7 + pref_score * 0.3, 1)

    return {
        "candidate_id": cid,
        "required_match_score": req_score,
        "preferred_match_score": pref_score,
        "combined_score": combined,
        "matched_required": matched_required,
        "missing_required": missing_required,
        "matched_preferred": matched_preferred,
    }


# Deterministic salary bands keyed by (seniority, broad_category)
_SALARY_BANDS: dict[str, dict[str, tuple[int, int]]] = {
    "junior": {"default": (70_000, 100_000), "ml": (80_000, 115_000)},
    "mid": {"default": (100_000, 140_000), "ml": (120_000, 160_000)},
    "senior": {"default": (140_000, 190_000), "ml": (160_000, 220_000)},
    "staff": {"default": (190_000, 250_000), "ml": (210_000, 270_000)},
    "principal": {"default": (230_000, 300_000), "ml": (250_000, 320_000)},
}
_ML_KEYWORDS = {"ml", "machine learning", "ai", "llm", "data scientist", "nlp"}


async def _tool_get_market_salary_data(args: dict[str, Any]) -> dict[str, Any]:
    job_title: str = args.get("job_title", "Software Engineer")
    location: str = args.get("location", "United States")
    seniority: str = args.get("seniority", "mid").lower()

    band_key = "ml" if any(kw in job_title.lower() for kw in _ML_KEYWORDS) else "default"
    level_bands = _SALARY_BANDS.get(seniority, _SALARY_BANDS["mid"])
    low, high = level_bands.get(band_key, level_bands["default"])

    # Apply a rough location multiplier
    location_mult = 1.2 if "san francisco" in location.lower() or "new york" in location.lower() else 1.0
    low_adj = int(low * location_mult)
    high_adj = int(high * location_mult)

    return {
        "job_title": job_title,
        "location": location,
        "seniority": seniority,
        "salary_range": {"min": low_adj, "max": high_adj, "currency": "USD", "period": "annual"},
        "median": int((low_adj + high_adj) / 2),
        "p25": int(low_adj + (high_adj - low_adj) * 0.25),
        "p75": int(low_adj + (high_adj - low_adj) * 0.75),
    }


async def _tool_draft_outreach_message(args: dict[str, Any]) -> dict[str, Any]:
    cid: str = args.get("candidate_id", "")
    job_title: str = args.get("job_title", "this role")
    company_name: str = args.get("company_name", "our company")
    recruiter_name: str = args.get("recruiter_name", "The Recruiting Team")
    selling_points: list[str] = args.get("key_selling_points", [])

    cand = next((c for c in _MOCK_CANDIDATES if c["id"] == cid), None)
    cand_name = cand["name"] if cand else "there"
    cand_skill = cand["skills"][0] if cand and cand.get("skills") else "engineering"

    points_text = ""
    if selling_points:
        bullet_list = "\n".join(f"  • {p}" for p in selling_points[:3])
        points_text = f"\n\nHere's why we think you'd love it:\n{bullet_list}\n"

    subject = f"Exciting {job_title} opportunity at {company_name}"
    body = (
        f"Hi {cand_name},\n\n"
        f"I came across your profile and was impressed by your background in {cand_skill}. "
        f"We have an opening for a {job_title} at {company_name} that I think aligns really well "
        f"with your experience.{points_text}\n"
        f"Would you be open to a 20-minute chat to explore this further?\n\n"
        f"Best,\n{recruiter_name}"
    )

    return {
        "candidate_id": cid,
        "subject": subject,
        "body": body,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_server() -> None:
    """Run the MCP server over stdio."""
    import asyncio

    async def _main() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_main())


if __name__ == "__main__":
    run_server()
