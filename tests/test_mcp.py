"""Tests for MCP server tools and client."""

from __future__ import annotations

import pytest

from src.mcp.client import MCPClient, MCPToolAdapter
from src.mcp.server import (
    _tool_analyze_skill_match,
    _tool_draft_outreach_message,
    _tool_get_candidate_profile,
    _tool_get_market_salary_data,
    _tool_search_candidates,
)


class TestMCPServerTools:
    """Tests for individual MCP tool handler functions."""

    @pytest.mark.asyncio
    async def test_search_candidates_no_filters(self):
        result = await _tool_search_candidates({})
        assert "candidates" in result
        assert isinstance(result["candidates"], list)
        assert len(result["candidates"]) > 0

    @pytest.mark.asyncio
    async def test_search_candidates_by_skill(self):
        result = await _tool_search_candidates({"skills": ["Python"], "limit": 5})
        assert "candidates" in result
        # All returned candidates should have Python
        for cand in result["candidates"]:
            assert "Python" in cand["skills"]

    @pytest.mark.asyncio
    async def test_search_candidates_limit(self):
        result = await _tool_search_candidates({"limit": 2})
        assert len(result["candidates"]) <= 2

    @pytest.mark.asyncio
    async def test_search_candidates_min_experience(self):
        result = await _tool_search_candidates({"min_years_experience": 8.0})
        for cand in result["candidates"]:
            assert cand["years_experience"] >= 8.0

    @pytest.mark.asyncio
    async def test_search_candidates_by_seniority(self):
        result = await _tool_search_candidates({"seniority": "senior"})
        for cand in result["candidates"]:
            # Senior ±1 level
            assert cand["seniority"] in ("mid", "senior", "staff")

    @pytest.mark.asyncio
    async def test_get_candidate_profile_found(self):
        result = await _tool_get_candidate_profile({"candidate_id": "cand-001"})
        assert "candidate" in result
        assert result["candidate"]["id"] == "cand-001"

    @pytest.mark.asyncio
    async def test_get_candidate_profile_not_found(self):
        result = await _tool_get_candidate_profile({"candidate_id": "nonexistent"})
        assert "error" in result

    @pytest.mark.asyncio
    async def test_analyze_skill_match(self):
        result = await _tool_analyze_skill_match(
            {
                "candidate_id": "cand-001",
                "required_skills": ["Python", "PyTorch"],
                "preferred_skills": ["LangChain"],
            }
        )
        assert "required_match_score" in result
        assert "combined_score" in result
        assert result["required_match_score"] == 100.0  # Alice has both Python and PyTorch

    @pytest.mark.asyncio
    async def test_analyze_skill_match_candidate_not_found(self):
        result = await _tool_analyze_skill_match(
            {"candidate_id": "bad-id", "required_skills": ["Python"]}
        )
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_market_salary_data(self):
        result = await _tool_get_market_salary_data(
            {"job_title": "ML Engineer", "location": "San Francisco, CA", "seniority": "senior"}
        )
        assert "salary_range" in result
        sr = result["salary_range"]
        assert sr["min"] > 0
        assert sr["max"] > sr["min"]
        assert sr["currency"] == "USD"

    @pytest.mark.asyncio
    async def test_get_market_salary_data_sf_premium(self):
        result_sf = await _tool_get_market_salary_data(
            {"job_title": "Software Engineer", "location": "San Francisco, CA", "seniority": "mid"}
        )
        result_other = await _tool_get_market_salary_data(
            {"job_title": "Software Engineer", "location": "Austin, TX", "seniority": "mid"}
        )
        assert result_sf["salary_range"]["min"] > result_other["salary_range"]["min"]

    @pytest.mark.asyncio
    async def test_draft_outreach_message(self):
        result = await _tool_draft_outreach_message(
            {
                "candidate_id": "cand-001",
                "job_title": "Senior ML Engineer",
                "company_name": "TechCorp",
                "recruiter_name": "Jane Smith",
                "key_selling_points": ["Remote-friendly", "Competitive salary"],
            }
        )
        assert "subject" in result
        assert "body" in result
        assert "Alice Chen" in result["body"]
        assert "TechCorp" in result["body"]

    @pytest.mark.asyncio
    async def test_draft_outreach_unknown_candidate(self):
        result = await _tool_draft_outreach_message(
            {
                "candidate_id": "unknown",
                "job_title": "Engineer",
                "company_name": "Corp",
            }
        )
        assert "body" in result
        assert "there" in result["body"]  # fallback name


class TestMCPClient:
    @pytest.fixture
    def client(self) -> MCPClient:
        return MCPClient()

    @pytest.mark.asyncio
    async def test_call_tool_search(self, client: MCPClient):
        result = await client.call_tool("search_candidates", {"limit": 3})
        assert "candidates" in result

    @pytest.mark.asyncio
    async def test_call_tool_unknown(self, client: MCPClient):
        result = await client.call_tool("nonexistent_tool", {})
        assert "error" in result

    def test_get_all_tools(self, client: MCPClient):
        tools = client.get_all_tools()
        assert len(tools) == 5
        tool_names = {t.name for t in tools}
        assert "search_candidates" in tool_names
        assert "draft_outreach_message" in tool_names

    def test_as_langchain_tool(self, client: MCPClient):
        tool = client.as_langchain_tool("search_candidates", "Search for candidates")
        assert isinstance(tool, MCPToolAdapter)
        assert tool.name == "search_candidates"
