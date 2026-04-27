"""Tests for the FastAPI REST API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.models.candidate import SeniorityLevel
from src.models.job import JobDescription, JobRequirement
from src.models.search import SearchResponse, SearchStatus
from src.utils.fake_llm import FakeLLM


@pytest.fixture
def app_with_orchestrator(fake_llm: FakeLLM):
    """Create a test app with a real orchestrator using FakeLLM."""
    from src.agents.orchestrator import Orchestrator
    from src.mcp.client import MCPClient
    from src.main import create_app

    test_app = create_app()

    orchestrator = Orchestrator(llm=fake_llm, mcp_client=MCPClient())

    # Patch get_orchestrator in the search routes module
    with patch("src.main._orchestrator", orchestrator):
        with patch("src.api.routes.search._get_orchestrator", return_value=lambda: orchestrator):
            yield test_app, orchestrator


@pytest.fixture
def client(fake_llm: FakeLLM):
    """Return a TestClient with the orchestrator patched in."""
    from src.agents.orchestrator import Orchestrator
    from src.mcp.client import MCPClient
    from src.main import app

    orchestrator = Orchestrator(llm=fake_llm, mcp_client=MCPClient())

    with patch("src.main._orchestrator", orchestrator):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


class TestHealthEndpoints:
    def test_health_ok(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "service" in data

    def test_root_redirects_to_health(self, client: TestClient):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestCandidateEndpoints:
    def test_list_candidates(self, client: TestClient):
        resp = client.get("/api/v1/candidates")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_list_candidates_limit(self, client: TestClient):
        resp = client.get("/api/v1/candidates?limit=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) <= 2

    def test_get_candidate_by_id(self, client: TestClient):
        resp = client.get("/api/v1/candidates/cand-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "cand-001"
        assert data["name"] == "Alice Chen"

    def test_get_candidate_not_found(self, client: TestClient):
        resp = client.get("/api/v1/candidates/nonexistent-id")
        assert resp.status_code == 404

    def test_search_candidates_by_skills(self, client: TestClient):
        resp = client.post(
            "/api/v1/candidates/search",
            json={"skills": ["Python"], "limit": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_search_candidates_empty_skills(self, client: TestClient):
        resp = client.post("/api/v1/candidates/search", json={})
        assert resp.status_code == 200

    def test_candidate_has_required_fields(self, client: TestClient):
        resp = client.get("/api/v1/candidates/cand-001")
        data = resp.json()
        for field in ("id", "name", "skills", "experience", "seniority_level"):
            assert field in data


class TestSearchEndpoints:
    def test_search_with_valid_jd(self, client: TestClient):
        resp = client.post(
            "/api/v1/search",
            json={
                "job_description_text": (
                    "We are looking for a Senior Python Engineer with LangChain experience. "
                    "5+ years required. Remote friendly."
                ),
                "top_k": 3,
                "include_outreach_drafts": True,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert "search_id" in data
        assert "ranked_candidates" in data
        assert "agent_traces" in data

    def test_search_empty_jd_rejected(self, client: TestClient):
        resp = client.post(
            "/api/v1/search",
            json={"job_description_text": "   "},
        )
        assert resp.status_code == 422

    def test_search_top_k_enforced(self, client: TestClient):
        resp = client.post(
            "/api/v1/search",
            json={
                "job_description_text": "Python developer needed",
                "top_k": 2,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["ranked_candidates"]) <= 2

    def test_search_response_has_job_description(self, client: TestClient):
        resp = client.post(
            "/api/v1/search",
            json={"job_description_text": "Software engineer role"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("job_description") is not None

    def test_search_no_outreach_drafts(self, client: TestClient):
        resp = client.post(
            "/api/v1/search",
            json={
                "job_description_text": "Python developer needed",
                "include_outreach_drafts": False,
                "top_k": 3,
            },
        )
        assert resp.status_code == 200
