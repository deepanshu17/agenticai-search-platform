# AgenticAI Search Platform

> **AI-native talent search service with multi-agent orchestration, LangChain pipelines, and MCP-enabled tool calling — built for the next generation of intelligent recruiting.**

---

## Overview

AgenticAI Search Platform is a production-ready recruiting intelligence engine that combines:

- **Multi-Agent Orchestration** — five specialised AI agents coordinate to deliver end-to-end talent discovery
- **LangChain LCEL Pipelines** — structured LLM chains for job description analysis, candidate evaluation, and match scoring
- **Model Context Protocol (MCP)** — a fully-featured MCP server exposes recruiting tools that any MCP-compatible client or AI model can call
- **FastAPI REST API** — a clean, versioned HTTP interface for integrating with your ATS, HR systems, or front-end

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI REST API                         │
│  POST /api/v1/search   GET /api/v1/candidates   /health    │
└──────────────────────────────┬──────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │     Orchestrator     │   (multi-agent coordinator)
                    └─────────┬───────────┘
          ┌──────────┬────────┼────────────┬──────────┐
          │          │        │            │          │
   ┌──────▼──┐ ┌─────▼──┐ ┌──▼──────┐ ┌──▼──────┐ ┌─▼────────┐
   │  Job    │ │Search  │ │ Skills  │ │ Ranking │ │ Outreach │
   │Analyzer │ │ Agent  │ │  Agent  │ │  Agent  │ │  Agent   │
   │Pipeline │ │        │ │         │ │         │ │          │
   └─────────┘ └───┬────┘ └────┬────┘ └────┬────┘ └──────────┘
                   │           │            │
            ┌──────▼───────────▼────────────▼──────┐
            │           MCP Server                  │
            │  search_candidates                    │
            │  get_candidate_profile                │
            │  analyze_skill_match                  │
            │  get_market_salary_data               │
            │  draft_outreach_message               │
            └───────────────────────────────────────┘
```

### Pipeline Stages

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | `JobAnalyzerPipeline` | Parses raw JD text → structured `JobDescription` via LangChain LCEL |
| 2 | `SearchAgent` | Uses MCP `search_candidates` tool to find matching profiles |
| 3 | `SkillsAgent` | Evaluates each candidate with LLM scoring + MCP `analyze_skill_match` |
| 4 | `RankingAgent` | Sorts by composite score; enriches with `get_market_salary_data` |
| 5 | `OutreachAgent` | Drafts personalised emails for top-K via `draft_outreach_message` |

---

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key (optional — service runs in demo mode without one)

### Installation

```bash
git clone https://github.com/deepanshu17/agenticai-search-platform.git
cd agenticai-search-platform
pip install -e .
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `""` | OpenAI API key (required for full LLM features) |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model name |
| `OPENAI_TEMPERATURE` | `0.0` | Sampling temperature |
| `API_HOST` | `0.0.0.0` | Uvicorn bind host |
| `API_PORT` | `8000` | Uvicorn bind port |
| `MAX_CANDIDATES_PER_SEARCH` | `20` | Max candidates to retrieve per search |
| `DEFAULT_TOP_K` | `10` | Default number of ranked results |

### Run the API Server

```bash
talent-search
# or
python -m src.main
```

Interactive docs at `http://localhost:8000/docs`.

### Run the MCP Server (standalone)

```bash
mcp-server
# or
python -m src.mcp.server
```

---

## API Reference

### `POST /api/v1/search`

Run a full multi-agent talent search pipeline.

**Request:**

```json
{
  "job_description_text": "We are hiring a Senior ML Engineer...",
  "filters": {
    "locations": ["San Francisco", "Remote"],
    "seniority_levels": ["senior", "staff"],
    "min_years_experience": 5
  },
  "top_k": 10,
  "include_outreach_drafts": true
}
```

**Response:**

```json
{
  "search_id": "uuid",
  "status": "completed",
  "job_description": { "title": "Senior ML Engineer", "..." : "..." },
  "ranked_candidates": [
    {
      "rank": 1,
      "candidate": { "name": "Alice Chen", "skills": ["..."], "..." : "..." },
      "score": {
        "overall_score": 91.5,
        "skills_match_score": 95.0,
        "strengths": ["Python expertise", "LLM production experience"],
        "gaps": ["Limited Kubernetes exposure"]
      },
      "outreach_draft": "Hi Alice,\n\nI came across your profile..."
    }
  ],
  "total_candidates_found": 6,
  "agent_traces": [{"agent_name": "search_agent", "duration_ms": 12.3}],
  "summary": "Found 6 candidates; top 5 after filtering.",
  "duration_ms": 1234.5
}
```

### `GET /api/v1/candidates`

List all candidates.

### `GET /api/v1/candidates/{candidate_id}`

Get a single candidate profile.

### `POST /api/v1/candidates/search`

Search candidates by skills, location, seniority.

### `GET /health`

Returns service health, version, and name.

---

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_candidates` | Fuzzy-search by skills, location, seniority, experience |
| `get_candidate_profile` | Retrieve full profile by ID |
| `analyze_skill_match` | Compute skill overlap score |
| `get_market_salary_data` | Market salary benchmarks |
| `draft_outreach_message` | Personalised recruiting email generator |

### Using MCP Tools from Python

```python
from src.mcp.client import MCPClient

client = MCPClient()

result = await client.call_tool("search_candidates", {
    "skills": ["Python", "PyTorch"],
    "seniority": "senior",
    "min_years_experience": 5,
})
```

### Using MCP Tools as LangChain Tools

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from src.mcp.client import MCPClient

llm = ChatOpenAI(model="gpt-4o-mini")
tools = MCPClient().get_all_tools()  # returns list[BaseTool]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a recruiting assistant. Use tools to find talent."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
response = executor.invoke({"input": "Find me Python ML engineers in San Francisco"})
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

> Tests use a built-in `FakeLLM` — **no OpenAI API key required**.

**Coverage:**

| File | What it tests |
|------|---------------|
| `test_models.py` | Pydantic data model validation |
| `test_mcp.py` | MCP server tools + client adapter |
| `test_pipelines.py` | LangChain pipeline logic |
| `test_agents.py` | All five agents + Orchestrator integration |
| `test_api.py` | FastAPI endpoint tests |

---

## Project Structure

```
src/
├── config.py                  # Pydantic-settings configuration
├── main.py                    # FastAPI app factory & entry point
├── models/                    # Candidate, Job, Search Pydantic models
├── pipelines/                 # LangChain LCEL pipelines
│   ├── job_analyzer.py
│   ├── candidate_evaluator.py
│   └── match_scorer.py
├── agents/                    # Multi-agent system
│   ├── orchestrator.py        # Multi-agent coordinator
│   ├── search_agent.py
│   ├── skills_agent.py
│   ├── ranking_agent.py
│   └── outreach_agent.py
├── mcp/
│   ├── server.py              # MCP server (5 recruiting tools)
│   └── client.py              # MCP client + LangChain adapters
├── api/routes/                # FastAPI route handlers
└── utils/fake_llm.py          # Deterministic LLM for testing
```

---

## License

MIT
