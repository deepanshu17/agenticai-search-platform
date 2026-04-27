"""MCP client and LangChain tool adapters.

This module provides:
- :class:`MCPToolAdapter` – wraps an async callable as a LangChain BaseTool.
- :class:`MCPClient` – thin client that calls MCP tools directly (no network transport
  needed when the server is in-process).  For out-of-process use-cases, swap the
  ``_call_tool`` implementation to use ``mcp.ClientSession`` over stdio/SSE.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic adapter: wraps any async callable as a LangChain tool
# ---------------------------------------------------------------------------


class MCPToolAdapter(BaseTool):
    """Wraps an MCP tool dispatch function as a LangChain :class:`BaseTool`.

    Usage::

        adapter = MCPToolAdapter(
            name="search_candidates",
            description="Search the candidate database",
            tool_fn=mcp_client.call_tool,
        )
    """

    name: str
    description: str
    tool_fn: Any = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> str:
        """Synchronous wrapper — delegates to the async implementation."""
        import asyncio

        return asyncio.get_event_loop().run_until_complete(self._arun(**kwargs))

    async def _arun(self, **kwargs: Any) -> str:
        """Async execution: calls the underlying MCP tool dispatch function."""
        result = await self.tool_fn(self.name, kwargs)
        return json.dumps(result, indent=2) if not isinstance(result, str) else result


# ---------------------------------------------------------------------------
# In-process MCP client
# ---------------------------------------------------------------------------


class MCPClient:
    """In-process client that calls the MCP server's dispatch function directly.

    For production deployments where the MCP server runs as a separate process,
    replace ``_call_tool`` with a real ``mcp.ClientSession`` call over stdio or SSE.
    """

    def __init__(self) -> None:
        from src.mcp.server import _dispatch  # imported lazily to avoid circular deps

        self._dispatch = _dispatch

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a named MCP tool with the given arguments.

        Args:
            tool_name: The registered MCP tool name.
            arguments: JSON-serialisable arguments dict.

        Returns:
            Parsed tool result (dict or list).
        """
        logger.debug("MCP call: tool=%s args=%s", tool_name, arguments)
        return await self._dispatch(tool_name, arguments)

    def as_langchain_tool(self, tool_name: str, description: str) -> MCPToolAdapter:
        """Return a LangChain :class:`BaseTool` backed by this client."""
        return MCPToolAdapter(
            name=tool_name,
            description=description,
            tool_fn=self.call_tool,
        )

    def get_all_tools(self) -> list[MCPToolAdapter]:
        """Return all available MCP tools as LangChain adapters."""
        tool_specs = [
            (
                "search_candidates",
                "Search the candidate database by skills, location, seniority, and experience.",
            ),
            (
                "get_candidate_profile",
                "Retrieve the full profile for a candidate by their ID.",
            ),
            (
                "analyze_skill_match",
                "Compute skill-overlap percentage between a candidate and a job's required skills.",
            ),
            (
                "get_market_salary_data",
                "Return market salary benchmarks for a given role and location.",
            ),
            (
                "draft_outreach_message",
                "Generate a personalised outreach email for a candidate.",
            ),
        ]
        return [self.as_langchain_tool(name, desc) for name, desc in tool_specs]
