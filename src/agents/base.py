"""Base class for all recruiting agents."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from src.models.search import AgentTrace

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Common interface and telemetry helpers for all agents."""

    agent_name: str = "base_agent"

    def _trace(
        self,
        action: str,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
        duration_ms: float,
    ) -> AgentTrace:
        return AgentTrace(
            agent_name=self.agent_name,
            action=action,
            input=input_data,
            output=output_data,
            duration_ms=duration_ms,
        )

    async def run_traced(self, action: str, input_data: dict[str, Any]) -> tuple[Any, AgentTrace]:
        """Execute :meth:`run` and wrap result in an :class:`AgentTrace`."""
        start = time.monotonic()
        result = await self.run(input_data)
        elapsed_ms = (time.monotonic() - start) * 1000
        output_summary = result if isinstance(result, dict) else {"result": str(result)[:200]}
        trace = self._trace(action, input_data, output_summary, elapsed_ms)
        return result, trace

    @abstractmethod
    async def run(self, input_data: dict[str, Any]) -> Any:
        """Execute the agent's primary task."""
