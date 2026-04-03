from __future__ import annotations

from collections.abc import Callable
from typing import Any


class MCPToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._trace: list[dict[str, Any]] = []

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        self._tools[name] = fn

    def call(self, name: str, *, agent: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown MCP tool: {name}")
        result = self._tools[name](**kwargs)
        self._trace.append(
            {
                "agent": agent,
                "tool": name,
                "inputs": sorted(kwargs.keys()),
            }
        )
        return result

    def consume_trace(self) -> list[dict[str, Any]]:
        trace = list(self._trace)
        self._trace.clear()
        return trace
