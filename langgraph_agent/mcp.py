"""Convenience wrappers around the repo's MCP client helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from mcp_client import MCPClientError, MCPServerConfig, call_tool


@dataclass
class ToolCallResult:
    """Normalized response from an MCP tool call."""

    server: str
    tool: str
    arguments: Dict[str, Any]
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class MCPRegistry:
    """Loads and manages MCP server configurations."""

    def __init__(self, servers: Mapping[str, MCPServerConfig]):
        self._servers = dict(servers)

    def call(self, server: str, tool: str, arguments: Optional[Dict[str, Any]] = None) -> ToolCallResult:
        payload = dict(arguments or {})
        cfg = self._servers.get(server)
        if cfg is None:
            return ToolCallResult(
                server=server,
                tool=tool,
                arguments=payload,
                error=f"MCP server '{server}' is not defined in the current configuration.",
            )
        try:
            response = call_tool(cfg, tool, payload)
        except MCPClientError as exc:
            return ToolCallResult(server=server, tool=tool, arguments=payload, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            return ToolCallResult(
                server=server,
                tool=tool,
                arguments=payload,
                error=f"Unexpected MCP failure: {exc}",
            )
        return ToolCallResult(server=server, tool=tool, arguments=payload, response=response)

    @classmethod
    def from_path(cls, path: Optional[Path]) -> "MCPRegistry":
        """Load servers from the specified configuration file."""

        if path is None or not path.exists():
            return cls({})

        payload = json.loads(path.read_text(encoding="utf-8"))
        servers_raw: Mapping[str, Any] = payload.get("mcpServers", {})
        servers: Dict[str, MCPServerConfig] = {}
        for name, spec in servers_raw.items():
            if spec.get("type", "stdio") != "stdio":
                continue
            servers[name] = MCPServerConfig(
                command=spec["command"],
                args=spec.get("args", []),
                env=spec.get("env", {}),
                cwd=spec.get("cwd"),
            )
        return cls(servers)


__all__ = ["MCPRegistry", "ToolCallResult"]
