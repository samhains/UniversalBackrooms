"""Configuration helpers for the LangGraph agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

from dotenv import load_dotenv

# Load environment variables early so CLI usage works without extra imports.
load_dotenv()


@dataclass
class AgentSettings:
    """Container for the agent's configurable settings."""

    openrouter_model: str
    openrouter_temperature: float
    max_loops: int
    discord_channel_id: Optional[str]
    discord_server_name: str
    mcp_config_path: Optional[Path]
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    @classmethod
    def from_env(cls, overrides: Optional[Mapping[str, object]] = None) -> "AgentSettings":
        """Construct settings from environment variables, applying overrides when supplied."""

        overrides = overrides or {}

        model = str(overrides.get("model") or os.getenv("LANGGRAPH_AGENT_MODEL", "nousresearch/hermes-4-405b"))

        temperature_override = overrides.get("temperature")
        if temperature_override is not None:
            try:
                temperature = float(temperature_override)
            except (TypeError, ValueError):
                temperature = 0.2
        else:
            try:
                temperature = float(os.getenv("LANGGRAPH_AGENT_TEMPERATURE", "0.2"))
            except ValueError:
                temperature = 0.2

        max_loops_override = overrides.get("max_loops")
        if max_loops_override is not None:
            try:
                max_loops = int(max_loops_override)
            except (TypeError, ValueError):
                max_loops = 4
        else:
            try:
                max_loops = int(os.getenv("LANGGRAPH_AGENT_MAX_LOOPS", "4"))
            except ValueError:
                max_loops = 4

        if "discord_channel" in overrides and overrides.get("discord_channel") is not None:
            discord_channel: Optional[str] = str(overrides.get("discord_channel"))
        else:
            env_channel = os.getenv("LANGGRAPH_AGENT_DISCORD_CHANNEL") or os.getenv("DISCORD_CHANNEL_ID")
            discord_channel = env_channel if env_channel else None

        discord_server = str(
            overrides.get("discord_server") or os.getenv("LANGGRAPH_AGENT_DISCORD_SERVER", "discord")
        )

        base_url = str(overrides.get("base_url") or os.getenv("LANGGRAPH_AGENT_BASE_URL", "https://openrouter.ai/api/v1"))

        mcp_path_value = overrides.get("mcp_config")
        if isinstance(mcp_path_value, Path):
            mcp_path = mcp_path_value
        elif isinstance(mcp_path_value, str) and mcp_path_value:
            mcp_path = Path(mcp_path_value)
        else:
            mcp_default = os.getenv("LANGGRAPH_AGENT_MCP_CONFIG", "mcp.config.json")
            mcp_path = Path(mcp_default) if mcp_default else None

        return cls(
            openrouter_model=model,
            openrouter_temperature=temperature,
            max_loops=max(1, max_loops),
            discord_channel_id=discord_channel,
            discord_server_name=discord_server,
            mcp_config_path=mcp_path,
            openrouter_base_url=base_url,
        )


__all__ = ["AgentSettings"]
