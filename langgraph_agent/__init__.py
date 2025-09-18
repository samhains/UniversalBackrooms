"""LangGraph-based autonomous scenario agent."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["run_agent", "build_agent_app"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        module = import_module(".runner", __name__)
        return getattr(module, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
