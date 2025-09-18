"""Typed state containers shared across the LangGraph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, cast


class ToolCall(TypedDict, total=False):
    """Structured description of a tool invocation."""

    server: str
    tool: str
    arguments: Dict[str, Any]


class PlanDecision(TypedDict, total=False):
    """Decision payload produced by the planning node."""

    description: str
    rationale: Optional[str]
    tool_call: ToolCall
    stop: bool


class AgentState(TypedDict, total=False):
    """Primary state mapping threaded through the LangGraph workflow."""

    goal: str
    loop_count: int
    decision: Optional[PlanDecision]
    last_observation: Optional[str]
    history: List[Dict[str, Any]]
    mcp_config_path: Optional[str]
    done: bool
    error: Optional[str]


@dataclass
class InitialStateFactory:
    """Helper for constructing a default agent state."""

    goal: str
    mcp_config_path: Optional[str] = None

    def build(self) -> AgentState:
        return cast(
            AgentState,
            {
                "goal": self.goal,
                "loop_count": 0,
                "decision": None,
                "last_observation": None,
                "history": [],
                "mcp_config_path": self.mcp_config_path,
                "done": False,
                "error": None,
            },
        )


def append_history(state: AgentState, entry: Dict[str, Any]) -> AgentState:
    """Return a shallow-copied state with an additional history entry."""

    new_state = cast(AgentState, dict(state))
    history = list(new_state.get("history", []))
    history.append(entry)
    new_state["history"] = history
    return new_state


def set_decision(state: AgentState, decision: Optional[PlanDecision]) -> AgentState:
    """Attach the latest planning decision to the state."""

    new_state = cast(AgentState, dict(state))
    new_state["decision"] = decision
    return new_state


def clear_decision(state: AgentState) -> AgentState:
    """Remove any pending decision from the state."""

    return set_decision(state, None)


def set_done(state: AgentState, message: str) -> AgentState:
    """Mark the run as complete, updating the final observation."""

    new_state = cast(AgentState, dict(state))
    new_state["done"] = True
    new_state["last_observation"] = message
    return new_state


def set_error(state: AgentState, error: Optional[str]) -> AgentState:
    """Record an error condition without mutating other fields."""

    new_state = cast(AgentState, dict(state))
    new_state["error"] = error
    return new_state


__all__ = [
    "AgentState",
    "InitialStateFactory",
    "PlanDecision",
    "ToolCall",
    "append_history",
    "clear_decision",
    "set_decision",
    "set_done",
    "set_error",
]
