"""LangGraph construction and node definitions for the autonomous agent."""

from __future__ import annotations

import json
from typing import Any, Dict, List, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .config import AgentSettings
from .mcp import MCPRegistry, ToolCallResult
from .state import (
    AgentState,
    PlanDecision,
    ToolCall,
    append_history,
    clear_decision,
    set_decision,
    set_done,
    set_error,
)


_SYSTEM_PROMPT = (
    "You are an autonomous scenario exploration agent. You think step-by-step,"
    " decide whether to gather more context, call tools, or finish, and you respond"
    " strictly in JSON.\n"
    "Return an object with keys: description (string), rationale (string), stop (boolean),"
    " and optionally tool_call (object with server, tool, arguments).\n"
    "Only produce valid JSON."
)


def _format_history(history: List[Dict[str, Any]], limit: int = 6) -> str:
    """Condense recent history entries into a readable string."""

    if not history:
        return "(no prior observations)"
    recent = history[-limit:]
    lines = []
    for entry in recent:
        stage = entry.get("stage", "?")
        detail = entry.get("detail") or entry.get("observation") or ""
        if isinstance(detail, (dict, list)):
            try:
                detail = json.dumps(detail, ensure_ascii=False)
            except Exception:
                detail = str(detail)
        lines.append(f"- {stage}: {detail}")
    return "\n".join(lines)


def _parse_decision(raw_content: str, settings: AgentSettings) -> PlanDecision:
    """Best-effort JSON parsing with sensible defaults."""

    content_str = raw_content.strip()
    data: Dict[str, Any]
    try:
        data = json.loads(content_str)
    except json.JSONDecodeError:
        # Attempt to salvage first JSON object in the string
        start = content_str.find("{")
        end = content_str.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(content_str[start : end + 1])
            except Exception:
                data = {}
        else:
            data = {}

    description = str(data.get("description") or data.get("action") or "Provide a scenario update")
    rationale = data.get("rationale") or data.get("reason")
    stop_flag = bool(data.get("stop", False))

    tool_spec = data.get("tool_call") or data.get("tool")
    decision: PlanDecision = cast(PlanDecision, {"description": description, "stop": stop_flag})
    if rationale:
        decision["rationale"] = str(rationale)

    if isinstance(tool_spec, dict):
        tool_call: ToolCall = cast(
            ToolCall,
            {
                "server": str(tool_spec.get("server") or settings.discord_server_name),
                "tool": str(tool_spec.get("tool") or "send-message"),
                "arguments": dict(tool_spec.get("arguments") or {}),
            },
        )
        # Ensure Discord posts have a channel by default
        if tool_call.get("tool") == "send-message":
            args = dict(tool_call.get("arguments") or {})
            if settings.discord_channel_id and "channel" not in args:
                args["channel"] = settings.discord_channel_id
            if "message" not in args and description:
                args["message"] = description
            tool_call["arguments"] = args
        decision["tool_call"] = tool_call

    return decision


def build_graph(
    *,
    llm: BaseChatModel,
    settings: AgentSettings,
    registry: MCPRegistry,
) -> StateGraph[AgentState]:
    """Create an uncompiled LangGraph for the agent."""

    graph = StateGraph(AgentState)

    def plan_node(state: AgentState) -> AgentState:
        if state.get("done"):
            return state

        goal = state.get("goal", "Explore the scenario")
        loop_count = int(state.get("loop_count", 0))
        last_observation = state.get("last_observation") or "(none)"
        history_text = _format_history(state.get("history", []))

        user_prompt = (
            f"Goal: {goal}\n"
            f"Completed loops: {loop_count}\n"
            f"Most recent observation: {last_observation}\n"
            f"History:\n{history_text}\n"
            "You may call at most one tool per step."
            " If you call the Discord send-message tool, only provide the message body;"
            " the channel is managed for you."
        )

        try:
            response = llm.invoke([SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
            decision = _parse_decision(response.content if hasattr(response, "content") else str(response), settings)
            next_state = set_error(state, None)
        except Exception as exc:  # pragma: no cover - defensive fallback
            fallback_decision: PlanDecision = cast(
                PlanDecision,
                {
                    "description": f"Fallback Discord update about '{goal}'.",
                    "rationale": f"LLM planning failed ({exc}); using deterministic fallback.",
                    "stop": False,
                },
            )
            if settings.discord_channel_id:
                fallback_decision["tool_call"] = cast(
                    ToolCall,
                    {
                        "server": settings.discord_server_name,
                        "tool": "send-message",
                        "arguments": {
                            "channel": settings.discord_channel_id,
                            "message": fallback_decision["description"],
                        },
                    },
                )
            decision = fallback_decision
            next_state = set_error(state, f"Planning failed: {exc}")

        next_state = set_decision(next_state, decision)
        return append_history(
            next_state,
            {
                "stage": "plan",
                "detail": {
                    "description": decision.get("description"),
                    "stop": decision.get("stop", False),
                    "tool_call": decision.get("tool_call"),
                },
            },
        )

    def act_node(state: AgentState) -> AgentState:
        decision = state.get("decision")
        if not decision:
            decision = cast(PlanDecision, {"description": "No decision", "stop": True})

        if decision.get("stop"):
            message = decision.get("description") or "Agent elected to stop."
            next_state = set_done(state, message)
            return append_history(next_state, {"stage": "act", "detail": message})

        tool_call = decision.get("tool_call")
        if not tool_call:
            observation = decision.get("description") or "No tool action executed."
            next_state = cast(AgentState, dict(state))
            next_state["last_observation"] = observation
            return append_history(next_state, {"stage": "act", "detail": observation})

        arguments = dict(tool_call.get("arguments") or {})
        server = tool_call.get("server", settings.discord_server_name)
        tool = tool_call.get("tool", "send-message")

        # If this is a Discord send, ensure channel exists
        if tool == "send-message" and settings.discord_channel_id and "channel" not in arguments:
            arguments["channel"] = settings.discord_channel_id
        if tool == "send-message" and "message" not in arguments:
            arguments["message"] = decision.get("description") or "Scenario update."

        result = registry.call(server, tool, arguments)

        observation = _summarize_tool_result(result)
        next_state = cast(AgentState, dict(state))
        next_state["last_observation"] = observation
        if result.error:
            next_state = set_error(next_state, result.error)
        else:
            next_state = set_error(next_state, None)
        return append_history(
            next_state,
            {
                "stage": "act",
                "detail": {
                    "server": server,
                    "tool": tool,
                    "arguments": arguments,
                    "observation": observation,
                    "error": result.error,
                },
            },
        )

    def observe_node(state: AgentState) -> AgentState:
        next_state = cast(AgentState, dict(state))
        next_state = clear_decision(next_state)
        loop_count = int(next_state.get("loop_count", 0)) + 1
        next_state["loop_count"] = loop_count

        if next_state.get("done"):
            pass
        elif loop_count >= settings.max_loops:
            final_observation = next_state.get("last_observation") or "Reached configured max loops."
            next_state = set_done(next_state, final_observation)

        return append_history(
            next_state,
            {
                "stage": "observe",
                "detail": {
                    "loop_count": loop_count,
                    "done": next_state.get("done", False),
                    "observation": next_state.get("last_observation"),
                },
            },
        )

    def route_after_observe(state: AgentState) -> str:
        return "done" if state.get("done") else "continue"

    graph.add_node("plan", plan_node)
    graph.add_node("act", act_node)
    graph.add_node("observe", observe_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("act", "observe")
    graph.add_conditional_edges("observe", route_after_observe, {"continue": "plan", "done": END})

    return graph


def _summarize_tool_result(result: ToolCallResult) -> str:
    if result.error:
        return f"Tool {result.tool} on {result.server} failed: {result.error}"
    if not result.response:
        return f"Tool {result.tool} on {result.server} executed with no payload."
    content = result.response.get("content")
    if isinstance(content, list) and content:
        snippet = content[0]
    else:
        snippet = content
    if isinstance(snippet, (dict, list)):
        try:
            snippet = json.dumps(snippet, ensure_ascii=False)
        except Exception:
            snippet = str(snippet)
    return f"Tool {result.tool} on {result.server} succeeded: {snippet}"


__all__ = ["build_graph"]
