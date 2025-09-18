"""Runner utilities and CLI entry point for the LangGraph agent."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Tuple, cast

from langchain_openai import ChatOpenAI

from .config import AgentSettings
from .graph import build_graph
from .mcp import MCPRegistry
from .state import AgentState, InitialStateFactory


def _build_llm(settings: AgentSettings) -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY must be set to run the LangGraph agent with OpenRouter models."
        )
    headers = {
        "X-Title": "UniversalBackrooms LangGraph Agent",
    }
    referer = os.getenv("OPENROUTER_HTTP_REFERER")
    if referer:
        headers["HTTP-Referer"] = referer

    return ChatOpenAI(
        model=settings.openrouter_model,
        base_url=settings.openrouter_base_url,
        temperature=settings.openrouter_temperature,
        openai_api_key=api_key,
        api_key=api_key,
        default_headers=headers,
    )


def build_agent_app(
    settings: AgentSettings | None = None,
) -> Tuple[Any, AgentSettings]:
    """Instantiate and compile the LangGraph along with resolved settings."""

    resolved = settings or AgentSettings.from_env()
    llm = _build_llm(resolved)
    registry = MCPRegistry.from_path(resolved.mcp_config_path)
    graph = build_graph(llm=llm, settings=resolved, registry=registry)
    compiled = graph.compile()
    return compiled, resolved


def run_agent(goal: str, *, settings: AgentSettings | None = None) -> AgentState:
    """Run the agent to completion and return the final state."""

    app, resolved = build_agent_app(settings)
    initial_state = InitialStateFactory(
        goal=goal,
        mcp_config_path=str(resolved.mcp_config_path) if resolved.mcp_config_path else None,
    ).build()
    return app.invoke(initial_state)


def stream_agent(goal: str, *, settings: AgentSettings | None = None) -> Iterator[AgentState]:
    """Yield state updates as the agent progresses through the graph."""

    app, resolved = build_agent_app(settings)
    initial_state = InitialStateFactory(
        goal=goal,
        mcp_config_path=str(resolved.mcp_config_path) if resolved.mcp_config_path else None,
    ).build()

    try:
        iterator = app.stream(initial_state, stream_mode="values")
    except TypeError:  # Older langgraph fallback without stream_mode support
        iterator = app.stream(initial_state)

    last_state: AgentState | None = None
    last_chunk: Any = None
    for chunk in iterator:
        last_chunk = chunk
        state = _extract_state(chunk)
        if state is not None:
            last_state = state
            yield state

    if last_state is None and last_chunk is not None:
        fallback_state = _extract_state(last_chunk)
        if fallback_state is not None:
            yield fallback_state


def _extract_state(chunk: Any) -> AgentState | None:
    """Best-effort extraction of the agent state from LangGraph stream chunks."""

    if isinstance(chunk, dict):
        # LangGraph `stream_mode="values"` yields partial state dictionaries
        if _looks_like_state(chunk):
            return cast(AgentState, chunk)

        # Newer LangGraph versions may wrap output/state keys
        for key in ("state", "output", "value"):
            candidate = chunk.get(key)
            if isinstance(candidate, dict) and _looks_like_state(candidate):
                return cast(AgentState, candidate)

    return None


def _looks_like_state(candidate: Dict[str, Any]) -> bool:
    return "goal" in candidate and "loop_count" in candidate


def _cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LangGraph scenario agent.")
    parser.add_argument("goal", nargs="?", help="Scenario description for the agent to explore.")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream intermediate state updates as JSON lines.",
    )
    parser.add_argument(
        "--config",
        help="Path to a JSON config providing default goal and settings overrides.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_overrides: Mapping[str, object] | None = None
    config_goal: str | None = None
    if args.config:
        config_path = Path(args.config)
        try:
            config_data = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:  # pragma: no cover - handled at runtime
            parser.error(f"Config file not found: {exc.filename}")
        except json.JSONDecodeError as exc:
            parser.error(f"Invalid JSON in config file: {exc}")

        if isinstance(config_data, dict):
            raw_goal = config_data.get("goal")
            if isinstance(raw_goal, str) and raw_goal.strip():
                config_goal = raw_goal.strip()

            raw_settings = config_data.get("settings")
            if isinstance(raw_settings, dict):
                settings = dict(raw_settings)
                mcp_value = settings.get("mcp_config")
                if isinstance(mcp_value, str) and mcp_value and not Path(mcp_value).is_absolute():
                    settings["mcp_config"] = str((config_path.parent / mcp_value).resolve())
                config_overrides = settings

    goal = args.goal or config_goal
    if not goal:
        parser.error("Provide a goal argument or specify 'goal' in the config file.")

    agent_settings = AgentSettings.from_env(config_overrides)

    if args.stream:
        emitted = False
        for state in stream_agent(goal, settings=agent_settings):
            print(json.dumps(state, indent=2))
            emitted = True
        if not emitted:
            print(
                json.dumps(
                    {
                        "warning": "No incremental state events were emitted; rerun without --stream to view the final state.",
                    }
                )
            )
        return 0

    final_state = run_agent(goal, settings=agent_settings)
    print(json.dumps(final_state, indent=2))
    return 0


def main() -> None:
    raise SystemExit(_cli())


__all__ = ["build_agent_app", "run_agent", "stream_agent", "main"]
