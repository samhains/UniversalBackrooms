import json
import os
from typing import Any, Dict, List, Optional

from mcp_cli import load_server_config  # reuse config loader
from mcp_client import MCPServerConfig, call_tool


def load_discord_config(profile_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load a Discord agent profile from ./discord/<profile>.json.

    Returns None if no profile_name provided or file not found or disabled.
    """
    if not profile_name:
        return None
    path = os.path.join("discord", f"{profile_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # Always treat profiles as enabled; no config flag required
    return cfg


def _resolve_model_api(pref: str, selected_models: List[str], model_info: Dict[str, Dict[str, Any]]) -> str:
    """Resolve provider API model string given a preference like 'same-as-lm1' or a key.
    Falls back to 'sonnet' if available.
    """
    model_key = pref or "same-as-lm1"
    if model_key == "same-as-lm1":
        for m in selected_models:
            if m in model_info:
                return model_info[m]["api_name"]
        # fallback
        if "sonnet" in model_info:
            return model_info["sonnet"]["api_name"]
    if model_key in model_info:
        return model_info[model_key]["api_name"]
    return model_key


def _build_round_summary_prompt(round_entries: List[Dict[str, str]]) -> str:
    lines = [
        "Summarize the latest round of the backrooms conversation in 1-3 sentences.",
        "Be clear, concise, and evocative; avoid spoilers or meta commentary.",
        "Write in present tense and speak as an observer.",
        "Latest round:",
    ]
    for e in round_entries:
        actor = e.get("actor", "")
        text = e.get("text", "")
        lines.append(f"- {actor}: {text}")
    return "\n".join(lines)


def _build_default_user_prompt(
    *, round_entries: List[Dict[str, str]], transcript: List[Dict[str, str]], window: int
) -> str:
    """Default prompt builder when no user_template is provided.

    Includes a recent transcript context window (if available) and the latest round.
    """
    lines: List[str] = [
        "Summarize the latest round of the backrooms conversation in 1-3 sentences.",
        "Be clear, concise, and evocative; avoid spoilers or meta commentary.",
        "Write in present tense and speak as an observer. Use the recent context to maintain continuity.",
    ]

    # Append recent transcript context if provided
    win = max(0, int(window))
    if transcript and win > 0:
        tb_src = transcript[-win:]
        lines.append("Recent context (most recent first):")
        for e in tb_src:
            actor = e.get("actor", "")
            text = e.get("text", "")
            lines.append(f"- {actor}: {text}")

    # Append latest round bullets
    lines.append("Latest round:")
    for e in round_entries:
        actor = e.get("actor", "")
        text = e.get("text", "")
        lines.append(f"- {actor}: {text}")

    return "\n".join(lines)


def run_discord_agent(
    *,
    discord_cfg: Dict[str, Any],
    selected_models: List[str],
    round_entries: List[Dict[str, str]],
    transcript: List[Dict[str, str]],
    generate_text_fn,
    model_info: Dict[str, Dict[str, Any]],
    media_url: Optional[str] = None,
    override_channel: Optional[str] = None,
    override_server: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Post a per-round summary to Discord via MCP 'discord' server.

    generate_text_fn(system_prompt: str, api_model: str, user_message: str) -> str
    """
    # 1) Build summary text via LLM (optional; can be disabled)
    api_model = _resolve_model_api(discord_cfg.get("model", "same-as-lm1"), selected_models, model_info)
    # Always generate with LLM (config flag removed)
    if True:
        system_prompt = discord_cfg.get(
            "system_prompt",
            "You produce concise narrative summaries suitable for Discord updates.",
        )
        # Build user prompt, preferring template if provided, otherwise include transcript context
        template = discord_cfg.get("user_template")
        if isinstance(template, str) and template.strip():
            # Build bullet lists
            round_bullets = "\n".join(
                f"- {e.get('actor','')}: {e.get('text','')}" for e in round_entries
            )
            last_actor = round_entries[-1].get("actor", "") if round_entries else ""
            last_text = round_entries[-1].get("text", "") if round_entries else ""
            # Optional transcript window
            win = int(discord_cfg.get("transcript_window", 10))
            tb_src = transcript[-win:] if transcript else []
            transcript_bullets = "\n".join(
                f"- {e.get('actor','')}: {e.get('text','')}" for e in tb_src
            )

            # Template variables (legacy placeholders removed)
            fmt_vars = {
                "latest_round_bullets": round_bullets,
                "context_bullets": transcript_bullets,
                "last_actor": last_actor,
                "last_text": last_text,
            }
            user_prompt = template.format(**fmt_vars)
        else:
            user_prompt = _build_default_user_prompt(
                round_entries=round_entries,
                transcript=transcript,
                window=int(discord_cfg.get("transcript_window", 10)),
            )
        summary = generate_text_fn(system_prompt, api_model, user_prompt).strip()
    # No non-LLM path (was gated by use_llm)

    # 2) Build tool args from config (summary post)
    tool = discord_cfg.get("tool", {"server": "discord", "name": "send-message"})
    server_name = tool.get("server", "discord")
    tool_name = tool.get("name", "send-message")
    defaults = tool.get("defaults", {})
    # Expected by discord MCP: { server (optional), channel, message }
    args: Dict[str, Any] = {
        "channel": override_channel or defaults.get("channel", "general"),
        "message": summary,
    }
    if override_server is not None and str(override_server).strip():
        args["server"] = override_server
    else:
        sv = defaults.get("server")
        if isinstance(sv, str) and sv.strip():
            args["server"] = sv.strip()
    if media_url:
        args["mediaUrl"] = media_url

    # 3) Load MCP server config and call tool (no FastMCP wrapping)
    mcp_config_path = os.getenv("MCP_CONFIG") or os.getenv("MCP_SERVERS_CONFIG") or "mcp.config.json"
    if not os.path.exists(mcp_config_path):
        alt = "mcp_servers.json"
        if os.path.exists(alt):
            mcp_config_path = alt
    server_cfg: MCPServerConfig = load_server_config(mcp_config_path, server_name)
    result = call_tool(server_cfg, tool_name, args)

    out: Dict[str, Any] = {
        "posted": {
            "server": args.get("server"),
            "channel": args.get("channel"),
            "message": summary,
            "mediaUrl": media_url,
        },
        "result": result,
    }

    # 3) Optionally post verbatim transcript of the round to a different channel
    if bool(discord_cfg.get("post_transcript", False)):
        # Allow full separate tool configuration for transcript posts
        ttool = discord_cfg.get("transcript_tool") or tool
        t_server_name = ttool.get("server", "discord")
        t_tool_name = ttool.get("name", "send-message")
        t_defaults = ttool.get("defaults", {})
        # Channel precedence: transcript_channel (top-level) > transcript_tool.defaults.channel > tool.defaults.channel
        transcript_channel = (
            discord_cfg.get("transcript_channel")
            or t_defaults.get("channel")
            or defaults.get("channel", "transcripts")
        )
        # Post each round entry as its own message to ensure readability and avoid chunking
        posted_transcript: List[Dict[str, Any]] = []
        t_args_base: Dict[str, Any] = {"channel": transcript_channel}
        sv2 = t_defaults.get("server")
        if isinstance(sv2, str) and sv2.strip():
            t_args_base["server"] = sv2.strip()
        # Reuse same MCP server config if server name matches; else reload
        if t_server_name == server_name:
            t_server_cfg = server_cfg
        else:
            mcp_config_path = os.getenv("MCP_CONFIG") or os.getenv("MCP_SERVERS_CONFIG") or "mcp.config.json"
            if not os.path.exists(mcp_config_path):
                alt = "mcp_servers.json"
                if os.path.exists(alt):
                    mcp_config_path = alt
            t_server_cfg = load_server_config(mcp_config_path, t_server_name)

        for e in round_entries:
            actor = e.get("actor", "")
            text = e.get("text", "")
            body = f"{actor}:\n{text}"
            t_args = dict(t_args_base)
            t_args["message"] = body
            t_res = call_tool(t_server_cfg, t_tool_name, t_args)
            posted_transcript.append({
                "server": t_args.get("server"),
                "channel": t_args.get("channel"),
                "message": body,
            })
        out["posted_transcript"] = posted_transcript

    return out
