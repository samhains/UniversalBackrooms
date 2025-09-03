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
    if not cfg.get("enabled", False):
        return None
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


def run_discord_agent(
    *,
    discord_cfg: Dict[str, Any],
    selected_models: List[str],
    round_entries: List[Dict[str, str]],
    transcript: List[Dict[str, str]],
    generate_text_fn,
    model_info: Dict[str, Dict[str, Any]],
    media_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Post a per-round summary to Discord via MCP 'discord' server.

    generate_text_fn(system_prompt: str, api_model: str, user_message: str) -> str
    """
    # 1) Build summary text via LLM (optional; can be disabled)
    use_llm = bool(discord_cfg.get("use_llm", True))
    api_model = _resolve_model_api(discord_cfg.get("model", "same-as-lm1"), selected_models, model_info)
    if use_llm:
        system_prompt = discord_cfg.get(
            "system_prompt",
            "You produce concise narrative summaries suitable for Discord updates.",
        )
        # Build user prompt from template if provided
        template = discord_cfg.get("user_template")
        if isinstance(template, str) and template.strip():
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
            user_prompt = template.format(
                round_bullets=round_bullets,
                last_actor=last_actor,
                last_text=last_text,
                transcript_bullets=transcript_bullets,
            )
        else:
            user_prompt = _build_round_summary_prompt(round_entries)
        summary = generate_text_fn(system_prompt, api_model, user_prompt).strip()
    else:
        # Simple non-LLM summary (first 300 chars of last message)
        last = round_entries[-1]["text"] if round_entries else ""
        summary = (last[:297] + "...") if len(last) > 300 else last

    # 2) Build tool args from config
    tool = discord_cfg.get("tool", {"server": "discord", "name": "send-message"})
    server_name = tool.get("server", "discord")
    tool_name = tool.get("name", "send-message")
    defaults = tool.get("defaults", {})
    # Expected by discord MCP: { server (optional), channel, message }
    args: Dict[str, Any] = {
        "channel": defaults.get("channel", "general"),
        "message": summary,
    }
    if "server" in defaults:
        args["server"] = defaults["server"]
    if media_url:
        args["mediaUrl"] = media_url

    # 3) Load MCP server config and call tool (no FastMCP wrapping)
    mcp_config_path = os.getenv("MCP_CONFIG", "mcp.config.json")
    server_cfg: MCPServerConfig = load_server_config(mcp_config_path, server_name)
    result = call_tool(server_cfg, tool_name, args)
    return {
        "posted": {
            "server": args.get("server"),
            "channel": args.get("channel"),
            "message": summary,
            "mediaUrl": media_url,
        },
        "result": result,
    }
