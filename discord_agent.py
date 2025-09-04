import json
import os
from typing import Any, Dict, List, Optional

from mcp_cli import load_server_config  # reuse config loader
from mcp_client import MCPServerConfig, call_tool, list_tools


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
    # Always treat attached discord config as enabled
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


def _render_transcript(entries: List[Dict[str, str]]) -> str:
    """Render entries as simple transcript lines: "Actor: text"."""
    lines: List[str] = []
    for e in entries:
        actor = e.get("actor", "")
        text = e.get("text", "")
        lines.append(f"{actor}: {text}")
    return "\n".join(lines)


# No JSON fallback loader: prompts must come from the provided discord profile.


def run_discord_agent(
    *,
    discord_cfg: Dict[str, Any],
    selected_models: List[str],
    round_entries: List[Dict[str, str]],
    transcript: List[Dict[str, str]],
    generate_text_fn,
    generate_chat_fn=None,
    model_info: Dict[str, Dict[str, Any]],
    media_url: Optional[str] = None,
    filename: Optional[str] = None,
    assistant_actor: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Post a per-round summary to Discord via MCP 'discord' server.

    generate_text_fn(system_prompt: str, api_model: str, user_message: str) -> str
    """
    # 1) Build summary text via LLM
    api_model = _resolve_model_api(discord_cfg.get("model", "same-as-lm1"), selected_models, model_info)
    # Pull prompts strictly from the provided profile (no fallbacks)
    system_prompt = discord_cfg.get("system_prompt", "")
    # Use the profile system prompt as the system; do not duplicate it in the user body
    # Build a single transcript window from the provided transcript only
    # Assumes this function runs after the round completes and transcript is up-to-date
    win = int(discord_cfg.get("context_window", 50))
    window_src: List[Dict[str, str]] = transcript or []
    windowed = window_src[-win:] if window_src else []
    rendered_transcript = _render_transcript(windowed)

    # No separate instruction; we only send the transcript as the user content
    instruction: str = ""

    user_message = rendered_transcript

    # Build chat-style messages mirroring an agent perspective when possible
    messages: List[Dict[str, str]] = []
    # Map transcript actor names to roles relative to the chosen assistant perspective
    for e in windowed:
        actor = e.get("actor", "")
        text = e.get("text", "")
        role = "assistant" if assistant_actor and actor == assistant_actor else "user"
        messages.append({"role": role, "content": text})

    # Log the exact LLM inputs to a per-run file for debugging (only what is sent)
    if filename:
        try:
            log_path = f"{filename}.discord_llm_request.txt"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n=== Discord Agent LLM Request ===\n")
                f.write(f"Model: {api_model}\n")
                f.write("System:\n")
                f.write(system_prompt + "\n")
                if callable(generate_chat_fn):
                    f.write("Messages (chat array):\n")
                    import json as _json
                    f.write(_json.dumps(messages, ensure_ascii=False) + "\n")
                else:
                    f.write("User message:\n")
                    f.write(user_message + "\n")
        except Exception:
            pass

    # Prefer chat path if provided; fall back to text path
    if callable(generate_chat_fn):
        summary = generate_chat_fn(system_prompt, api_model, messages).strip()
    else:
        summary = generate_text_fn(system_prompt, api_model, user_message).strip()

    # 2) Build tool args from config (strictly opt-in)
    # If the profile omits a tool, do not post anything.
    tool = discord_cfg.get("tool")
    if not isinstance(tool, dict):
        return None
    server_name = tool.get("server")
    tool_name = tool.get("name")
    defaults = tool.get("defaults", {})
    if not server_name or not tool_name:
        return None
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
    # Optional preflight: verify tool exists to surface clearer errors
    try:
        available = list_tools(server_cfg)
        names = {t.get("name") for t in available if isinstance(t, dict)}
        if tool_name not in names:
            raise RuntimeError(
                f"Discord tool '{tool_name}' not found on server '{server_name}'."
            )
    except Exception:
        # If listing fails, proceed to attempt the call for backward compatibility
        pass

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
