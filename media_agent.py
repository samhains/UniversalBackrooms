import json
import os
import re
from typing import Any, Dict, List, Optional

from mcp_cli import load_server_config  # reuse config loader
from mcp_client import MCPServerConfig, call_tool, list_tools


def load_media_config(template_name: str) -> Optional[Dict[str, Any]]:
    """Load media config from flat media/ folder first, then fallbacks.

    Lookup order:
    1) media/<template_name>.json
    2) templates/<template_name>/media.json
    3) templates/<template_name>.media.json
    """
    candidates = [
        os.path.join("media", f"{template_name}.json"),
        os.path.join("templates", template_name, "media.json"),
        os.path.join("templates", f"{template_name}.media.json"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if not path:
        return None
    with open(path, "r") as f:
        cfg = json.load(f)
    return cfg


def resolve_media_model_api(media_cfg: Dict[str, Any], selected_models: List[str], model_info: Dict[str, Dict[str, Any]]) -> str:
    """Resolve to provider API model string given media_cfg and selected model keys.
    model_info follows the same shape as backrooms.MODEL_INFO.
    """
    model_key = media_cfg.get("model", "same-as-lm1")
    if model_key == "same-as-lm1":
        for m in selected_models:
            if m in model_info:
                return model_info[m]["api_name"]
        # No implicit fallback; require explicit declaration
        raise ValueError("Unable to resolve media model from selected models; declare 'model' explicitly in media config.")
    if model_key in model_info:
        return model_info[model_key]["api_name"]
    return model_key


def build_t2i_prompt(round_entries: List[Dict[str, str]]) -> str:
    lines = [
        "You are a media-generating sub-agent. Read the latest round and output a concise, vivid text-to-image prompt describing the core scene.",
        "Keep under 200 characters unless critical details require more.",
        "Latest round:",
    ]
    for e in round_entries:
        lines.append(f"- {e.get('actor','')}: {e.get('text','')}")
    lines.append("")
    lines.append("Return only the prompt text.")
    return "\n".join(lines)


def build_edit_prompt(
    *,
    last_image_ref: Optional[str],
    conversation_summary: str,
    round_entries: List[Dict[str, str]],
) -> str:
    lines = [
        "You are an image edit prompt designer. Compare the current round to the conversation summary and produce a brief, actionable edit instruction that keeps the image aligned to the conversation’s essence.",
        "If a base image URL is provided, include it in the response as BASE_IMAGE: <url> on its own line.",
        "Keep edit text under 220 characters unless crucial details are needed.",
        "Conversation summary:",
        conversation_summary,
        "",
        "Latest round:",
    ]
    for e in round_entries:
        lines.append(f"- {e.get('actor','')}: {e.get('text','')}")
    lines.append("")
    if last_image_ref:
        lines.append(f"BASE_IMAGE: {last_image_ref}")
    lines.append("Return only the edit instruction (and BASE_IMAGE line if present).")
    return "\n".join(lines)


class MediaAgentState:
    def __init__(self, state_path: str):
        self.path = state_path
        self.data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f)
        except Exception:
            pass

    @property
    def last_image_ref(self) -> Optional[str]:
        return self.data.get("last_image_ref")

    @last_image_ref.setter
    def last_image_ref(self, value: Optional[str]):
        if value:
            self.data["last_image_ref"] = value
        else:
            self.data.pop("last_image_ref", None)

    @property
    def conversation_summary(self) -> str:
        return self.data.get("conversation_summary", "")

    @conversation_summary.setter
    def conversation_summary(self, value: str):
        self.data["conversation_summary"] = value


def parse_result_for_image_ref(result: Dict[str, Any]) -> Optional[str]:
    # Best-effort extraction from MCP result formats
    # 1) Direct fields commonly returned by bespoke servers
    for key in ("image_url", "video_url", "url", "uri", "path"):
        val = result.get(key)
        if isinstance(val, str) and val:
            return val

    # 2) Standard MCP content array
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            uri = item.get("uri") or item.get("url") or item.get("path")
            if isinstance(uri, str) and uri:
                return uri
            txt = item.get("text")
            if isinstance(txt, str):
                m = re.search(r"https?://\S+\.(?:png|jpg|jpeg|webp|gif)", txt, re.I)
                if m:
                    return m.group(0)
    if isinstance(content, str):
        m = re.search(r"https?://\S+\.(?:png|jpg|jpeg|webp|gif)", content, re.I)
        if m:
            return m.group(0)
    return None


def log_media_result(filename: str, header: str, body: str) -> None:
    print(header)
    print(body)
    with open(filename, "a") as f:
        f.write("\n### Media Agent ###\n")
        f.write(body + "\n")


def run_media_agent(
    *,
    media_cfg: Dict[str, Any],
    selected_models: List[str],
    round_entries: List[Dict[str, str]],
    transcript: List[Dict[str, str]],
    filename: str,
    generate_text_fn,
    model_info: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
    """
    generate_text_fn(system_prompt: str, api_model: str, user_message: str) -> str
    """

    # 0) Load state tied to this run (logfile)
    state = MediaAgentState(state_path=f"{filename}.media_state.json")

    # 1) Build image prompt text via LLM
    system_prompt: str = media_cfg.get("system_prompt", "You create concise, vivid prompts.")
    if model_info is None:
        model_info = {}
    try:
        api_model = resolve_media_model_api(media_cfg, selected_models, model_info)
    except Exception as e:
        header = "\n\033[1m\033[38;2;180;130;255mMedia Agent (image)\033[0m"
        body = f"Configuration error: {e}"
        log_media_result(filename, header, body)
        return {"isError": True, "content": [{"type": "text", "text": str(e)}]}
    mode = media_cfg.get("mode", "t2i")  # 't2i' or 'edit'

    if mode == "edit":
        # Update conversation summary with recent transcript
        summary_prompt = (
            "Summarize the ongoing conversation so far in <=120 words focusing on enduring entities, scene, and motifs. Be concrete and visual."
        )
        transcript_text = "\n".join([f"- {e.get('actor','')}: {e.get('text','')}" for e in transcript[-10:]])
        summary_user = (
            f"{summary_prompt}\n\nRecent messages to consider:\n{transcript_text}\n\nCurrent summary (may be empty):\n{state.conversation_summary}"
        )
        new_summary = generate_text_fn(
            "You compress conversations into concise visual summaries.", api_model, summary_user
        ).strip()
        if new_summary:
            state.conversation_summary = new_summary

        user_content = build_edit_prompt(
            last_image_ref=state.last_image_ref,
            conversation_summary=state.conversation_summary,
            round_entries=round_entries,
        )
    else:
        if media_cfg.get("t2i_use_summary", False):
            # Build/update a concise summary to reflect current state
            summary_prompt = (
                "Summarize the ongoing conversation so far in <=120 words focusing on enduring entities, scene, and motifs. Be concrete and visual."
            )
            transcript_text = "\n".join([f"- {e.get('actor','')}: {e.get('text','')}" for e in transcript[-10:]])
            summary_user = (
                f"{summary_prompt}\n\nRecent messages to consider:\n{transcript_text}\n\nCurrent summary (may be empty):\n{state.conversation_summary}"
            )
            new_summary = generate_text_fn(
                "You compress conversations into concise visual summaries.", api_model, summary_user
            ).strip()
            if new_summary:
                state.conversation_summary = new_summary
                state.save()
            # Ask for a t2i prompt from the summary + latest round
            lines = [
                "Create a concise T2I prompt that captures the essence of the conversation so far and the latest round.",
                "Keep under 200 characters unless essential.",
                "Conversation summary:",
                state.conversation_summary,
                "",
                "Latest round:",
            ]
            for e in round_entries:
                lines.append(f"- {e.get('actor','')}: {e.get('text','')}")
            lines.append("")
            lines.append("Return only the prompt text.")
            user_content = "\n".join(lines)
        else:
            user_content = build_t2i_prompt(round_entries)

    prompt_text = generate_text_fn(system_prompt, api_model, user_content).strip()

    # 2) Prepare MCP tool call
    tool = media_cfg.get("tool")
    if not isinstance(tool, dict):
        err = "Media config must declare a 'tool' with 'server' and 'name'. No defaults are assumed."
        header = "\n\033[1m\033[38;2;180;130;255mMedia Agent (image)\033[0m"
        body = f"Error: {err}"
        log_media_result(filename, header, body)
        return {"isError": True, "content": [{"type": "text", "text": err}]}

    server_name = tool.get("server")
    tool_name = tool.get("name")
    if not server_name or not tool_name:
        err = "Media tool must include both 'server' and 'name'."
        header = "\n\033[1m\033[38;2;180;130;255mMedia Agent (image)\033[0m"
        body = f"Error: {err}"
        log_media_result(filename, header, body)
        return {"isError": True, "content": [{"type": "text", "text": err}]}

    defaults = tool.get("defaults", {})
    args = {"prompt": prompt_text, **defaults}

    # Load MCP server config
    mcp_config_path = os.getenv("MCP_CONFIG", "mcp.config.json")
    server_cfg: MCPServerConfig = load_server_config(mcp_config_path, server_name)

    # 3) Validate tool exists and call
    # Try to list tools first to provide a clearer error if mis-typed
    tool_exists = True
    try:
        available = list_tools(server_cfg)
        names = {t.get("name") for t in available if isinstance(t, dict)}
        tool_exists = tool_name in names
    except Exception:
        # If listing fails, proceed to attempt the call (original behavior)
        tool_exists = True

    if not tool_exists:
        err = (
            f"Media tool '{tool_name}' not found on server '{server_name}'. "
            f"Run: python mcp_cli.py --config {mcp_config_path} --server {server_name} list-tools"
        )
        header = "\n\033[1m\033[38;2;180;130;255mMedia Agent (image)\033[0m"
        body = f"Mode: {mode}\nPrompt: {prompt_text}\nError: {err}"
        log_media_result(filename, header, body)
        state.save()
        return {"isError": True, "content": [{"type": "text", "text": err}]}

    # Many FastMCP servers define a single parameter named 'params'.
    # Wrap arguments accordingly for compatibility.
    result = call_tool(server_cfg, tool_name, {"params": args})

    # 4) Log
    header = "\n\033[1m\033[38;2;180;130;255mMedia Agent (image)\033[0m"
    body = f"Mode: {mode}\nPrompt: {prompt_text}\nResult: {json.dumps(result, ensure_ascii=False)}"
    log_media_result(filename, header, body)

    # 5) Update state with newest image reference if present
    ref = parse_result_for_image_ref(result)
    if ref:
        state.last_image_ref = ref
    state.save()

    # Optionally inject a short note into contexts in the caller if desired later
    return result
