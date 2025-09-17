import json
import os
import re
from typing import Any, Dict, List, Optional

from mcp_cli import load_server_config  # reuse config loader
from mcp_client import MCPServerConfig, call_tool, list_tools

from search_eagle_images import get_supabase_client, search_images_semantic


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
        "Keep edit text under 220 characters unless crucial details are needed.",
        "Conversation summary:",
        conversation_summary,
        "",
        "Latest round:",
    ]
    for e in round_entries:
        lines.append(f"- {e.get('actor','')}: {e.get('text','')}")
    lines.append("")
    lines.append("Return only the edit instruction.")
    return "\n".join(lines)


def _supabase_public_base() -> Optional[str]:
    base = os.getenv("SUPABASE_URL", "").rstrip("/")
    if not base:
        return None
    return f"{base}/storage/v1/object/public/eagle-images/"


def _row_to_image_url(row: Dict[str, Any]) -> Optional[str]:
    for key in ("image_url", "imageUrl", "imageurl"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    storage_path = row.get("storage_path") or row.get("path")
    if isinstance(storage_path, str) and storage_path.strip():
        base = _supabase_public_base()
        if base:
            return base + storage_path.strip().lstrip("/")
    return None


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

    @property
    def last_posted_ref(self) -> Optional[str]:
        return self.data.get("last_posted_ref")

    @last_posted_ref.setter
    def last_posted_ref(self, value: Optional[str]):
        if value:
            self.data["last_posted_ref"] = value
        else:
            self.data.pop("last_posted_ref", None)

    @property
    def last_posted_task_id(self) -> Optional[str]:
        return self.data.get("last_posted_task_id")

    @last_posted_task_id.setter
    def last_posted_task_id(self, value: Optional[str]):
        if value:
            self.data["last_posted_task_id"] = value
        else:
            self.data.pop("last_posted_task_id", None)


def parse_result_for_image_ref(result: Dict[str, Any]) -> Optional[str]:
    # Best-effort extraction from MCP result formats
    # 1) Direct fields commonly returned by bespoke servers
    for key in ("image_url", "video_url", "audio_url", "url", "uri", "path"):
        val = result.get(key)
        if isinstance(val, str) and val:
            return val
    # Direct arrays or alt keys at top-level
    for key in ("image_urls", "images", "urls", "result_urls"):
        val = result.get(key)
        if isinstance(val, list):
            # Prefer the most recent URL if arrays are ordered oldest->newest
            cand = None
            for item in reversed(val):
                if isinstance(item, str) and item:
                    cand = item
                    break
                if isinstance(item, dict):
                    u = item.get("imageUrl") or item.get("url") or item.get("uri")
                    if isinstance(u, str) and u:
                        cand = u
                        break
            if cand:
                return cand
    if isinstance(result.get("result_url"), str) and result.get("result_url"):
        return result.get("result_url")
    # 1b) Common nested shapes
    try:
        nested_resp = (result.get("response", {}) or {}).get("data", {})
        for k in ("imageUrl", "image_url", "url", "uri", "result_url", "outputUrl", "output_url"):
            v = nested_resp.get(k)
            if isinstance(v, str) and v:
                return v
        # Arrays: imageUrls, image_urls, urls, images, outputUrls
        for k in ("imageUrls", "image_urls", "urls", "images", "result_urls", "outputUrls", "output_urls"):
            arr = nested_resp.get(k)
            if isinstance(arr, list):
                cand = next((s for s in reversed(arr) if isinstance(s, str) and s), None)
                if cand:
                    return cand
        # Rows array (Supabase execute_sql typical shape: data.rows)
        rows = nested_resp.get("rows")
        if isinstance(rows, list) and rows:
            # Prefer last row's imageUrl-like field
            for row in reversed(rows):
                if isinstance(row, dict):
                    u = (
                        row.get("imageUrl")
                        or row.get("image_url")
                        or row.get("imageurl")
                        or row.get("url")
                        or row.get("uri")
                    )
                    if isinstance(u, str) and u:
                        return u
    except Exception:
        pass
    # 1c) Nested under local_task
    try:
        lt = result.get("local_task", {}) or {}
        for k in ("result_url", "image_url", "url", "uri"):
            if isinstance(lt.get(k), str) and lt.get(k):
                return lt.get(k)
        for k in ("result_urls", "image_urls", "urls", "output_urls", "outputUrls", "images"):
            arr = lt.get(k)
            if isinstance(arr, list):
                cand = next((s for s in reversed(arr) if isinstance(s, str) and s), None)
                if cand:
                    return cand
    except Exception:
        pass

    # 2) Standard MCP content array
    content = result.get("content")
    if isinstance(content, list):
        for item in content:
            uri = item.get("uri") or item.get("url") or item.get("path")
            if isinstance(uri, str) and uri:
                return uri
            txt = item.get("text")
            if isinstance(txt, str):
                m = re.search(r"https?://\S+\.(?:png|jpg|jpeg|webp|gif|mp4|webm|mov|m4v|mp3|wav|flac|ogg)", txt, re.I)
                if m:
                    return m.group(0)
    if isinstance(content, str):
        m = re.search(r"https?://\S+\.(?:png|jpg|jpeg|webp|gif|mp4|webm|mov|m4v|mp3|wav|flac|ogg)", content, re.I)
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
    # Allow separate system prompts per stage, with fallbacks
    base_sp = media_cfg.get("system_prompt", "You create concise, vivid prompts.")
    system_prompt_t2i: str = media_cfg.get("t2i_system_prompt", base_sp)
    system_prompt_edit: str = media_cfg.get("edit_system_prompt", base_sp)
    summary_system_prompt: str = media_cfg.get(
        "summary_system_prompt",
        "You compress conversations into concise visual summaries.",
    )
    if model_info is None:
        model_info = {}
    try:
        api_model = resolve_media_model_api(media_cfg, selected_models, model_info)
    except Exception as e:
        header = "\n\033[1m\033[38;2;180;130;255mMedia Agent (media)\033[0m"
        body = f"Configuration error: {e}"
        log_media_result(filename, header, body)
        return {"isError": True, "content": [{"type": "text", "text": str(e)}]}
    # Mode selection
    # - 't2i': always generate from text
    # - 'edit': always issue edit instructions (requires prior image)
    # - 'chain'|'auto': first round t2i, subsequent rounds edit based on saved last_image_ref
    requested_mode = str(media_cfg.get("mode", "t2i")).lower()
    if requested_mode in ("chain", "auto"):
        mode = "edit" if state.last_image_ref else "t2i"
    else:
        mode = requested_mode

    if mode == "edit":
        # Update conversation summary with recent transcript
        summary_prompt = (
            "Summarize the ongoing conversation so far in <=120 words focusing on enduring entities, scene, and motifs. Be concrete and visual."
        )
        transcript_text = "\n".join([f"- {e.get('actor','')}: {e.get('text','')}" for e in transcript[-10:]])
        summary_user = (
            f"{summary_prompt}\n\nRecent messages to consider:\n{transcript_text}\n\nCurrent summary (may be empty):\n{state.conversation_summary}"
        )
        new_summary = generate_text_fn(summary_system_prompt, api_model, summary_user).strip()
        if new_summary:
            state.conversation_summary = new_summary

        user_content = build_edit_prompt(
            last_image_ref=state.last_image_ref,
            conversation_summary=state.conversation_summary,
            round_entries=round_entries,
        )
        system_prompt = system_prompt_edit
    else:
        # Allow presets to force the user content to be the raw round text
        force_from_round = media_cfg.get("user_from_round")
        if isinstance(force_from_round, str):
            force_from_round = force_from_round.strip().lower()
        if force_from_round in (True, "true", "last", "all"):
            texts = [str(e.get("text", "")) for e in round_entries if str(e.get("text", "")).strip()]
            if texts:
                if force_from_round in ("all",):
                    user_content = "\n".join(texts)
                else:
                    user_content = texts[-1]
            else:
                user_content = ""
            system_prompt = system_prompt_t2i
        elif media_cfg.get("t2i_use_summary", False):
            # Build/update a concise summary to reflect current state
            summary_prompt = (
                "Summarize the ongoing conversation so far in <=120 words focusing on enduring entities, scene, and motifs. Be concrete and visual."
            )
            transcript_text = "\n".join([f"- {e.get('actor','')}: {e.get('text','')}" for e in transcript[-10:]])
            summary_user = (
                f"{summary_prompt}\n\nRecent messages to consider:\n{transcript_text}\n\nCurrent summary (may be empty):\n{state.conversation_summary}"
            )
            new_summary = generate_text_fn(summary_system_prompt, api_model, summary_user).strip()
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
        system_prompt = system_prompt_t2i

    prompt_text = generate_text_fn(system_prompt, api_model, user_content).strip()

    strategy = str(media_cfg.get("strategy") or media_cfg.get("media_strategy") or "tool").lower()
    use_semantic_strategy = strategy in {
        "semantic",
        "semantic_search",
        "semantic-embedding",
        "semantic_embeddings",
    }

    tool: Optional[Dict[str, Any]] = None
    server_name: Optional[str] = None
    tool_name: Optional[str] = None
    debug_args_summary = ""

    if use_semantic_strategy:
        # Normalize prompt into a compact single-line query suited for semantic search
        semantic_query = prompt_text.strip() if isinstance(prompt_text, str) else str(prompt_text)
        try:
            target_n = int(media_cfg.get("n") or media_cfg.get("post_top_k") or 1)
        except Exception:
            target_n = 1
        target_n = max(1, target_n)
        try:
            min_similarity = float(media_cfg.get("min_similarity") or 0.0)
        except Exception:
            min_similarity = 0.0
        folders_cfg = media_cfg.get("folders")
        folders: Optional[List[str]]
        if isinstance(folders_cfg, str):
            folders = [folders_cfg]
        elif isinstance(folders_cfg, (list, tuple)):
            folders = [str(f).strip() for f in folders_cfg if isinstance(f, str) and str(f).strip()]
            folders = folders or None
        else:
            folders = None
        try:
            raw_limit = int(media_cfg.get("semantic_limit") or 0)
        except Exception:
            raw_limit = 0
        if raw_limit <= 0:
            raw_limit = max(target_n * 3, target_n)
        semantic_limit = max(raw_limit, target_n)

        items: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()
        seen_ids: set[str] = set()
        semantic_notes: List[str] = []
        try:
            semantic_rows = search_images_semantic(
                query=semantic_query,
                limit=semantic_limit,
                min_similarity=min_similarity,
                folders=folders,
            )
            semantic_notes.append(f"semantic_hit_count={len(semantic_rows)}")
        except Exception as e:
            semantic_rows = []
            semantic_notes.append(f"semantic_error={e}")
        for row in semantic_rows:
            rid = str(row.get("id") or row.get("eagle_id") or "").strip()
            if rid:
                seen_ids.add(rid)
            url = _row_to_image_url(row)
            if url and url not in seen_urls:
                seen_urls.add(url)
                items.append(
                    {
                        "url": url,
                        "similarity": row.get("similarity_score"),
                        "source": "semantic",
                    }
                )
            if len(items) >= target_n:
                break
        if len(items) < target_n:
            try:
                sb = get_supabase_client()
                extra_limit = max(target_n * 3, 30)
                res = (
                    sb.table("eagle_images")
                    .select("id, eagle_id, image_url, storage_path, created_at")
                    .order("created_at", desc=True)
                    .limit(extra_limit)
                    .execute()
                )
                rows = res.data or []
                for r in rows:
                    rid = str(r.get("id") or r.get("eagle_id") or "").strip()
                    if rid and rid in seen_ids:
                        continue
                    url = _row_to_image_url(r)
                    if not url or url in seen_urls:
                        continue
                    seen_urls.add(url)
                    if rid:
                        seen_ids.add(rid)
                    items.append(
                        {
                            "url": url,
                            "similarity": None,
                            "source": "topoff",
                        }
                    )
                    if len(items) >= target_n:
                        break
            except Exception as e:
                semantic_notes.append(f"topoff_error={e}")
        semantic_notes.append(f"resolved_count={len(items)}/{target_n}")

        rows_payload: List[Dict[str, Any]] = []
        for entry in items:
            row_payload: Dict[str, Any] = {
                "imageUrl": entry["url"],
                "source": entry.get("source"),
            }
            if entry.get("similarity") is not None:
                row_payload["similarity"] = entry.get("similarity")
            rows_payload.append(row_payload)

        # Include all images in the content array so downstream tools can consume arrays directly
        result = {
            "content": (
                [{"type": "image", "uri": entry["url"]} for entry in items]
                if items
                else []
            ),
            "response": {"data": {"rows": rows_payload}},
            "semantic": {
                "query": semantic_query,
                "target": target_n,
                "min_similarity": min_similarity,
                "folders": folders,
                "notes": semantic_notes,
            },
        }
        if items:
            result["images"] = [entry["url"] for entry in items]
        server_name = "semantic"
        tool_name = "search_images_semantic"
        debug_args_summary = f"Strategy: semantic_search (n={target_n}, min_sim={min_similarity:.3f})"
    else:
        if mode == "edit":
            tool = media_cfg.get("edit_tool") or media_cfg.get("tool")
            if not state.last_image_ref:
                tool = media_cfg.get("t2i_tool") or media_cfg.get("tool")
                mode = "t2i"
        else:
            tool = media_cfg.get("t2i_tool") or media_cfg.get("tool")
        if not isinstance(tool, dict):
            err = "Media config must declare a 'tool' with 'server' and 'name'. No defaults are assumed."
            header = "\n\033[1m\033[38;2;180;130;255mMedia Agent (media)\033[0m"
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
        prompt_param = media_cfg.get("prompt_param", "prompt")

        try:
            if isinstance(prompt_text, str) and any(k in (prompt_param or "") for k in ("query", "sql", "command")):
                s = prompt_text.strip()
                if s.startswith("```"):
                    s = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", s)
                    s = re.sub(r"\n?```$", "", s.strip())
                else:
                    m = re.search(r"```[a-zA-Z0-9_-]*\s*\n([\s\S]*?)\n```", s)
                    if m:
                        s = m.group(1).strip()
                prompt_text = s
        except Exception:
            pass

        args = {prompt_param: prompt_text, **defaults}
        # For edit mode, attach the prior image reference using configurable param name/shape
        if mode == "edit" and state.last_image_ref:
            img_param_cfg = tool.get("image_param")
            if isinstance(img_param_cfg, dict):
                param_name = img_param_cfg.get("name") or "image_url"
                as_list = bool(img_param_cfg.get("as_list"))
            elif isinstance(img_param_cfg, str):
                param_name = img_param_cfg
                as_list = False
            else:
                param_name = "image_url"
                as_list = False
            ref_to_use = state.last_image_ref
            args[param_name] = [ref_to_use] if as_list else ref_to_use
            try:
                shown = ref_to_use if not isinstance(ref_to_use, str) else (
                    ref_to_use[:120] + ("…" if len(ref_to_use) > 120 else "")
                )
                debug_args_summary = f"EditParam: {param_name}={'list' if as_list else 'str'} -> {shown}"
            except Exception:
                pass

        mcp_config_path = os.getenv("MCP_CONFIG") or os.getenv("MCP_SERVERS_CONFIG") or "mcp.config.json"
        if not os.path.exists(mcp_config_path):
            alt = "mcp_servers.json"
            if os.path.exists(alt):
                mcp_config_path = alt
        server_cfg: MCPServerConfig = load_server_config(mcp_config_path, server_name)

        tool_exists = True
        try:
            available = list_tools(server_cfg)
            names = {t.get("name") for t in available if isinstance(t, dict)}
            tool_exists = tool_name in names
        except Exception:
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

        wrap_params = tool.get("wrap_params", True)
        payload = {"params": args} if wrap_params else args
        result = call_tool(server_cfg, tool_name, payload)

    # 4) Log
    header = "\n\033[1m\033[38;2;180;130;255mMedia Agent (media)\033[0m"
    dbg = ("\n" + debug_args_summary) if debug_args_summary else ""
    if isinstance(prompt_text, str):
        if "\n" in prompt_text:
            prompt_for_log = "\n".join(f"    {line}" for line in prompt_text.splitlines())
        else:
            prompt_for_log = f"    {prompt_text}"
    else:
        prompt_for_log = f"    {prompt_text!r}"
    body = (
        f"Mode: {mode}{dbg}\n"
        f"Prompt:\n{prompt_for_log}\n"
        f"Result: {json.dumps(result, ensure_ascii=False)}"
    )
    log_media_result(filename, header, body)

    # Helper: extract as many image/media URLs as possible from a tool result
    def _extract_all_image_urls(obj: dict) -> list[str]:
        urls: list[str] = []
        def _add(u):
            if isinstance(u, str) and u and u not in urls:
                urls.append(u)
        try:
            content = obj.get("content")
            if isinstance(content, list):
                for item in content:
                    _add(item.get("uri") or item.get("url") or item.get("path"))
                    t = item.get("text")
                    if isinstance(t, str):
                        import re as _re
                        for m in _re.finditer(r"https?://\S+\.(?:png|jpg|jpeg|webp|gif|mp4|webm|mov|m4v|mp3|wav|flac|ogg)", t, _re.I):
                            _add(m.group(0))
            elif isinstance(content, str):
                import re as _re
                for m in _re.finditer(r"https?://\S+\.(?:png|jpg|jpeg|webp|gif|mp4|webm|mov|m4v|mp3|wav|flac|ogg)", content, _re.I):
                    _add(m.group(0))
        except Exception:
            pass
        # Common top-level keys and arrays
        for key in ("image_url", "url", "uri", "result_url", "output_url", "outputUrl"):
            _add(obj.get(key))
        for key in ("image_urls", "images", "urls", "result_urls", "output_urls", "outputUrls"):
            arr = obj.get(key)
            if isinstance(arr, list):
                for it in arr:
                    if isinstance(it, str):
                        _add(it)
                    elif isinstance(it, dict):
                        _add(it.get("imageUrl") or it.get("url") or it.get("uri"))
        # Nested API response shapes
        try:
            nested_resp = (obj.get("response", {}) or {}).get("data", {})
            for key in ("imageUrl", "url", "uri", "resultUrl", "outputUrl"):
                _add(nested_resp.get(key))
            for key in ("imageUrls", "urls", "images", "resultUrls", "outputUrls"):
                arr = nested_resp.get(key)
                if isinstance(arr, list):
                    for it in arr:
                        if isinstance(it, str):
                            _add(it)
                        elif isinstance(it, dict):
                            _add(it.get("imageUrl") or it.get("url") or it.get("uri"))
            # Rows array from Supabase execute_sql
            rows = nested_resp.get("rows")
            if isinstance(rows, list):
                for row in rows:
                    if isinstance(row, dict):
                        _add(
                            row.get("imageUrl")
                            or row.get("image_url")
                            or row.get("imageurl")
                            or row.get("url")
                            or row.get("uri")
                        )
        except Exception:
            pass
        # local_task nesting (FastMCP servers)
        try:
            lt = obj.get("local_task", {}) or {}
            for key in ("result_url", "image_url", "url", "uri"):
                _add(lt.get(key))
            for key in ("result_urls", "image_urls", "urls", "output_urls", "outputUrls", "images"):
                arr = lt.get(key)
                if isinstance(arr, list):
                    for it in arr:
                        if isinstance(it, str):
                            _add(it)
        except Exception:
            pass
        return urls

    # 5) Try to extract an image ref; if missing, poll status tool when task_id is present
    ref = parse_result_for_image_ref(result)
    # If this is an EDIT call and the immediate result echoes the input image URL,
    # ignore it so we proceed to poll for the actual edited output URL.
    try:
        if (
            mode == "edit"
            and ref
            and isinstance(state.last_image_ref, str)
            and ref == state.last_image_ref
        ):
            ref = None
    except Exception:
        pass
    task_id: Optional[str] = None
    if not ref:
        # Attempt to parse JSON-encoded text content to find taskId
        content = result.get("content")
        texts: list[str] = []
        if isinstance(content, list):
            for item in content:
                t = item.get("text")
                if isinstance(t, str):
                    texts.append(t)
        elif isinstance(content, str):
            texts.append(content)
        for t in texts:
            try:
                data = json.loads(t)
            except Exception:
                continue
            # Common shapes
            task_id = (
                (data.get("response", {}).get("data", {}) or {}).get("taskId")
                or data.get("task_id")
                or data.get("taskId")
            )
            if task_id:
                break

    if not ref and task_id and not use_semantic_strategy:
        status_cfg = tool.get("status_tool") or {"name": "get_task_status"}
        status_name = status_cfg.get("name")
        poll_cfg = tool.get("poll", {})
        max_seconds = int(poll_cfg.get("max_seconds", 60))
        interval = max(1, int(poll_cfg.get("interval", 3)))
        elapsed = 0
        # Poll until an image URL appears or timeout
        while elapsed < max_seconds:
            try:
                # Be liberal in what we send: include both task_id and taskId.
                status_args = {"task_id": task_id, "taskId": task_id}
                # Some FastMCP servers expect params-wrapped input even for status tools.
                s_wrap = bool(status_cfg.get("wrap_params", False))
                status_payload = {"params": status_args} if s_wrap else status_args
                status_result = call_tool(server_cfg, status_name, status_payload)
                # Log heartbeat in file for transparency (truncate to keep logs tidy)
                hb = f"Status({task_id}) -> {json.dumps(status_result, ensure_ascii=False)[:1000]}..."
                with open(filename, "a") as f:
                    f.write("\n" + hb + "\n")
                ref = parse_result_for_image_ref(status_result)
                # Optionally ignore duplicate refs in edit mode; default is to accept same URLs
                if (
                    mode == "edit"
                    and ref
                    and state.last_image_ref
                    and ref == state.last_image_ref
                    and bool(media_cfg.get("dedupe_same_url_in_edit", False))
                ):
                    ref = None
                if not ref:
                    # Also attempt to parse JSON text for 'imageUrl' fields
                    c2 = status_result.get("content")
                    texts2: list[str] = []
                    if isinstance(c2, list):
                        for it in c2:
                            tt = it.get("text")
                            if isinstance(tt, str):
                                texts2.append(tt)
                    elif isinstance(c2, str):
                        texts2.append(c2)
                    for t in texts2:
                        try:
                            d = json.loads(t)
                        except Exception:
                            continue
                        # Prefer API response data if present (KIE uses 'api_response' here)
                        data_obj = (
                            (d.get("response", {}) or {}).get("data", {})
                            or (d.get("api_response", {}) or {}).get("data", {})
                            or {}
                        )
                        # Try direct scalar first (also accept generic 'url')
                        cand = (
                            data_obj.get("imageUrl")
                            or d.get("imageUrl")
                            or data_obj.get("url")
                            or d.get("url")
                            or data_obj.get("resultUrl")
                            or d.get("resultUrl")
                            or data_obj.get("outputUrl")
                            or d.get("outputUrl")
                        )
                        # Try arrays: imageUrls, image_urls, urls, images, result_urls (prefer newest)
                        if not cand:
                            for k in ("imageUrls", "image_urls", "urls", "images", "result_urls", "outputUrls", "output_urls", "resultUrls"):
                                arr = data_obj.get(k) or d.get(k)
                                if isinstance(arr, list):
                                    cand2 = None
                                    for item in reversed(arr):
                                        if isinstance(item, str) and item:
                                            if not state.last_image_ref or item != state.last_image_ref:
                                                cand2 = item
                                                break
                                        elif isinstance(item, dict):
                                            u = item.get("imageUrl") or item.get("url") or item.get("uri")
                                            if isinstance(u, str) and u:
                                                if not state.last_image_ref or u != state.last_image_ref:
                                                    cand2 = u
                                                    break
                                    if cand2:
                                        cand = cand2
                                        break
                        # Fallback to local_task.result_url
                        if not cand:
                            cand = (d.get("local_task", {}) or {}).get("result_url")
                        if isinstance(cand, str) and cand:
                            # Always avoid posting the prior image in edit mode
                            if mode == "edit" and state.last_image_ref and cand == state.last_image_ref:
                                continue
                            ref = cand
                            break
                # Final fallback: regex-scan for any image URL in raw status text, preferring URLs that contain the task_id
                if not ref:
                    import re as _re
                    raw_blob = "\n".join(texts2)
                    # Search for any image/media URL
                    matches = list(_re.finditer(r"https?://\S+\.(?:png|jpg|jpeg|webp|gif|mp4|webm|mov|m4v|mp3|wav|flac|ogg)", raw_blob, _re.I))
                    picked = None
                    for m in matches:
                        u = m.group(0)
                        if isinstance(u, str) and u:
                            # Prefer URLs that include the task_id (new output) and that differ from the prior ref
                            if (task_id and task_id in u) and (not state.last_image_ref or u != state.last_image_ref):
                                picked = u
                                break
                    if not picked and matches:
                        # Fallback to the most recent URL that differs from prior
                        for m in reversed(matches):
                            u = m.group(0)
                            if isinstance(u, str) and u and (not state.last_image_ref or u != state.last_image_ref):
                                picked = u
                                break
                    if picked:
                        ref = picked
                if ref:
                    # Prepend a standard content element so callers can extract easily
                    try:
                        if isinstance(status_result.get("content"), list):
                            status_result["content"].insert(0, {"type": "image", "uri": ref})
                        else:
                            status_result["content"] = [{"type": "image", "uri": ref}]
                    except Exception:
                        pass
                    result = status_result
                    break
            except Exception:
                pass
            try:
                import time as _time
                _time.sleep(interval)
            except Exception:
                break
            elapsed += interval

    # 6) If no media was produced (e.g., refusal/timeouts), optionally provide a placeholder image
    if not ref:
        try:
            if bool(media_cfg.get("post_placeholder", True)):
                # Allow override via media config; else use a tasteful 16:9 dark placeholder
                placeholder_url = (
                    media_cfg.get("placeholder_url")
                    or "https://placehold.co/1024x576/0b0f14/93c5fd?text=Dream%20Simulator%20404"
                )
                ref = placeholder_url
                # Synthesize a minimal MCP-like result so downstream parsing works
                if not isinstance(result, dict):
                    result = {}
                result.setdefault("content", [])
                try:
                    if isinstance(result["content"], list):
                        result["content"].insert(0, {"type": "image", "uri": ref})
                    else:
                        result["content"] = [{"type": "image", "uri": ref}]
                except Exception:
                    pass
                # Log a concise note in the run log
                with open(filename, "a") as f:
                    f.write("\n### Media Agent (Placeholder) ###\n")
                    f.write("No image URL returned by tool/status; using placeholder.\n")
                    f.write(f"Placeholder: {ref}\n")
            else:
                with open(filename, "a") as f:
                    f.write("\n### Media Agent (Skip) ###\n")
                    f.write("No image URL returned by tool/status; skipping placeholder and Discord post.\n")
        except Exception:
            # Last-resort: keep result as-is (may be None)
            pass

    # 7) Update state with newest image reference if present and not a placeholder
    try:
        _ph = media_cfg.get("placeholder_url") or "https://placehold.co/1024x576/0b0f14/93c5fd?text=Dream%20Simulator%20404"
    except Exception:
        _ph = None
    if ref and ref != _ph:
        state.last_image_ref = ref
    state.save()

    # 8) Best-effort JSONL event for diagnostics/consumption
    try:
        media_event = {
            "mode": mode,
            "prompt": prompt_text,
            "task_id": task_id,
            "image_url": ref,
            "server": server_name,
            "tool": tool_name,
        }
        with open(f"{filename}.media.jsonl", "a", encoding="utf-8") as jf:
            jf.write(json.dumps(media_event, ensure_ascii=False) + "\n")
    except Exception:
        pass

    # 9) Optionally post image(s) directly to Discord (media-only responsibility)
    try:
        if ref and media_cfg.get("post_image_to_discord", True):
            # Collect multiple image URLs when requested
            try:
                top_k = int(media_cfg.get("n") or media_cfg.get("post_top_k", 1))
            except Exception:
                top_k = 1
            urls_to_post = [ref]
            if top_k > 1 and isinstance(result, dict):
                try:
                    urls_to_post = _extract_all_image_urls(result)[: top_k]
                except Exception:
                    urls_to_post = [ref]
            # As a fallback for Supabase SQL tools that return JSON text arrays
            # inside content strings, try to extract URLs by parsing the JSON
            # between <untrusted-data> tags when no URLs were found.
            if not urls_to_post:
                try:
                    content = result.get("content")
                    texts: list[str] = []
                    if isinstance(content, list):
                        for it in content:
                            t = it.get("text")
                            if isinstance(t, str):
                                texts.append(t)
                    elif isinstance(content, str):
                        texts.append(content)
                    for t in texts:
                        m = re.search(r"<untrusted-data-[^>]+>\s*(\[.*?\])\s*</untrusted-data-[^>]+>", t, re.S)
                        if not m:
                            continue
                        raw = m.group(1)
                        # Unescape if the block is doubly-quoted
                        try:
                            parsed = json.loads(raw) if isinstance(raw, str) else raw
                        except Exception:
                            # Sometimes the JSON is embedded with backslashes; try to unescape once
                            try:
                                parsed = json.loads(json.loads(f'"{raw}"'))
                            except Exception:
                                parsed = None
                        if isinstance(parsed, list):
                            for row in parsed:
                                if isinstance(row, dict):
                                    u = (
                                        row.get("imageUrl")
                                        or row.get("image_url")
                                        or row.get("imageurl")
                                    )
                                    if isinstance(u, str) and u:
                                        if u not in urls_to_post:
                                            urls_to_post.append(u)
                        if urls_to_post:
                            break
                except Exception:
                    pass
            # Final safety: sanitize to ensure clean image URLs
            try:
                _pat = re.compile(r"https?://[^\s\)\"']+\.(?:png|jpg|jpeg|webp|gif)(?:\?[^\s\"']*)?", re.I)
                cleaned: list[str] = []
                for u in urls_to_post:
                    if not isinstance(u, str):
                        continue
                    m2 = _pat.search(u)
                    if m2:
                        v = m2.group(0)
                        if v not in cleaned:
                            cleaned.append(v)
                if cleaned:
                    urls_to_post = cleaned[: top_k]
            except Exception:
                pass
            # Avoid duplicate single-posts if configured (default: true)
            dedupe = bool(media_cfg.get("dedupe_discord_posts", True))
            if len(urls_to_post) == 1 and dedupe and (
                (task_id and state.last_posted_task_id == task_id)
                or (state.last_posted_ref == ref)
            ):
                with open(filename, "a") as f:
                    f.write("\n### Media Agent (Discord Post) ###\n")
                    f.write("Duplicate detected; skipping Discord post.\n")
                    if task_id:
                        f.write(f"Task: {task_id}\n")
                    f.write(f"Media: {ref}\n")
                return result
            # Support a dry-run mode that logs the intended post without calling Discord
            if media_cfg.get("discord_dry_run", False):
                with open(filename, "a") as f:
                    f.write("\n### Media Agent (Discord Post) ###\n")
                    f.write(f"Channel: {media_cfg.get('discord_channel') or 'media'}\n")
                    for u in urls_to_post:
                        f.write(f"Media: {u}\n")
                    f.write("Result: DRY RUN (not posted)\n")
                # Skip actual Discord call
                # Update last_posted markers so subsequent rounds don't re-attempt
                state.last_posted_ref = urls_to_post[-1]
                state.last_posted_task_id = task_id
                state.save()
                return result
            # Resolve Discord server config and channel
            dserver = media_cfg.get("discord_server", "discord")
            dtool = media_cfg.get("discord_tool", {"name": "send-message"})
            dtool_name = dtool.get("name", "send-message")
            ddefaults = dtool.get("defaults", {})
            channel = media_cfg.get("discord_channel") or ddefaults.get("channel", "media")
            # Optional caption
            caption = media_cfg.get("discord_caption", "")
            caption_to_use = caption if isinstance(caption, str) else ""
            include_prompt = bool(media_cfg.get("discord_include_prompt", False))
            prompt_line = ""
            try:
                if include_prompt and isinstance(prompt_text, str):
                    prompt_line = prompt_text.strip()
            except Exception:
                prompt_line = ""
            # Build final message body
            if caption_to_use and prompt_line:
                message_body = f"{caption_to_use}\n{prompt_line}"
            else:
                message_body = caption_to_use or prompt_line or ""

            last_posted_ref_local = None

            # Load discord MCP server config once
            mcp_config_path = os.getenv("MCP_CONFIG") or os.getenv("MCP_SERVERS_CONFIG") or "mcp.config.json"
            if not os.path.exists(mcp_config_path):
                alt = "mcp_servers.json"
                if os.path.exists(alt):
                    mcp_config_path = alt
            d_server_cfg: MCPServerConfig = load_server_config(mcp_config_path, dserver)

            # Prepare common override application
            def _apply_overrides(args_dict: dict) -> None:
                if "server" in ddefaults:
                    args_dict["server"] = ddefaults["server"]
                try:
                    _ov_env = os.getenv("BACKROOMS_DISCORD_OVERRIDES")
                    if _ov_env:
                        _ov = json.loads(_ov_env)
                        if isinstance(_ov, dict):
                            if isinstance(_ov.get("server"), str) and _ov.get("server").strip():
                                args_dict["server"] = _ov.get("server").strip()
                            if isinstance(_ov.get("channel"), str) and _ov.get("channel").strip():
                                args_dict["channel"] = _ov.get("channel").strip()
                except Exception:
                    pass

            # Post as a single array payload (one message) when multiple URLs
            post_as_array = len(urls_to_post) > 1
            if post_as_array:
                dargs = {"channel": channel, "message": message_body, "mediaUrl": urls_to_post}
                # Add compatibility keys
                if urls_to_post:
                    dargs.setdefault("imageUrl", urls_to_post[0])
                dargs.setdefault("attachments", urls_to_post)
                dargs.setdefault("imageUrls", urls_to_post)
                _apply_overrides(dargs)
                try:
                    d_res = call_tool(d_server_cfg, dtool_name, dargs)
                    with open(filename, "a") as f:
                        f.write("\n### Media Agent (Discord Post) ###\n")
                        f.write(f"Channel: {dargs.get('channel')}\n")
                        f.write(f"Media (array): {json.dumps(urls_to_post, ensure_ascii=False)}\n")
                        f.write(f"Args: {json.dumps({k:v for k,v in dargs.items() if k in ['server','channel','message','mediaUrl','imageUrl','attachments']}, ensure_ascii=False)}\n")
                        f.write(f"Result: {json.dumps(d_res, ensure_ascii=False)}\n")
                    last_posted_ref_local = urls_to_post[-1]
                except Exception as _post_err_arr:
                    # Fallback to per-URL posts if array post fails
                    with open(filename, "a") as f:
                        f.write("\n### Media Agent (Discord Post) ###\n")
                        f.write("Array post failed; falling back to per-URL posts.\n")
                        f.write(f"Error: {repr(_post_err_arr)}\n")
                    for u in urls_to_post:
                        dargs_u = {"channel": channel, "message": message_body, "mediaUrl": u}
                        dargs_u.setdefault("imageUrl", u)
                        dargs_u.setdefault("attachments", [u])
                        _apply_overrides(dargs_u)
                        try:
                            d_res_u = call_tool(d_server_cfg, dtool_name, dargs_u)
                            with open(filename, "a") as f:
                                f.write("\n### Media Agent (Discord Post) ###\n")
                                f.write(f"Channel: {dargs_u.get('channel')}\n")
                                f.write(f"Media: {u}\n")
                                f.write(f"Args: {json.dumps({k:v for k,v in dargs_u.items() if k in ['server','channel','message','mediaUrl','imageUrl']}, ensure_ascii=False)}\n")
                                f.write(f"Result: {json.dumps(d_res_u, ensure_ascii=False)}\n")
                            last_posted_ref_local = u
                        except Exception as _post_err:
                            # Fallback: append URL to the message and try a plain text post
                            try:
                                alt_args = dict(dargs_u)
                                url_text = f"\n{u}" if message_body else u
                                alt_args.pop("mediaUrl", None)
                                alt_args.pop("imageUrl", None)
                                alt_args.pop("attachments", None)
                                alt_args["message"] = (message_body + url_text) if message_body else url_text
                                d_res2 = call_tool(d_server_cfg, dtool_name, alt_args)
                                with open(filename, "a") as f:
                                    f.write("\n### Media Agent (Discord Post) ###\n")
                                    f.write(f"Channel: {alt_args.get('channel')}\n")
                                    f.write(f"Media: {u}\n")
                                    f.write(f"Args: {json.dumps({k:v for k,v in alt_args.items() if k in ['server','channel','message']}, ensure_ascii=False)}\n")
                                    f.write(f"Result (fallback): {json.dumps(d_res2, ensure_ascii=False)}\n")
                                last_posted_ref_local = u
                            except Exception as _post_err2:
                                with open(filename, "a") as f:
                                    f.write("\n### Media Agent (Discord Post) ###\n")
                                    f.write(f"Channel: {dargs_u.get('channel')}\n")
                                    f.write(f"Media: {u}\n")
                                    f.write(f"Args: {json.dumps({k:v for k,v in dargs_u.items() if k in ['server','channel','message','mediaUrl','imageUrl']}, ensure_ascii=False)}\n")
                                    f.write(f"Error: {repr(_post_err)}\n")
                                    f.write(f"Error (fallback): {repr(_post_err2)}\n")
            else:
                # Default behavior: post each URL individually with the composed message
                for u in urls_to_post:
                    dargs = {"channel": channel, "message": message_body, "mediaUrl": u}
                    # Maximize compatibility with different Discord MCP servers
                    # by also providing common alternative keys.
                    dargs.setdefault("imageUrl", u)
                    dargs.setdefault("attachments", [u])
                    _apply_overrides(dargs)
                    # Fire and log minimal outcome with fallback strategy
                    try:
                        d_res = call_tool(d_server_cfg, dtool_name, dargs)
                        with open(filename, "a") as f:
                            f.write("\n### Media Agent (Discord Post) ###\n")
                            f.write(f"Channel: {dargs.get('channel')}\n")
                            f.write(f"Media: {u}\n")
                            f.write(f"Args: {json.dumps({k:v for k,v in dargs.items() if k in ['server','channel','message','mediaUrl','imageUrl']}, ensure_ascii=False)}\n")
                            f.write(f"Result: {json.dumps(d_res, ensure_ascii=False)}\n")
                        last_posted_ref_local = u
                    except Exception as _post_err:
                        # Fallback: append URL to the message and try a plain text post
                        try:
                            alt_args = dict(dargs)
                            url_text = f"\n{u}" if message_body else u
                            alt_args.pop("mediaUrl", None)
                            alt_args.pop("imageUrl", None)
                            alt_args.pop("attachments", None)
                            alt_args["message"] = (message_body + url_text) if message_body else url_text
                            d_res2 = call_tool(d_server_cfg, dtool_name, alt_args)
                            with open(filename, "a") as f:
                                f.write("\n### Media Agent (Discord Post) ###\n")
                                f.write(f"Channel: {alt_args.get('channel')}\n")
                                f.write(f"Media: {u}\n")
                                f.write(f"Args: {json.dumps({k:v for k,v in alt_args.items() if k in ['server','channel','message']}, ensure_ascii=False)}\n")
                                f.write(f"Result (fallback): {json.dumps(d_res2, ensure_ascii=False)}\n")
                            last_posted_ref_local = u
                        except Exception as _post_err2:
                            with open(filename, "a") as f:
                                f.write("\n### Media Agent (Discord Post) ###\n")
                                f.write(f"Channel: {dargs.get('channel')}\n")
                                f.write(f"Media: {u}\n")
                                f.write(f"Args: {json.dumps({k:v for k,v in dargs.items() if k in ['server','channel','message','mediaUrl','imageUrl']}, ensure_ascii=False)}\n")
                                f.write(f"Error: {repr(_post_err)}\n")
                                f.write(f"Error (fallback): {repr(_post_err2)}\n")
            # Remember last posted ref after a successful call
            if last_posted_ref_local:
                state.last_posted_ref = last_posted_ref_local
            state.last_posted_task_id = task_id
            state.save()
    except Exception as _e:
        # Non-fatal: keep the media result even if posting fails
        with open(filename, "a") as f:
            f.write(f"\nMedia Agent (Discord Post) error: {_e}\n")

    # Return the (possibly enriched) result
    return result
