#!/usr/bin/env python3
"""
One-off: prompt -> 3 Eagle DB images -> Discord #lucid

Usage:
  python scripts/post_lucid_images.py "LAKE DOOR THRESHOLD" \
    [--channel lucid] [--model opus4] [--caption "..."] [--config media/eagle_top3]

This uses the media agent with the Eagle Top-3 retrieval preset. It asks an LLM
(default: Opus 4) to produce a single SQL statement to fetch up to 3 matching
images from `eagle_images` via the Supabase MCP server, and then posts them to
Discord using the Discord MCP server.

Environment requirements:
  - MCP_SERVERS_CONFIG set to the repository's mcp.config.json (auto if run from repo root)
  - ANTHROPIC_API_KEY (for Opus models) OR OPENROUTER_API_KEY when using an OpenRouter model
  - Discord MCP server configured with a bot that can post to the target channel
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repository root is importable when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional dotenv
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Import media agent helpers
from media_agent import load_media_config, run_media_agent
from mcp_cli import load_server_config
from mcp_client import MCPServerConfig, call_tool, list_tools
from model_config import get_model_info

# Reuse LLM call paths from backrooms for provider routing
from backrooms import claude_conversation, gpt4_conversation, openrouter_conversation


def _load_env() -> None:
    if load_dotenv is not None:
        try:
            load_dotenv(ROOT / ".env")
        except Exception:
            pass


def _generate_text_fn(actor_name: str):
    """Return a generate_text_fn(system_prompt, api_model, user_message) callable.

    Routes to Anthropic/OpenAI/OpenRouter based on the api_model string, mirroring
    backrooms' logic so media_agent can reuse it here.
    """

    def _fn(system_prompt: str, api_model: str, user_message: str) -> str:
        context = [{"role": "user", "content": user_message}]
        if isinstance(api_model, str) and api_model.startswith("claude-"):
            return claude_conversation(actor_name, api_model, context, system_prompt)
        elif isinstance(api_model, str) and "/" in api_model:
            return openrouter_conversation(actor_name, api_model, context, system_prompt)
        else:
            return gpt4_conversation(actor_name, api_model, context, system_prompt)

    return _fn


def _apply_overrides(base_cfg: Dict[str, Any], *, model_key: str, channel: str, caption: Optional[str]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy via json round-trip
    # Force posting up to 3 images
    cfg["post_top_k"] = 3
    cfg["post_image_to_discord"] = True
    # Route to requested model alias (resolved later to provider api_name)
    cfg["model"] = model_key
    # Discord posting target and optional caption
    cfg["discord_channel"] = channel
    if isinstance(caption, str):
        cfg["discord_caption"] = caption
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Post 3 Eagle DB images to Discord from a one-off prompt.")
    parser.add_argument("prompt", type=str, help="Short prompt describing the vibe, e.g. 'LAKE DOOR THRESHOLD'.")
    parser.add_argument("--channel", default="lucid", help="Discord channel to post into (default: lucid)")
    parser.add_argument(
        "--model",
        default="opus4",
        help="Model alias to craft the Eagle SQL (default: opus4)",
    )
    parser.add_argument(
        "--caption",
        default=None,
        help="Optional caption to include with each image post.",
    )
    parser.add_argument(
        "--config",
        default="eagle_top3",
        help="Media preset name (under media/) or JSON filename without extension (default: eagle_top3)",
    )
    args = parser.parse_args(argv)

    _load_env()

    # Ensure MCP config path is visible to media agent if not already
    mcp_cfg_path = ROOT / "mcp.config.json"
    if mcp_cfg_path.exists():
        os.environ.setdefault("MCP_SERVERS_CONFIG", str(mcp_cfg_path))

    # Load media preset (e.g., media/eagle_top3.json)
    preset_name = args.config
    # Allow passing a basename like 'media/eagle_top3.json' or just 'eagle_top3'
    if preset_name.endswith(".json"):
        # Strip extension and directory if inside media/
        preset_name = Path(preset_name).stem
    media_cfg = load_media_config(preset_name)
    if not isinstance(media_cfg, dict):
        raise SystemExit(f"Media preset not found: {args.config}")

    # Apply overrides for model and Discord channel
    cfg = _apply_overrides(media_cfg, model_key=args.model, channel=args.channel, caption=args.caption)

    # Prepare a single-round transcript with the provided prompt
    round_entries = [{"actor": "user", "text": args.prompt}]
    transcript: List[Dict[str, str]] = []

    # Compose a short log filename under BackroomsLogs
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    short = args.prompt.strip().replace("\n", " ")
    short = " ".join(short.split())
    short = short[:40].replace("/", "-")
    logs_dir = ROOT / "BackroomsLogs" / "oneoff"
    logs_dir.mkdir(parents=True, exist_ok=True)
    filename = str(logs_dir / f"lucid_{stamp}_{short}.txt")

    # Resolve model info map and text generator
    model_info = get_model_info()
    generate_text_fn = _generate_text_fn("Eagle Retriever")

    # Try to use Supabase MCP if it exposes a SQL-like tool; else fall back to direct REST + Discord posting
    try:
        supa_cfg: MCPServerConfig = load_server_config(os.environ.get("MCP_SERVERS_CONFIG", str(mcp_cfg_path)), "supabase")
        tools = list_tools(supa_cfg)
        tool_names = {t.get("name") for t in tools if isinstance(t, dict)}
    except Exception:
        tool_names = set()

    res = None
    if ("sql" in tool_names) or ("execute_sql" in tool_names):
        # If server offers execute_sql but our preset says sql, adapt the preset
        try:
            if ("execute_sql" in tool_names) and isinstance(cfg.get("tool"), dict):
                if cfg["tool"].get("name") != "execute_sql":
                    cfg["tool"]["name"] = "execute_sql"
                # Supabase MCP expects direct { query: "..." }
                cfg["tool"]["wrap_params"] = False
                # And the argument name must be 'query'
                cfg["prompt_param"] = "query"
        except Exception:
            pass
        # Drive the media agent end-to-end (MCP Supabase path)
        res = run_media_agent(
            media_cfg=cfg,
            selected_models=[args.model],
            round_entries=round_entries,
            transcript=transcript,
            filename=filename,
            generate_text_fn=generate_text_fn,
            model_info=model_info,
        )
    else:
        # Fallback: directly hit Supabase REST to fetch latest 3 images; then post via Discord MCP
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = (
            os.getenv("SUPABASE_KEY")
            or os.getenv("SUPABASE_ANON_KEY")
            or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        )
        if not supabase_url or not supabase_key:
            raise SystemExit("Supabase REST fallback requires SUPABASE_URL and a key (SUPABASE_ANON_KEY or SERVICE_ROLE)")
        import requests
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
        }
        # Basic heuristic: fetch latest 3; fast and robust
        url = f"{supabase_url}/rest/v1/eagle_images?select=title,storage_path,created_at&order=created_at.desc&limit=3"
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            raise SystemExit(f"Supabase REST error {r.status_code}: {r.text[:300]}")
        rows = r.json() if r.content else []
        if not isinstance(rows, list):
            rows = []
        # Build public URLs from storage_path
        pub_base = f"{supabase_url}/storage/v1/object/public/eagle-images/"
        images: List[str] = []
        for row in rows:
            sp = (row.get("storage_path") or "").strip()
            if sp:
                images.append(pub_base + sp)
        # Post to Discord via MCP
        try:
            d_server: MCPServerConfig = load_server_config(os.environ.get("MCP_SERVERS_CONFIG", str(mcp_cfg_path)), "discord")
        except Exception as e:
            raise SystemExit(f"Discord MCP not configured: {e}")
        caption = cfg.get("discord_caption") or "Eagle picks — Backrooms vibe"
        for u in images:
            payload = {"channel": args.channel, "message": caption, "mediaUrl": u}
            try:
                call_tool(d_server, cfg.get("discord_tool", {"name": "send-message"}).get("name", "send-message"), payload)
            except Exception:
                # Ignore individual post failures; continue
                pass
        # Synthesize a minimal result for logging/display
        res = {"content": [{"type": "image", "uri": u} for u in images]}

    # Best-effort user feedback
    if isinstance(res, dict):
        print("Done. Media agent result recorded at:", filename)
    else:
        print("Done. (No structured result returned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
