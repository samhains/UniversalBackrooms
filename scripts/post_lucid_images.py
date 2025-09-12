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

    # Drive the media agent end-to-end
    res = run_media_agent(
        media_cfg=cfg,
        selected_models=[args.model],
        round_entries=round_entries,
        transcript=transcript,
        filename=filename,
        generate_text_fn=generate_text_fn,
        model_info=model_info,
    )

    # Best-effort user feedback
    if isinstance(res, dict):
        print("Done. Media agent result recorded at:", filename)
    else:
        print("Done. (No structured result returned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

