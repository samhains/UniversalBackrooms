#!/usr/bin/env python3
"""
Post N Eagle images to Discord (#lucid) using semantic search.

Usage:
  python scripts/post_lucid_images.py "WORLD IN A DREAM SPIRE" --n 3
  python scripts/post_lucid_images.py "vaporwave neon arcade" --n 5 --min-similarity 0.6 --channel lucid

Requires:
  - OPENAI_API_KEY
  - SUPABASE_URL and SUPABASE_KEY (or SUPABASE_ANON_KEY)
  - mcp.config.json configured with a "discord" server exposing a "send-message" tool
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Ensure project root is on sys.path so we can import top-level modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the semantic search implemented in search_eagle_images.py
from search_eagle_images import search_images_semantic, get_supabase_client

# Minimal MCP client to call the Discord tool directly
from mcp_cli import load_server_config
from mcp_client import MCPServerConfig, call_tool


load_dotenv()


def _supabase_public_base() -> Optional[str]:
    base = os.getenv("SUPABASE_URL", "").rstrip("/")
    if not base:
        return None
    return f"{base}/storage/v1/object/public/eagle-images/"


def _row_to_image_url(row: Dict[str, Any]) -> Optional[str]:
    # Try common key variants
    for k in ("image_url", "imageUrl", "imageurl"):
        val = row.get(k)
        if isinstance(val, str) and val.strip():
            return val.strip()
    # Derive from storage path if present
    storage_path = row.get("storage_path") or row.get("path")
    if isinstance(storage_path, str) and storage_path.strip():
        base = _supabase_public_base()
        if base:
            return base + storage_path.strip().lstrip("/")
        # If base cannot be determined, skip building a URL from storage path
    return None


def _post_image_to_discord(
    *,
    media_url: str,
    channel: str = "lucid",
    message: str = "",
    server_name: str = "discord",
    mcp_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    cfg_path = (
        mcp_config_path
        or os.getenv("MCP_CONFIG")
        or os.getenv("MCP_SERVERS_CONFIG")
        or "mcp.config.json"
    )
    # Try legacy filename if needed
    if not os.path.exists(cfg_path):
        alt = "mcp_servers.json"
        if os.path.exists(alt):
            cfg_path = alt

    server_cfg: MCPServerConfig = load_server_config(cfg_path, server_name)

    # Many Discord MCP servers accept both mediaUrl and imageUrl; include both
    args = {
        "channel": channel,
        "message": message or "",
        "mediaUrl": media_url,
        "imageUrl": media_url,
    }
    return call_tool(server_cfg, "send-message", args)


def run(query: str, n: int, min_similarity: float, channel: str, folders: Optional[List[str]], dry_run: bool) -> int:
    # Fetch images via semantic search
    images = search_images_semantic(
        query=query,
        limit=n,
        min_similarity=min_similarity,
        folders=folders,
    )
    if not images:
        print("No images found.")
        return 1

    # Prepare URL items and track IDs to avoid duplicates during top-off
    # Each item: {"url": str, "similarity": Optional[float], "source": "semantic"|"topoff"}
    items: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_ids: set[str] = set()
    for row in images:
        rid = str(row.get("id") or row.get("eagle_id") or "").strip()
        if rid:
            seen_ids.add(rid)
        url = _row_to_image_url(row)
        if url and url not in seen_urls:
            seen_urls.add(url)
            items.append({
                "url": url,
                "similarity": row.get("similarity_score"),
                "source": "semantic",
            })
        if len(items) >= n:
            break

    if not items:
        print("No image URLs resolvable from results.")
        # Fall back immediately to recent images
        items = []
    
    # Top-off to exactly N using recent images if needed
    if len(items) < n:
        try:
            sb = get_supabase_client()
            # Fetch more than needed to allow for dedupe and missing URLs
            extra_limit = max(n * 3, 30)
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
                u = _row_to_image_url(r)
                if not u:
                    continue
                if u in seen_urls:
                    continue
                seen_urls.add(u)
                items.append({
                    "url": u,
                    "similarity": None,
                    "source": "topoff",
                })
                if rid:
                    seen_ids.add(rid)
                if len(items) >= n:
                    break
        except Exception as e:
            print(f"Top-off fetch failed: {e}")
    
    if len(items) < n:
        print(f"Warning: only resolved {len(items)}/{n} URLs.")

    print(f"Posting {len(items)} image(s) to #{channel}...")
    for i, it in enumerate(items, start=1):
        u = it["url"]
        sim = it.get("similarity")
        src = it.get("source") or "semantic"
        if isinstance(sim, (int, float)):
            print(f" {i:2d}. {u} (sim: {sim:.3f}, {src})")
        else:
            print(f" {i:2d}. {u} (sim: n/a, {src})")
        if dry_run:
            continue
        try:
            _ = _post_image_to_discord(media_url=u, channel=channel, message="")
        except Exception as e:
            print(f"    -> Failed to post: {e}")

    if dry_run:
        print("Dry run complete — no posts made.")
    else:
        print("Done.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Post N Eagle images to Discord using semantic search")
    ap.add_argument("query", help="Short prompt to search for (semantic)")
    ap.add_argument("--n", type=int, default=3, help="Number of images to post (default: 3)")
    ap.add_argument("--min-similarity", type=float, default=0.0, help="Minimum similarity 0.0–1.0 (default: 0.0)")
    ap.add_argument("--channel", default="lucid", help="Discord channel name (default: lucid)")
    ap.add_argument("--folders", nargs="+", help="Optional Eagle folder filters")
    ap.add_argument("--dry-run", action="store_true", help="List URLs but do not post to Discord")

    args = ap.parse_args()
    try:
        return run(
            query=args.query,
            n=max(1, int(args.n)),
            min_similarity=float(args.min_similarity),
            channel=str(args.channel or "lucid"),
            folders=args.folders,
            dry_run=bool(args.dry_run),
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
