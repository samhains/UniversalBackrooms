#!/usr/bin/env python3
"""
Generate a short video from three Eagle images selected via semantic search.

Example:
  python scripts/generate_semantic_video.py "neon arcade in the rain" --min-similarity 0.55 --width 1280 --height 720
  python scripts/generate_semantic_video.py "icy crystalline cathedral" --discord-channel lucid-video

Requirements:
  - Environment variables for Supabase + OpenAI (see search_eagle_images.py)
  - mcp.config.json (or MCP_CONFIG/MCP_SERVERS_CONFIG env) containing a "comfyui" server
  - comfyui MCP server exposing the "generate_3_image_video" tool
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from search_eagle_images import search_images_semantic, get_supabase_client  # noqa: E402
from mcp_cli import load_server_config  # noqa: E402
from mcp_client import MCPServerConfig, call_tool  # noqa: E402

load_dotenv()


@dataclass
class SelectedImage:
    url: str
    similarity: Optional[float] = None
    source: str = "semantic"
    title: Optional[str] = None
    eagle_id: Optional[str] = None


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


def _collect_semantic_images(
    *,
    query: str,
    desired: int,
    min_similarity: float,
    folders: Optional[Sequence[str]] = None,
    semantic_limit: int = 12,
) -> List[SelectedImage]:
    rows = search_images_semantic(
        query=query,
        limit=max(desired, semantic_limit),
        min_similarity=min_similarity,
        folders=list(folders) if folders else None,
    )

    selected: List[SelectedImage] = []
    seen_urls: set[str] = set()
    seen_ids: set[str] = set()
    for row in rows:
        url = _row_to_image_url(row)
        if not url or url in seen_urls:
            continue
        eagle_id = str(row.get("id") or row.get("eagle_id") or "").strip() or None
        if eagle_id:
            seen_ids.add(eagle_id)
        seen_urls.add(url)
        selected.append(
            SelectedImage(
                url=url,
                similarity=(
                    float(row["similarity_score"])
                    if isinstance(row.get("similarity_score"), (int, float))
                    else None
                ),
                source="semantic",
                title=row.get("title"),
                eagle_id=eagle_id,
            )
        )
        if len(selected) >= desired:
            break
    return selected


def _top_off_recent(
    *,
    current: List[SelectedImage],
    desired: int,
    seen_ids: Iterable[str],
    seen_urls: Iterable[str],
) -> List[SelectedImage]:
    try:
        sb = get_supabase_client()
    except Exception as exc:  # pragma: no cover - fallback path
        print(f"Top-off skipped (Supabase unavailable): {exc}")
        return current

    seen_ids_set = {sid for sid in seen_ids if sid}
    seen_urls_set = {url for url in seen_urls if url}
    extra_needed = desired - len(current)
    if extra_needed <= 0:
        return current

    try:
        extra_limit = max(desired * 4, 24)
        res = (
            sb.table("eagle_images")
            .select("id, eagle_id, image_url, storage_path, title, created_at")
            .order("created_at", desc=True)
            .limit(extra_limit)
            .execute()
        )
    except Exception as exc:  # pragma: no cover - best-effort fallback
        print(f"Top-off fetch failed: {exc}")
        return current

    rows = res.data or []
    for row in rows:
        rid = str(row.get("id") or row.get("eagle_id") or "").strip()
        if rid and rid in seen_ids_set:
            continue
        url = _row_to_image_url(row)
        if not url or url in seen_urls_set:
            continue
        seen_ids_set.add(rid)
        seen_urls_set.add(url)
        current.append(
            SelectedImage(
                url=url,
                similarity=None,
                source="recent",
                title=row.get("title"),
                eagle_id=rid or None,
            )
        )
        if len(current) >= desired:
            break
    return current


def _load_server(server_name: str, cfg_path: Optional[str]) -> MCPServerConfig:
    path = (
        cfg_path
        or os.getenv("MCP_CONFIG")
        or os.getenv("MCP_SERVERS_CONFIG")
        or "mcp.config.json"
    )
    if not os.path.exists(path):
        alt = "mcp_servers.json"
        if os.path.exists(alt):
            path = alt
    return load_server_config(path, server_name)


def _call_generate_video(
    *,
    image_urls: Sequence[str],
    width: Optional[int],
    height: Optional[int],
    frame_length: Optional[int],
    server_name: str,
    mcp_config_path: Optional[str],
) -> Dict[str, Any]:
    if len(image_urls) != 3:
        raise ValueError("Exactly three image URLs are required for generate_3_image_video")
    server_cfg = _load_server(server_name, mcp_config_path)

    params: Dict[str, Any] = {
        "image1_url": image_urls[0],
        "image2_url": image_urls[1],
        "image3_url": image_urls[2],
    }
    if isinstance(width, int) and width > 0:
        params["width"] = width
    if isinstance(height, int) and height > 0:
        params["height"] = height
    if isinstance(frame_length, int) and frame_length > 0:
        params["frame_length"] = frame_length

    payload = {"params": params}
    result = call_tool(server_cfg, "generate_3_image_video", payload)
    return result


def _post_video_to_discord(
    *,
    channel: str,
    message: str,
    video_url: str,
    server_name: str,
    config_path: Optional[str],
) -> Dict[str, Any]:
    server_cfg = _load_server(server_name, config_path)
    payload = {
        "channel": channel,
        "message": message,
        "mediaUrl": video_url,
        "videoUrl": video_url,
    }
    return call_tool(server_cfg, "send-message", payload)


def _extract_video_url(result: Dict[str, Any]) -> Optional[str]:
    if not isinstance(result, dict):
        return None

    for key in ("video_url", "url", "uri", "result_url", "output_url"):
        val = result.get(key)
        if isinstance(val, str) and val:
            return val

    content = result.get("content")
    if isinstance(content, dict):
        return _extract_video_url(content)
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if "value" in item and isinstance(item["value"], dict):
                maybe = _extract_video_url(item["value"])
                if maybe:
                    return maybe
            for key in ("uri", "url", "path"):
                val = item.get(key)
                if isinstance(val, str) and val:
                    return val
            text_val = item.get("text")
            if isinstance(text_val, str):
                try:
                    parsed = json.loads(text_val)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    maybe = _extract_video_url(parsed)
                    if maybe:
                        return maybe
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            maybe = _extract_video_url(parsed)
            if maybe:
                return maybe
    return None


def run(
    *,
    query: str,
    min_similarity: float,
    folders: Optional[Sequence[str]],
    width: Optional[int],
    height: Optional[int],
    frame_length: Optional[int],
    dry_run: bool,
    mcp_config_path: Optional[str],
    server_name: str,
    verbose: bool,
    post_to_discord: bool,
    discord_channel: str,
    discord_message: str,
    discord_server_name: str,
    discord_config_path: Optional[str],
) -> int:
    desired = 3
    print(f"Searching for {desired} images matching '{query}' (min similarity {min_similarity:.2f})...")
    selected = _collect_semantic_images(
        query=query,
        desired=desired,
        min_similarity=min_similarity,
        folders=folders,
    )
    raw_seen_ids = [img.eagle_id or "" for img in selected]
    raw_seen_urls = [img.url for img in selected]
    if len(selected) < desired:
        selected = _top_off_recent(
            current=selected,
            desired=desired,
            seen_ids=raw_seen_ids,
            seen_urls=raw_seen_urls,
        )

    if len(selected) < desired:
        print(f"Only resolved {len(selected)} image URL(s). Unable to proceed.")
        return 1

    print("Selected images:")
    for idx, img in enumerate(selected[:desired], start=1):
        sim_text = f"sim {img.similarity:.3f}" if isinstance(img.similarity, float) else "sim n/a"
        print(f" {idx}. {img.url}")
        print(f"    source: {img.source}; {sim_text}; eagle_id: {img.eagle_id or 'n/a'}")
        if verbose and img.title:
            print(f"    title: {img.title}")

    if dry_run:
        print("Dry run requested — skipping video generation.")
        return 0

    print("Calling ComfyUI MCP tool generate_3_image_video...")
    try:
        result = _call_generate_video(
            image_urls=[img.url for img in selected[:desired]],
            width=width,
            height=height,
            frame_length=frame_length,
            server_name=server_name,
            mcp_config_path=mcp_config_path,
        )
    except Exception as exc:
        print(f"Error calling MCP tool: {exc}")
        return 1

    print("Raw MCP response:")
    try:
        print(json.dumps(result, indent=2))
    except TypeError:
        print(result)

    video_url = _extract_video_url(result)
    if video_url:
        print(f"\n✅ Video URL: {video_url}")
        if post_to_discord:
            print(f"Posting video to Discord channel #{discord_channel}...")
            try:
                discord_result = _post_video_to_discord(
                    channel=discord_channel,
                    message=discord_message,
                    video_url=video_url,
                    server_name=discord_server_name,
                    config_path=discord_config_path,
                )
            except Exception as exc:
                print(f"⚠️ Failed to post to Discord: {exc}")
            else:
                try:
                    print(json.dumps(discord_result, indent=2))
                except TypeError:
                    print(discord_result)
                print("✅ Posted to Discord.")
        return 0

    print("⚠️ Unable to locate a video URL in the MCP response.")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select three Eagle images via semantic search and generate a ComfyUI video."
    )
    parser.add_argument("query", help="Semantic search query for Eagle images")
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.0,
        help="Minimum cosine similarity score (default: 0.0)",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        help="Optional Eagle folder filters",
    )
    parser.add_argument("--width", type=int, help="Video width override")
    parser.add_argument("--height", type=int, help="Video height override")
    parser.add_argument(
        "--frame-length",
        type=int,
        default=81,
        help="Video frame count (default: 81)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the image URLs without invoking ComfyUI",
    )
    parser.add_argument(
        "--mcp-config",
        help="Explicit path to MCP server config JSON",
    )
    parser.add_argument(
        "--server-name",
        default="comfyui",
        help="MCP server name providing generate_3_image_video (default: comfyui)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra metadata for selected images",
    )
    parser.add_argument(
        "--no-post",
        action="store_true",
        help="Skip posting the resulting video to Discord",
    )
    parser.add_argument(
        "--discord-channel",
        default="lucid",
        help="Discord channel name for posting (default: lucid)",
    )
    parser.add_argument(
        "--discord-message",
        help="Custom message to accompany the video (default: auto-generated)",
    )
    parser.add_argument(
        "--discord-server",
        default="discord",
        help="MCP server name for Discord posting (default: discord)",
    )
    parser.add_argument(
        "--discord-config",
        help="Alternate MCP config file for the Discord server",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        return run(
            query=args.query,
            min_similarity=float(args.min_similarity),
            folders=args.folders,
            width=args.width,
            height=args.height,
            frame_length=args.frame_length,
            dry_run=bool(args.dry_run),
            mcp_config_path=args.mcp_config,
            server_name=str(args.server_name or "comfyui"),
            verbose=bool(args.verbose),
            post_to_discord=not bool(args.no_post),
            discord_channel=str(args.discord_channel or "lucid"),
            discord_message=(
                args.discord_message
                if isinstance(args.discord_message, str) and args.discord_message.strip()
                else f"{args.query}"
            ),
            discord_server_name=str(args.discord_server or "discord"),
            discord_config_path=args.discord_config,
        )
    except KeyboardInterrupt:
        print("Interrupted by user")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
