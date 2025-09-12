#!/usr/bin/env python3
"""
One-off: prompt -> N Eagle DB images -> Discord #lucid

Usage:
  python scripts/post_lucid_images.py "LAKE DOOR THRESHOLD" \
    [--channel lucid] [--caption "..."] [--config eagle_top3_fts] [--n 3] \
    [--semantic] [--vector-column caption_embedding] [--embed-model text-embedding-3-small] [--no-llm]

This can either:
  Priority is a pure embedding-based semantic search (pgvector) — no LLM needed.
  If embeddings are not available, we fall back to a deterministic fused
  FTS+trigram SQL. LLM-driven SQL remains optional via --use-llm.

  Retrieval modes:
  - Semantic (default when OPENAI_API_KEY present):
      1) Compute prompt embedding via OpenAI embeddings.
      2) ORDER BY e.<vector-column> <-> '[...]'::vector LIMIT N.
  - Deterministic (no-LLM):
      Fused FTS + trigram + small recency, always returns rows.
  - LLM (optional):
      Ask a model to write SQL (kept for compatibility; not recommended).

Refinement: `--n` controls the exact number of images to post. If the agent
returns fewer than N, we "top off" by fetching the latest images from Supabase
REST to reach N, best-effort.

Environment requirements:
  - MCP_SERVERS_CONFIG set to the repository's mcp.config.json (auto if run from repo root)
  - ANTHROPIC_API_KEY (for Anthropic models) OR OPENROUTER_API_KEY when using an OpenRouter model
  - Discord MCP server configured with a bot that can post to the target channel
  - SUPABASE_URL and a key (SUPABASE_ANON_KEY or SERVICE_ROLE) for REST fallback/top-off
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import re
import unicodedata

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

# Embeddings
try:
    from openai import OpenAI as _OpenAI
except Exception:  # pragma: no cover
    _OpenAI = None


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


def _apply_overrides(
    base_cfg: Dict[str, Any], *, model_key: str, channel: str, caption: Optional[str], n: Optional[int] = None
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_cfg))  # deep copy via json round-trip
    # Desired number of images
    if not (isinstance(n, int) and n > 0):
        n = int(cfg.get("post_top_k", 3))
    cfg["post_top_k"] = n
    cfg["post_image_to_discord"] = True
    # Route to requested model alias (resolved later to provider api_name)
    cfg["model"] = model_key
    # Discord posting target and optional caption
    cfg["discord_channel"] = channel
    if isinstance(caption, str):
        cfg["discord_caption"] = caption
    # Adjust retrieval prompt to reflect N
    sp = cfg.get("system_prompt")
    if isinstance(sp, str) and sp.strip():
        # Replace common numeric directives like "THREE" and LIMIT 3 with N
        try:
            import re as _re
            # Replace spelled-out THREE occurrences with N
            sp2 = _re.sub(r"\bTHREE\b", str(n), sp)
            sp2 = _re.sub(r"\bthree\b", str(n), sp2)
            sp2 = _re.sub(r"\bThree\b", str(n), sp2)
            # Replace phrases like 'up to 3' or 'LIMIT 3' with N
            sp2 = _re.sub(r"up to\s+\d+", f"up to {n}", sp2, flags=_re.I)
            sp2 = _re.sub(r"LIMIT\s+\d+", f"LIMIT {n}", sp2)
            # Add an explicit instruction so the model aligns to N
            extra = f"\n\nInstruction override: Return up to {n} rows and set LIMIT {n}. If few matches exist, still produce the single SELECT with LIMIT {n} and rely on ordering heuristics."
            cfg["system_prompt"] = sp2 + extra
        except Exception:
            pass
    return cfg


def _normalize_text(text: str) -> str:
    s = text or ""
    # Lowercase and strip accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    # Replace non-word with spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_STOPWORDS = set(
    """
    a an and the of to in on at for from with without within into onto by is are was were be been being this that these those it its as if then else when while do did done doing over under again further so very just not nor only own same than too can will would could should shall might must may about above after against all am any because before below both but down during each few more most other out our ours ourselves some such their theirs them themselves there through until up we you your yours yourself yourselves he she they i me my myself him his her hers itself who whom why how what which where
    """.split()
)


def _summarize_prompt(prompt: str, *, max_tokens: int = 12) -> str:
    """Generate a lightweight, deterministic summary/keyword string.

    No LLM: normalize, drop short words/stopwords, keep unique order.
    """
    norm = _normalize_text(prompt)
    if not norm:
        return ""
    tokens: List[str] = []
    seen: Set[str] = set()
    for w in norm.split():
        if len(w) <= 2:
            continue
        if w in _STOPWORDS:
            continue
        if w not in seen:
            seen.add(w)
            tokens.append(w)
        if len(tokens) >= max_tokens:
            break
    return " ".join(tokens)


def _openai_embed(text: str, *, model: str = "text-embedding-3-small") -> List[float]:
    """Return an embedding vector for the given text using OpenAI embeddings.

    Requires OPENAI_API_KEY. This is not an LLM call.
    """
    if _OpenAI is None:
        raise RuntimeError("openai package not available; cannot compute embeddings")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set; cannot compute embeddings")
    client = _OpenAI(api_key=api_key)
    # Strip to keep payload small and consistent
    text = (text or "").strip()
    if not text:
        text = " "
    try:
        emb = client.embeddings.create(model=model, input=text)
        vec = emb.data[0].embedding  # type: ignore[attr-defined]
        if not isinstance(vec, list) or not vec:
            raise RuntimeError("Empty embedding returned")
        return [float(x) for x in vec]
    except Exception as e:
        raise RuntimeError(f"Embedding failed: {e}")


def _vector_literal(vec: List[float]) -> str:
    """Format a Python list[float] for pgvector: '[v1,v2,...]'::vector.

    Keep a reasonable precision to avoid huge SQL strings.
    """
    # Limit precision to 6 decimal places
    parts = [format(float(x), ".6f") for x in vec]
    return "[" + ",".join(parts) + "]::vector"


def _build_vector_sql(*, prompt_vec: List[float], limit: int, pub_base: str, vector_column: str) -> str:
    """Build a pgvector semantic search SQL over eagle_images.

    Orders by vector distance asc, then recency. No WHERE.
    """
    base = pub_base.rstrip("/") + "/"
    concat_expr = f"concat('{base}', e.storage_path)"
    vec_lit = _vector_literal(prompt_vec)
    col = re.sub(r"[^a-zA-Z0-9_]+", "", vector_column or "caption_embedding")
    sql = f"""
SELECT
  e.id,
  e.title,
  {concat_expr} AS imageUrl
FROM eagle_images e
ORDER BY e.{col} <-> {vec_lit} ASC,
         e.created_at DESC
LIMIT {int(max(1, limit))};
""".strip()
    return sql


def _sql_literal(s: str) -> str:
    """Escape a Python string as a single-quoted SQL literal."""
    return "'" + (s or "").replace("'", "''") + "'"


def _build_fused_sql(*, prompt: str, summary: str, limit: int, pub_base: str, include_caption: bool = True) -> str:
    """Build deterministic always-return SQL with fused scoring.

    If include_caption=False, omits caption to handle schemas without it.
    """
    P = _sql_literal(prompt)
    S = _sql_literal(summary or prompt)
    base = pub_base.rstrip("/") + "/"
    concat_expr = f"concat('{base}', e.storage_path)"

    if include_caption:
        tsv_expr = (
            "setweight(to_tsvector('english', unaccent(coalesce(e.caption,''))), 'A') || "
            "setweight(to_tsvector('simple',  unaccent(array_to_string(e.tags,' '))), 'B') || "
            "setweight(to_tsvector('simple',  unaccent(coalesce(e.title,''))), 'C')"
        )
        cap_sim = "similarity(unaccent(coalesce(e.caption,'')), unaccent(" + S + ")) AS cap_sim,"
        cap_weight = "0.8*COALESCE(s.cap_sim,0) + "
    else:
        tsv_expr = (
            "setweight(to_tsvector('simple',  unaccent(array_to_string(e.tags,' '))), 'B') || "
            "setweight(to_tsvector('simple',  unaccent(coalesce(e.title,''))), 'C')"
        )
        cap_sim = ""
        cap_weight = ""

    sql = f"""
SELECT
  e.id,
  e.title,
  {concat_expr} AS imageUrl
FROM eagle_images e
CROSS JOIN LATERAL (
  SELECT
    {tsv_expr} AS tsv,
    websearch_to_tsquery('english', unaccent({P})) AS q_web,
    plainto_tsquery('simple',  unaccent({P}))       AS q_plain,
    {cap_sim}
    similarity(unaccent(coalesce(e.title,'')),   unaccent({S})) AS title_sim,
    (SELECT max(similarity(unaccent(tag), unaccent({S}))) FROM unnest(e.tags) AS tag) AS tag_sim
) s
ORDER BY
  (0.9*COALESCE(ts_rank_cd(s.tsv, s.q_web), 0) + 0.8*COALESCE(ts_rank_cd(s.tsv, s.q_plain), 0)) +
  ({cap_weight}0.6*COALESCE(s.tag_sim,0) + 0.4*COALESCE(s.title_sim,0)) +
  (0.05 * (extract(epoch from (now() - e.created_at)) * -1.0 / 86400.0)) +
  (0.02*random()) DESC,
  e.created_at DESC
LIMIT {int(max(1, limit))};
""".strip()
    return sql


def _fetch_latest_image_urls(*, supabase_url: str, supabase_key: str, limit: int, exclude: Set[str] | None = None) -> List[str]:
    import requests
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
    }
    # Over-fetch a bit to allow exclusion filtering
    fetch_limit = max(limit * 2, limit)
    url = f"{supabase_url}/rest/v1/eagle_images?select=title,storage_path,created_at&order=created_at.desc&limit={fetch_limit}"
    r = requests.get(url, headers=headers, timeout=20)
    rows = r.json() if (r.status_code == 200 and r.content) else []
    pub_base = f"{supabase_url}/storage/v1/object/public/eagle-images/"
    images: List[str] = []
    ex = exclude or set()
    if isinstance(rows, list):
        for row in rows:
            sp = (row.get("storage_path") or "").strip()
            if sp:
                u = pub_base + sp
                if u not in ex and u not in images:
                    images.append(u)
            if len(images) >= limit:
                break
    return images


def _post_images_to_discord(
    *,
    urls: List[str],
    channel: str,
    caption: str,
    media_cfg: Dict[str, Any],
    mcp_cfg_path: Path,
    filename: str,
    try_batch: bool = True,
) -> None:
    """Robustly post a list of image URLs to Discord via MCP.

    Tries mediaUrl + imageUrl + attachments, then falls back to plain text with URL.
    Mirrors media_agent's compatibility strategy.
    """
    try:
        d_server: MCPServerConfig = load_server_config(os.environ.get("MCP_SERVERS_CONFIG", str(mcp_cfg_path)), "discord")
    except Exception as e:
        with open(filename, "a") as f:
            f.write(f"\n### Discord Post Error ###\nCould not load Discord MCP: {e}\n")
        return
    dtool = media_cfg.get("discord_tool", {"name": "send-message"})
    dtool_name = dtool.get("name", "send-message")
    posted = 0
    # URL sanitizer (match a single valid-looking image URL)
    _pat = re.compile(r"https?://[^\s\)\"']+\.(?:png|jpg|jpeg|webp|gif)(?:\?[^\s\"']*)?", re.I)
    # Sanitize and dedupe
    clean_urls: List[str] = []
    for u in urls:
        # Sanitize to a clean image URL
        clean_u = None
        if isinstance(u, str):
            m = _pat.search(u)
            if m:
                clean_u = m.group(0)
        if not clean_u:
            with open(filename, "a") as f:
                f.write("\n### Discord Post Skip (no clean URL) ###\n")
                f.write(f"Raw: {repr(u)}\n")
            continue
        if clean_u not in clean_urls:
            clean_urls.append(clean_u)

    # Try to post all attachments in a single message if supported
    if try_batch and len(clean_urls) > 1:
        first = clean_urls[0]
        msg = caption if caption else ""
        try:
            batch_args = {
                "channel": channel,
                "message": msg,
                "mediaUrl": first,
                "imageUrl": first,
                "attachments": clean_urls,
            }
            d_res = call_tool(d_server, dtool_name, batch_args)
            posted = len(clean_urls)
            with open(filename, "a") as f:
                f.write("\n### Discord Post (batch) ###\n")
                f.write(f"Channel: {channel}\n")
                f.write(f"Media: {', '.join(clean_urls)}\n")
                f.write(f"Args: {json.dumps({k:v for k,v in batch_args.items() if k in ['channel','message','mediaUrl','imageUrl','attachments']}, ensure_ascii=False)}\n")
                f.write(f"Result: {json.dumps(d_res, ensure_ascii=False)}\n")
            return
        except Exception as e:
            with open(filename, "a") as f:
                f.write("\n### Discord Post (batch) failed, falling back per-image ###\n")
                f.write(f"Error: {repr(e)}\n")

    # Per-image fallback
    for clean_u in clean_urls:
        dargs = {"channel": channel, "message": caption, "mediaUrl": clean_u}
        dargs.setdefault("imageUrl", clean_u)
        dargs.setdefault("attachments", [clean_u])
        try:
            d_res = call_tool(d_server, dtool_name, dargs)
            posted += 1
            with open(filename, "a") as f:
                f.write("\n### Discord Post ###\n")
                f.write(f"Channel: {channel}\n")
                f.write(f"Media: {clean_u}\n")
                f.write(f"Args: {json.dumps({k:v for k,v in dargs.items() if k in ['channel','message','mediaUrl','imageUrl']}, ensure_ascii=False)}\n")
                f.write(f"Result: {json.dumps(d_res, ensure_ascii=False)}\n")
        except Exception:
            try:
                alt_msg = (caption + "\n" + clean_u) if caption else clean_u
                alt = {"channel": channel, "message": alt_msg}
                d_res2 = call_tool(d_server, dtool_name, alt)
                posted += 1
                with open(filename, "a") as f:
                    f.write("\n### Discord Post (fallback) ###\n")
                    f.write(f"Channel: {channel}\n")
                    f.write(f"Media: {clean_u}\n")
                    f.write(f"Args: {json.dumps({k:v for k,v in alt.items() if k in ['channel','message']}, ensure_ascii=False)}\n")
                    f.write(f"Result: {json.dumps(d_res2, ensure_ascii=False)}\n")
            except Exception as e2:
                with open(filename, "a") as f:
                    f.write("\n### Discord Post Error ###\n")
                    f.write(f"Channel: {channel}\n")
                    f.write(f"Media: {clean_u}\n")
                    f.write(f"Error: {repr(e2)}\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Post N Eagle DB images to Discord from a one-off prompt.")
    parser.add_argument("prompt", type=str, help="Short prompt describing the vibe, e.g. 'LAKE DOOR THRESHOLD'.")
    parser.add_argument("--channel", default="lucid", help="Discord channel to post into (default: lucid)")
    # Model only used for optional LLM path; not required for semantic/FTS
    parser.add_argument(
        "--model",
        default="haiku3",
        help="Model alias for optional LLM SQL path (default: haiku3)",
    )
    parser.add_argument(
        "--caption",
        default=None,
        help="Optional caption to include with each image post.",
    )
    parser.add_argument(
        "--config",
        default="eagle_top3_fts",
        help="Media preset name (under media/) or JSON filename without extension (default: eagle_top3_fts)",
    )
    # Semantic-embedding path controls
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Prefer semantic search with embeddings + pgvector (no LLM). Defaults on when OPENAI_API_KEY exists.",
    )
    parser.add_argument(
        "--vector-column",
        default="caption_embedding",
        help="pgvector column on eagle_images to search (default: caption_embedding)",
    )
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        help="OpenAI embedding model to use (default: text-embedding-3-small)",
    )
    # Retrieval mode selection
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Force deterministic fused SQL (caption+tags+title) via Supabase MCP; do not ask an LLM to write SQL.",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use the LLM-driven SQL preset (overrides default deterministic path).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Exact number of images to post (best-effort, tops up if needed).",
    )
    # Backward compatibility: map deprecated --top-k to --n
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
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
    # Resolve desired N (prefer --n; allow deprecated --top-k)
    n = int(args.n or 0)
    if args.top_k is not None and args.top_k > 0:
        n = int(args.top_k)
        print(f"[deprecation] --top-k is deprecated; use --n {n} instead.")
    if n <= 0:
        n = 1
    cfg = _apply_overrides(media_cfg, model_key=args.model, channel=args.channel, caption=args.caption, n=n)

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
    # Preferred modes: semantic (if OPENAI_API_KEY or --semantic), else deterministic FTS; LLM only when explicitly requested
    use_llm = bool(args.use_llm) and not bool(args.no_llm)
    prefer_semantic = bool(args.semantic or os.getenv("OPENAI_API_KEY")) and not use_llm
    if (not use_llm) and (("sql" in tool_names) or ("execute_sql" in tool_names)):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = (
            os.getenv("SUPABASE_KEY")
            or os.getenv("SUPABASE_ANON_KEY")
            or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        )
        if not supabase_url:
            raise SystemExit("Semantic/FTS path requires SUPABASE_URL for public image URL prefix and potential fallback")

        pub_base = f"{supabase_url}/storage/v1/object/public/eagle-images/"

        # Helper to extract URLs from result (reuse logic below)
        def _extract_urls_from_result(r: Dict[str, Any]) -> List[str]:
            urls: List[str] = []
            try:
                content = r.get("content")
                if isinstance(content, list):
                    for it in content:
                        u = it.get("uri") or it.get("url")
                        if isinstance(u, str) and u:
                            urls.append(u)
                        t = it.get("text")
                        if isinstance(t, str):
                            import re as _re
                            for m in _re.finditer(r"https?://\S+\.(?:png|jpg|jpeg|webp|gif)", t, _re.I):
                                urls.append(m.group(0))
                data = (r.get("response", {}) or {}).get("data", {})
                rows = data.get("rows")
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict):
                            u = row.get("imageUrl") or row.get("image_url") or row.get("imageurl")
                            if isinstance(u, str) and u:
                                urls.append(u)
            except Exception:
                pass
            dedup: List[str] = []
            for u in urls:
                if u not in dedup:
                    dedup.append(u)
            return dedup

        # Prefer execute_sql tool if available
        tool_name = "execute_sql" if ("execute_sql" in tool_names) else "sql"
        urls: List[str] = []

        if prefer_semantic:
            # Try semantic vector search first
            try:
                vec = _openai_embed(args.prompt, model=args.embed_model)
                sql = _build_vector_sql(prompt_vec=vec, limit=n, pub_base=pub_base, vector_column=args.vector_column)
                # Log SQL for traceability
                try:
                    with open(filename, "a") as f:
                        f.write("\n### SQL (semantic) ###\n")
                        f.write(sql + "\n")
                except Exception:
                    pass
                res = call_tool(supa_cfg, tool_name, {"query": sql})
                urls = _extract_urls_from_result(res) if isinstance(res, dict) else []
                if urls:
                    try:
                        with open(filename, "a") as f:
                            f.write(f"URLs: {len(urls)}\n")
                    except Exception:
                        pass
            except Exception:
                urls = []

        # If semantic failed or not preferred, try deterministic FTS/trigram
        if not urls:
            try:
                summary = _summarize_prompt(args.prompt)
                sql = _build_fused_sql(prompt=args.prompt, summary=summary, limit=n, pub_base=pub_base, include_caption=True)
                try:
                    with open(filename, "a") as f:
                        f.write("\n### SQL (fts+trgm, caption) ###\n")
                        f.write(sql + "\n")
                except Exception:
                    pass
                res = call_tool(supa_cfg, tool_name, {"query": sql})
                urls = _extract_urls_from_result(res) if isinstance(res, dict) else []
            except Exception:
                urls = []
        if not urls:
            try:
                summary = _summarize_prompt(args.prompt)
                sql = _build_fused_sql(prompt=args.prompt, summary=summary, limit=n, pub_base=pub_base, include_caption=False)
                try:
                    with open(filename, "a") as f:
                        f.write("\n### SQL (fts+trgm, no caption) ###\n")
                        f.write(sql + "\n")
                except Exception:
                    pass
                res = call_tool(supa_cfg, tool_name, {"query": sql})
                urls = _extract_urls_from_result(res) if isinstance(res, dict) else []
            except Exception:
                urls = []

        # Fallback to REST latest if needed
        if not urls:
            if not supabase_key:
                raise SystemExit("No results and no Supabase key available for REST fallback.")
            urls = _fetch_latest_image_urls(supabase_url=supabase_url, supabase_key=supabase_key, limit=n)

        # Post to Discord via MCP (robust)
        caption = args.caption or ""
        chosen = urls[:n]
        _post_images_to_discord(
            urls=chosen, channel=args.channel, caption=caption, media_cfg=media_cfg, mcp_cfg_path=mcp_cfg_path, filename=filename
        )

        # Top-off to exactly N if we posted fewer than requested
        try:
            need_more = max(0, n - len(chosen))
            if need_more > 0 and supabase_key:
                extra = _fetch_latest_image_urls(
                    supabase_url=supabase_url, supabase_key=supabase_key, limit=need_more, exclude=set(chosen)
                )
                if extra:
                    _post_images_to_discord(
                        urls=extra,
                        channel=args.channel,
                        caption=caption,
                        media_cfg=media_cfg,
                        mcp_cfg_path=mcp_cfg_path,
                        filename=filename,
                    )
                    chosen = chosen + extra
        except Exception:
            pass

        # Synthesize a minimal result for logging/display
        res = {"content": [{"type": "image", "uri": u} for u in chosen]}

    elif ("sql" in tool_names) or ("execute_sql" in tool_names):
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
        # If no URLs found or error returned, fall back to REST to ensure we post images
        def _extract_urls_from_result(r: Dict[str, Any]) -> List[str]:
            urls: List[str] = []
            try:
                content = r.get("content")
                if isinstance(content, list):
                    for it in content:
                        u = it.get("uri") or it.get("url")
                        if isinstance(u, str) and u:
                            urls.append(u)
                        t = it.get("text")
                        if isinstance(t, str):
                            import re as _re
                            for m in _re.finditer(r"https?://\S+\.(?:png|jpg|jpeg|webp|gif)", t, _re.I):
                                urls.append(m.group(0))
                # Supabase execute_sql rows
                data = (r.get("response", {}) or {}).get("data", {})
                rows = data.get("rows")
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict):
                            u = row.get("imageUrl") or row.get("image_url") or row.get("imageurl")
                            if isinstance(u, str) and u:
                                urls.append(u)
            except Exception:
                pass
            # Dedup preserve order
            dedup: List[str] = []
            for u in urls:
                if u not in dedup:
                    dedup.append(u)
            return dedup

        need_fallback = False
        try:
            if not isinstance(res, dict):
                need_fallback = True
            else:
                urls = _extract_urls_from_result(res)
                # If no valid-looking image URLs, fallback
                need_fallback = len(urls) == 0
        except Exception:
            need_fallback = True

        if need_fallback:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = (
                os.getenv("SUPABASE_KEY")
                or os.getenv("SUPABASE_ANON_KEY")
                or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            )
            if not supabase_url or not supabase_key:
                raise SystemExit("Supabase REST fallback requires SUPABASE_URL and a key (SUPABASE_ANON_KEY or SERVICE_ROLE)")
            # Fetch latest N as a reliable default
            images = _fetch_latest_image_urls(supabase_url=supabase_url, supabase_key=supabase_key, limit=n)
            # Post to Discord via MCP
            caption = cfg.get("discord_caption") or ""
            _post_images_to_discord(
                urls=images, channel=args.channel, caption=caption, media_cfg=cfg, mcp_cfg_path=mcp_cfg_path, filename=filename
            )
            res = {"content": [{"type": "image", "uri": u} for u in images]}
        else:
            # Top-off to exactly N images when the agent produced fewer
            try:
                supabase_url = os.getenv("SUPABASE_URL")
                supabase_key = (
                    os.getenv("SUPABASE_KEY")
                    or os.getenv("SUPABASE_ANON_KEY")
                    or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
                )
                urls = _extract_urls_from_result(res)
                urls = list(dict.fromkeys(urls))  # dedupe
                need_more = max(0, n - len(urls))
                if need_more > 0 and supabase_url and supabase_key:
                    extra = _fetch_latest_image_urls(
                        supabase_url=supabase_url, supabase_key=supabase_key, limit=need_more, exclude=set(urls)
                    )
                    if extra:
                        # Post only the additional images (agent already posted its own)
                        caption = cfg.get("discord_caption") or ""
                        _post_images_to_discord(
                            urls=extra,
                            channel=args.channel,
                            caption=caption,
                            media_cfg=cfg,
                            mcp_cfg_path=mcp_cfg_path,
                            filename=filename,
                        )
                        # Extend the result list for logging/completeness
                        if isinstance(res, dict):
                            res.setdefault("content", [])
                            if isinstance(res["content"], list):
                                res["content"].extend({"type": "image", "uri": u} for u in extra)
            except Exception:
                pass
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
        # Basic heuristic: fetch latest N; fast and robust
        url = f"{supabase_url}/rest/v1/eagle_images?select=title,storage_path,created_at&order=created_at.desc&limit={n}"
        r = requests.get(url, headers=headers, timeout=20)
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
        # Post to Discord via MCP (robust)
        caption = cfg.get("discord_caption") or ""
        _post_images_to_discord(
            urls=images, channel=args.channel, caption=caption, media_cfg=cfg, mcp_cfg_path=mcp_cfg_path, filename=filename
        )
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
