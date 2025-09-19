#!/usr/bin/env python3
"""
Search dreams in Supabase and print matching rows.

Environment:
  - SUPABASE_URL: e.g. https://<project>.supabase.co
  - SUPABASE_ANON_KEY (preferred for read) or SUPABASE_SERVICE_ROLE_KEY

Usage examples:
  # Human-readable list
  python scripts/search_dreams.py --query "rollercoaster" --limit 200

  # JSONL output (id, date, content)
  python scripts/search_dreams.py --query "kanye ship" --jsonl --limit 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Dict, Optional

import requests


def _env_keys():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        sys.exit(
            "Missing SUPABASE_URL and/or SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY)."
        )
    return url.rstrip("/"), key


def _headers(key: str):
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def search_rpc(url: str, key: str, q: str, limit: int, offset: int):
    endpoint = f"{url}/rest/v1/rpc/dreams_search"
    payload = {"q": q, "limit": limit, "offset": offset}
    r = requests.post(endpoint, headers=_headers(key), data=json.dumps(payload), timeout=20)
    if r.status_code == 404:
        raise FileNotFoundError("RPC dreams_search not found")
    r.raise_for_status()
    return r.json()


def search_ilike(
    url: str,
    key: str,
    q: str,
    limit: int,
    offset: int,
    source: Optional[str],
):
    endpoint = f"{url}/rest/v1/dreams"
    terms = [t for t in (q or "").split() if t]
    if not terms:
        return []
    ors = ",".join([f"content.ilike.*{t}*" for t in terms])
    params = {
        "select": "id,content,date",
        "or": f"({ors})",
        "order": "date.desc",
        "limit": str(limit),
        "offset": str(offset),
    }
    if source and source != "all":
        params["source"] = f"eq.{source}"
    r = requests.get(endpoint, headers=_headers(key), params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def normalize_row(r: Dict) -> Dict:
    return {
        "id": r.get("id") or r.get("dream_id") or r.get("uuid"),
        "date": r.get("date") or r.get("created_at") or r.get("dream_date"),
        "content": (r.get("content") or "").strip(),
    }


def main():
    ap = argparse.ArgumentParser(description="Search dreams in Supabase and print matching rows")
    ap.add_argument("--query", "-q", required=True, help="Search query (RPC if available, else OR ilike on tokens)")
    ap.add_argument("--limit", type=int, default=200, help="Fetch limit (default: 200)")
    ap.add_argument("--offset", type=int, default=0, help="Fetch offset (default: 0)")
    ap.add_argument("--source", choices=["mine", "rsos", "all"], default="mine", help="Source filter (default: mine)")
    ap.add_argument("--jsonl", action="store_true", help="Output JSONL rows (id,date,content)")
    ap.add_argument(
        "--ids-only",
        action="store_true",
        help="Only print matching dream IDs (one per line)",
    )
    args = ap.parse_args()

    if args.ids_only and args.jsonl:
        ap.error("--ids-only cannot be combined with --jsonl")

    url, key = _env_keys()

    try:
        # If a specific source is requested, prefer ilike with a server-side source filter
        if args.source and args.source != "all":
            rows = search_ilike(url, key, args.query, args.limit, args.offset, args.source)
        else:
            try:
                rows = search_rpc(url, key, args.query, args.limit, args.offset)
            except Exception:
                rows = search_ilike(url, key, args.query, args.limit, args.offset, None)
    except requests.HTTPError as e:
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text if e.response is not None else str(e)
        sys.exit(f"Supabase request failed: {detail}")

    out: List[Dict] = [normalize_row(r) for r in rows if (r.get("content") or "").strip()]

    if args.jsonl:
        for r in out:
            print(json.dumps(r, ensure_ascii=False))
        return

    if args.ids_only:
        for r in out:
            print(r["id"])
        return

    print(f"Found {len(out)} rows for query: '{args.query}' (source={args.source})\n")
    for r in out:
        content = " ".join(r["content"].split())
        snippet = content if len(content) <= 180 else content[:177] + "..."
        print(f"- id={r['id']} date={r['date']}\n  {snippet}\n")


if __name__ == "__main__":
    main()
