#!/usr/bin/env python3
"""
Seed templates/dreamsim3/vars.json (used by initiator.history.template.md) with a dream from Supabase.

Environment:
  - SUPABASE_URL: e.g. https://<project>.supabase.co
  - SUPABASE_ANON_KEY (preferred for read) or SUPABASE_SERVICE_ROLE_KEY

Usage examples:
  python scripts/seed_dreamsim3.py                    # pick random from recent
  python scripts/seed_dreamsim3.py --query rollercoaster
  python scripts/seed_dreamsim3.py --limit 100 --query "kanye ship"

Notes:
- Template now references initiator.history.template.md and backrooms injects variables from vars.json.
- DREAM_TEXT is inserted via Python .format; curly braces in the dream are auto-escaped.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import requests


TEMPLATE_DIR = Path("templates/dreamsim3")
VARS_FILE = TEMPLATE_DIR / "vars.json"


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


def fetch_recent(url: str, key: str, limit: int = 50, source: str | None = None):
    """Fetch recent dreams directly from the dreams table via PostgREST.

    Uses a minimal column set that matches our CSV and dataset expectations.
    """
    endpoint = f"{url}/rest/v1/dreams"
    params = {
        "select": "id,content,date",
        "order": "date.desc",
        "limit": str(limit),
    }
    if source and source != "all":
        params["source"] = f"eq.{source}"
    r = requests.get(endpoint, headers=_headers(key), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def search_dreams(url: str, key: str, q: str, limit: int = 50, source: str | None = None):
    """Search dreams by content.

    - If a specific source is requested, use PostgREST ilike with source filter.
    - Otherwise, prefer RPC `dreams_search` (if available) for better fuzzy ranking.
    """
    terms = [t for t in (q or "").split() if t]
    if not terms:
        return []

    # If source constrained, go straight to PostgREST to guarantee proper filtering
    if source and source != "all":
        endpoint = f"{url}/rest/v1/dreams"
        ors = ",".join([f"content.ilike.*{t}*" for t in terms])
        params = {
            "select": "id,content,date",
            "or": f"({ors})",
            "order": "date.desc",
            "limit": str(limit),
            "source": f"eq.{source}",
        }
        r = requests.get(endpoint, headers=_headers(key), params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    # Try RPC first (unconstrained source)
    try:
        endpoint = f"{url}/rest/v1/rpc/dreams_search"
        payload = {"q": q, "limit": limit, "offset": 0}
        r = requests.post(endpoint, headers=_headers(key), data=json.dumps(payload), timeout=20)
        if r.status_code == 404:
            raise FileNotFoundError("RPC dreams_search not found")
        r.raise_for_status()
        return r.json()
    except Exception:
        # Fallback: PostgREST ilike OR across whitespace-separated tokens
        endpoint = f"{url}/rest/v1/dreams"
        ors = ",".join([f"content.ilike.*{t}*" for t in terms])
        params = {
            "select": "id,content,date",
            "or": f"({ors})",
            "order": "date.desc",
            "limit": str(limit),
        }
        r = requests.get(endpoint, headers=_headers(key), params=params, timeout=20)
        r.raise_for_status()
        return r.json()


def normalize_dream_text(row: dict) -> str:
    text = (row.get("content") or "").strip()
    # Collapse whitespace/newlines to keep CLI command on one line
    text = " ".join(text.split())
    return text


def write_vars(dream_text: str):
    # Write template variables consumed by backrooms (vars.json in template folder)
    VARS_FILE.write_text(json.dumps({"DREAM_TEXT": dream_text}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Seed dreamsim3 initiator with a dream from Supabase")
    ap.add_argument("--query", "-q", help="Fuzzy search query (uses RPC dreams_search)")
    ap.add_argument("--limit", type=int, default=50, help="Limit for recent/search fetch (default: 50)")
    ap.add_argument("--source", choices=["mine", "rsos", "all"], default="mine", help="Source filter for dreams (default: mine)")
    ap.add_argument("--print", dest="do_print", action="store_true", help="Print chosen dream text to stdout")
    args = ap.parse_args()

    url, key = _env_keys()

    try:
        if args.query:
            rows = search_dreams(url, key, args.query, args.limit, args.source)
        else:
            rows = fetch_recent(url, key, args.limit, args.source)
    except requests.HTTPError as e:
        # Surface useful server error details if present
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text if e.response is not None else str(e)
        sys.exit(f"Supabase request failed: {detail}")

    if not rows:
        sys.exit("No dreams found for the given parameters.")

    row = random.choice(rows)
    dream_text = normalize_dream_text(row)

    write_vars(dream_text)

    if args.do_print:
        print(dream_text)
    print(f"Wrote {VARS_FILE} (seeded from Supabase)")


if __name__ == "__main__":
    main()
