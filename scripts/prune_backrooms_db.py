#!/usr/bin/env python3
"""
Prune Supabase backrooms rows to match cleaned metadata.

Features:
- Loads .env automatically (python-dotenv if available, otherwise a tiny parser).
- Reads a JSONL metadata file and builds the allowed set of log_file paths.
- Can delete DB rows for a given template where log_file is NOT in metadata.
- Can delete DB rows with tiny transcripts (length < --min-bytes) for a template.

Usage examples:
  # Dry-run: show counts only
  python scripts/prune_backrooms_db.py --template dreamsim3 --meta BackroomsLogs/dreamsim3/dreamsim3_meta.jsonl

  # Delete rows not in metadata (scoped to template)
  python scripts/prune_backrooms_db.py --template dreamsim3 --meta BackroomsLogs/dreamsim3/dreamsim3_meta.jsonl --delete-not-in-meta

  # Delete rows with tiny transcripts (<128 bytes)
  python scripts/prune_backrooms_db.py --template dreamsim3 --delete-tiny --min-bytes 128
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None


def _load_env_from_file(path: Path) -> None:
    try:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


def env_keys() -> Tuple[str, str]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL and a key (prefer SERVICE_ROLE). Update .env.")
    return url.rstrip("/"), key


def headers(key: str) -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Prefer": "return=representation",
    }


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path:
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except Exception:
                continue
    return rows


def fetch_backrooms(url: str, key: str, template: str) -> List[Dict]:
    # Fetch all rows for template with minimal fields
    endpoint = f"{url}/rest/v1/backrooms"
    params = {
        "select": "log_file,template,transcript",
        "template": f"eq.{template}",
        "limit": "10000",
    }
    r = requests.get(endpoint, headers=headers(key), params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise SystemExit("Unexpected response from Supabase backrooms fetch")
    return data


def chunk(seq: List[str], n: int) -> List[List[str]]:
    return [seq[i : i + n] for i in range(0, len(seq), n)]


def delete_by_log_files(url: str, key: str, template: str, logs: List[str]) -> int:
    total = 0
    for batch in chunk(logs, 200):
        # PostgREST in() list must be comma-separated, values quoted if they contain commas.
        # We'll URL-encode via requests; just build the string.
        in_list = ",".join([f"\"{lf}\"" for lf in batch])
        endpoint = f"{url}/rest/v1/backrooms"
        params = {
            "template": f"eq.{template}",
            "log_file": f"in.({in_list})",
        }
        r = requests.delete(endpoint, headers=headers(key), params=params, timeout=60)
        if not r.ok:
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            raise SystemExit(f"Delete failed: HTTP {r.status_code}: {detail}")
        try:
            payload = r.json()
            total += len(payload) if isinstance(payload, list) else 0
        except Exception:
            total += len(batch)
    return total


def main():
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    if load_dotenv is not None:
        try:
            load_dotenv(env_path)
        except Exception:
            pass
    _load_env_from_file(env_path)

    ap = argparse.ArgumentParser(description="Prune backrooms rows to match metadata and size constraints")
    ap.add_argument("--template", required=True, help="Template name to scope deletes (e.g., dreamsim3)")
    ap.add_argument("--meta", default="", help="Path to cleaned JSONL metadata to keep (optional)")
    ap.add_argument("--delete-not-in-meta", action="store_true", help="Delete rows where log_file not in the JSONL set")
    ap.add_argument("--delete-tiny", action="store_true", help="Delete rows with short transcripts (< min-bytes)")
    ap.add_argument("--min-bytes", type=int, default=128, help="Minimum transcript length in bytes when --delete-tiny (default: 128)")
    ap.add_argument("--dry-run", action="store_true", help="Preview counts only; do not delete")
    args = ap.parse_args()

    url, key = env_keys()
    template = args.template

    # Fetch current DB rows
    rows = fetch_backrooms(url, key, template)
    db_total = len(rows)

    to_delete: Set[str] = set()

    if args.delete_tiny:
        tiny = [r.get("log_file") for r in rows if isinstance(r.get("transcript"), str) and len(r.get("transcript") or "") < int(args.min_bytes)]
        to_delete.update(lf for lf in tiny if lf)

    if args.delete_not_in_meta:
        if not args.meta:
            raise SystemExit("--delete-not-in-meta requires --meta pointing to cleaned JSONL")
        meta_items = read_jsonl(Path(args.meta))
        keep_set = {str(it.get("log_file")) for it in meta_items if it.get("log_file")}
        db_logs = {str(r.get("log_file")) for r in rows if r.get("log_file")}
        not_in_meta = sorted(db_logs - keep_set)
        to_delete.update(not_in_meta)

    print(f"DB rows for template '{template}': {db_total}")
    print(f"Would delete: {len(to_delete)}")

    if args.dry_run or not to_delete:
        return

    deleted = delete_by_log_files(url, key, template, sorted(to_delete))
    print(f"Deleted rows: {deleted}")


if __name__ == "__main__":
    main()

