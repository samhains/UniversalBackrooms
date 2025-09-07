#!/usr/bin/env python3
"""
Sync Backrooms runs into Supabase.

Reads a JSONL metadata file (default: BackroomsLogs/dreamsim3/dreamsim3_meta.jsonl)
and upserts rows into public.backrooms, attaching the transcript from each
log_file if present. Safe to re-run; uses on_conflict=log_file to merge.

Usage:
  python scripts/sync_backrooms.py                      # sync default JSONL
  python scripts/sync_backrooms.py --meta path/to.jsonl # custom file
  python scripts/sync_backrooms.py --dry-run            # preview only

Env:
  SUPABASE_URL, SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import requests

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # optional
    load_dotenv = None


def _load_env_from_file(path: Path) -> None:
    """Lightweight .env loader that doesn't require python-dotenv.

    Only parses simple KEY=VALUE lines; ignores blanks and comments.
    Does not overwrite existing environment variables.
    """
    try:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "=" not in s:
                continue
            key, val = s.split("=", 1)
            key = key.strip()
            # Strip optional surrounding quotes
            v = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = v
    except Exception:
        pass


def env_keys() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL and/or SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY).")
    return url.rstrip('/'), key


def headers(key: str) -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }


def read_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            out.append(obj)
    return out


def _is_tiny_log(p: Path, min_bytes: int) -> bool:
    try:
        return p.exists() and p.is_file() and p.stat().st_size < max(1, min_bytes)
    except Exception:
        return True


def clean_meta_inplace(meta_path: Path, *, min_bytes: int = 128, delete_logs: bool = True) -> None:
    """Remove entries with missing or tiny log files; de-duplicate by log_file.

    Rewrites the JSONL in-place and writes a .bak backup when changes occur.
    """
    items = read_jsonl(meta_path)
    keep: List[Dict] = []
    seen: set[str] = set()
    removed_any = False
    for it in items:
        lf = it.get("log_file")
        if not lf:
            removed_any = True
            continue
        p = Path(lf)
        if not p.exists() or _is_tiny_log(p, min_bytes):
            removed_any = True
            if delete_logs and p.exists():
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            continue
        if lf in seen:
            removed_any = True
            continue
        seen.add(lf)
        keep.append(it)
    if removed_any:
        backup = meta_path.with_suffix(meta_path.suffix + ".bak")
        try:
            backup.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
        with meta_path.open("w", encoding="utf-8") as f:
            for row in keep:
                f.write(json.dumps(row) + "\n")


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def to_backrooms_rows(items: List[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    seen = set()
    for it in items:
        lf = it.get("log_file")
        if not lf or lf in seen:
            continue
        seen.add(lf)
        # Map fields; created_at <- start (if present)
        created_at = it.get("start") or it.get("created_at")
        # Attach transcript from file, best-effort
        transcript = None
        try:
            p = Path(lf)
            if p.exists():
                transcript = p.read_text(encoding="utf-8")
        except Exception:
            transcript = None
        rows.append(
            {
                "dream_index": it.get("dream_index"),
                "dream_id": it.get("dream_id") or it.get("id"),
                "prompt": it.get("prompt"),
                "models": it.get("models"),
                "template": it.get("template"),
                "max_turns": it.get("max_turns"),
                "created_at": created_at,
                "duration_sec": it.get("duration_sec"),
                "log_file": lf,
                "exit_reason": it.get("exit_reason"),
                "transcript": transcript,
            }
        )
    return rows


def upsert_rows(url: str, key: str, rows: List[Dict], dry_run: bool = False) -> int:
    if dry_run:
        print(f"[dry-run] Would upsert {len(rows)} rows")
        return 0
    endpoint = f"{url}/rest/v1/backrooms?on_conflict=log_file"
    total = 0
    for batch in chunked(rows, 50):
        r = requests.post(endpoint, headers=headers(key), json=batch, timeout=60)
        if not r.ok:
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            raise SystemExit(f"Upsert failed: HTTP {r.status_code}: {detail}")
        try:
            payload = r.json()
            total += len(payload) if isinstance(payload, list) else 0
        except Exception:
            # Prefer: return=representation ensures list; but be resilient
            total += len(batch)
    return total


def main():
    # Best-effort load .env from repository root
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if load_dotenv is not None:
        try:
            load_dotenv(env_path)
        except Exception:
            pass
    # Fallback tiny parser if python-dotenv is unavailable
    _load_env_from_file(env_path)

    ap = argparse.ArgumentParser(description="Sync Backrooms runs into Supabase")
    ap.add_argument("--meta", default="BackroomsLogs/dreamsim3/dreamsim3_meta.jsonl", help="Path to JSONL metadata file")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing to Supabase")
    ap.add_argument("--no-clean", action="store_true", help="Skip tiny/missing log cleanup before syncing")
    ap.add_argument("--min-bytes", type=int, default=128, help="Minimum size threshold for logs when cleaning (default: 128)")
    args = ap.parse_args()

    url, key = env_keys()
    meta_path = Path(args.meta)
    if not meta_path.exists():
        raise SystemExit(f"Metadata file not found: {meta_path}")

    # Clean tiny/missing logs and de-duplicate JSONL by default
    if not args.no_clean:
        clean_meta_inplace(meta_path, min_bytes=int(args.min_bytes), delete_logs=True)

    items = read_jsonl(meta_path)
    rows = to_backrooms_rows(items)
    count = upsert_rows(url, key, rows, dry_run=args.dry_run)
    print(f"Synced {count} rows from {meta_path}")


if __name__ == "__main__":
    main()
