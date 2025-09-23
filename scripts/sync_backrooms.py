#!/usr/bin/env python3
"""
Sync Backrooms runs into Supabase.

Reads one or more JSONL metadata files (default: var/backrooms_logs/; scans **/*_meta.jsonl)
and upserts rows into public.backrooms, attaching the transcript from each
log_file if present. Safe to re-run; uses on_conflict=log_file to merge.

Usage:
  python scripts/sync_backrooms.py                      # sync all *_meta.jsonl under var/backrooms_logs/
  python scripts/sync_backrooms.py --meta path/to.jsonl # custom file
  python scripts/sync_backrooms.py --meta var/backrooms_logs # scan a directory
  python scripts/sync_backrooms.py --dry-run            # preview only

Env:
  SUPABASE_URL, SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Iterable
import re

import requests

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # optional
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import BACKROOMS_LOGS_DIR


def _relativize_to_root(path: Path) -> Optional[Path]:
    """Return the path made relative to the repo root when possible."""
    try:
        return path.relative_to(ROOT)
    except Exception:
        pass
    try:
        parts = path.parts
        root_name = ROOT.name
        if root_name in parts:
            idx = parts.index(root_name)
            return Path(*parts[idx + 1 :])
    except Exception:
        return None
    return None


def _resolve_log_path(path_str: str) -> Optional[Path]:
    """Best-effort resolution of a log path originating from any machine."""
    if not path_str:
        return None
    try:
        raw = Path(path_str)
    except Exception:
        return None

    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        rel = _relativize_to_root(raw)
        if rel is not None:
            candidates.append(ROOT / rel)
    else:
        candidates.append(ROOT / raw)

    seen: set[str] = set()
    for cand in candidates:
        if not cand:
            continue
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            if cand.exists():
                return cand
        except Exception:
            continue
    return None


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


def find_meta_files(base: Path) -> List[Path]:
    """Return a list of JSONL meta files.

    - If base is a file, return [base]
    - If base is a directory, scan recursively for *_meta.jsonl
    - If base does not exist, return []
    """
    if base.is_file():
        return [base]
    if base.is_dir():
        return sorted(base.rglob("*_meta.jsonl"))
    return []


def _clean_string(s: str) -> str:
    """Remove characters not accepted by Postgres text (e.g., NUL bytes).

    - Strips '\x00' (NUL) which Postgres rejects
    - Removes other non-printable C0 control chars except tab/newline/carriage-return
    """
    if not isinstance(s, str):
        return s  # type: ignore[return-value]
    s = s.replace("\x00", "")
    return "".join(ch for ch in s if (ch >= " " or ch in "\t\n\r"))


def _sanitize_row_strings(row: Dict) -> Dict:
    for k, v in list(row.items()):
        if isinstance(v, str):
            row[k] = _clean_string(v)
    return row


def _is_tiny_log(p: Path, min_bytes: int) -> bool:
    try:
        # Size filter disabled when min_bytes <= 0
        if int(min_bytes) <= 0:
            return False
        return p.exists() and p.is_file() and p.stat().st_size < max(1, int(min_bytes))
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
        raw_lf = it.get("log_file")
        if not raw_lf:
            removed_any = True
            continue
        raw_lf_str = str(raw_lf)
        resolved = _resolve_log_path(raw_lf_str)
        if not resolved:
            removed_any = True
            continue

        rel = _relativize_to_root(resolved)
        if rel is not None:
            lf = str(rel)
            p = ROOT / rel
        else:
            lf = str(resolved)
            p = resolved

        if lf in seen:
            removed_any = True
            continue
        seen.add(lf)
        if _is_tiny_log(p, min_bytes):
            removed_any = True
            if delete_logs:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            continue
        if lf != raw_lf_str:
            removed_any = True
        it["log_file"] = lf
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


def _count_replies_from_text(txt: str) -> int:
    """Count reply sections in a Backrooms log.

    Heuristic: count actor headers like '### <Actor> ###'.
    Falls back to counting lines that start with '### ' but not '### Round'.
    """
    if not isinstance(txt, str) or not txt:
        return 0
    count = 0
    for line in txt.splitlines():
        if not line.startswith("### "):
            continue
        if line.startswith("### Round"):
            continue
        # Typical pattern: ### <Actor> ###
        count += 1
    return count


def to_backrooms_rows(items: List[Dict], *, min_replies: int = 0) -> List[Dict]:
    """Map JSONL items to backrooms table rows.

    Also ensures the `source` is populated:
      - prefer `item['source']` from metadata
      - else fetch from Supabase `dreams` by `dream_id` (cached per run)
    """
    rows: List[Dict] = []
    seen = set()

    # Lazy env and cache for dream -> source lookups
    _env: Optional[tuple[str, str]] = None
    dream_source_cache: dict[str, Optional[str]] = {}

    def _get_env() -> tuple[str, str]:
        nonlocal _env
        if _env is None:
            _env = env_keys()
        return _env

    def _fetch_source_for_dream(dream_id: str) -> Optional[str]:
        if not dream_id:
            return None
        if dream_id in dream_source_cache:
            return dream_source_cache[dream_id]
        try:
            url, key = _get_env()
            endpoint = f"{url}/rest/v1/dreams"
            params = {"select": "source", "id": f"eq.{dream_id}", "limit": "1"}
            r = requests.get(endpoint, headers=headers(key), params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            src: Optional[str] = None
            if isinstance(data, list) and data:
                src = data[0].get("source")
            dream_source_cache[dream_id] = src
            return src
        except Exception:
            dream_source_cache[dream_id] = None
            return None

    for it in items:
        lf = it.get("log_file")
        if not lf or lf in seen:
            continue
        seen.add(lf)
        # Map fields; created_at <- start (if present)
        created_at = it.get("start") or it.get("created_at")
        # Attach transcript from file, best-effort
        transcript = None
        replies_count = 0
        try:
            resolved = _resolve_log_path(str(lf))
            if resolved and resolved.exists():
                raw = resolved.read_text(encoding="utf-8", errors="replace")
                replies_count = _count_replies_from_text(raw)
                if min_replies and replies_count < int(min_replies):
                    continue
                transcript = _clean_string(raw)
        except Exception:
            transcript = None

        dream_id = it.get("dream_id") or it.get("id")
        # Resolve source with fallback to dreams table
        src = it.get("source")
        if not src and dream_id:
            src = _fetch_source_for_dream(str(dream_id))

        row = {
            "dream_index": it.get("dream_index"),
            "dream_id": dream_id,
            "prompt": it.get("prompt"),
            "models": it.get("models"),
            "template": it.get("template"),
            "max_turns": it.get("max_turns"),
            "created_at": created_at,
            "duration_sec": it.get("duration_sec"),
            "log_file": lf,
            "exit_reason": it.get("exit_reason"),
            "transcript": transcript,
            "replies": replies_count or None,
            # Always include 'source' key (None when unknown) to ensure
            # all objects in a batch share identical keys for PostgREST.
            "source": src if src is not None else None,
        }
        rows.append(_sanitize_row_strings(row))
    return rows


def _normalize_rows(rows: List[Dict], drop_keys: Optional[set[str]] = None) -> List[Dict]:
    """Ensure all objects share identical keys (fill missing with None).

    Optionally drop specific keys across the whole batch.
    """
    drop_keys = drop_keys or set()
    # Union of keys across rows
    keys: set[str] = set()
    for r in rows:
        keys.update(r.keys())
    keys -= drop_keys
    norm: List[Dict] = []
    for r in rows:
        norm.append({k: r.get(k, None) for k in keys})
    return norm


def upsert_rows(url: str, key: str, rows: List[Dict], dry_run: bool = False) -> int:
    if dry_run:
        print(f"[dry-run] Would upsert {len(rows)} rows")
        return 0
    endpoint = f"{url}/rest/v1/backrooms?on_conflict=log_file"
    total = 0
    def _extract_unknown_cols(msg: Optional[str]) -> List[str]:
        if not msg:
            return []
        try:
            s = msg.lower()
        except Exception:
            s = str(msg)
        # Common PostgREST error phrasing examples:
        # - Could not find the 'replies' column of 'backrooms' in the schema cache
        # - column "source" of relation "backrooms" does not exist
        import re as _re
        cols: set[str] = set()
        for pat in [
            r"the '([a-zA-Z0-9_]+)' column",
            r"column\s+\"([a-zA-Z0-9_]+)\"",
            r"column\s+'([a-zA-Z0-9_]+)'",
        ]:
            m = _re.search(pat, msg)
            if m:
                cols.add(m.group(1))
        return list(cols)

    for batch in chunked(rows, 50):
        # Normalize keys to avoid PGRST102 (all object keys must match)
        norm_batch = _normalize_rows(batch)
        drop: set[str] = set()
        attempt = 0
        while True:
            attempt += 1
            send_batch = _normalize_rows(norm_batch, drop_keys=drop) if drop else norm_batch
            r = requests.post(endpoint, headers=headers(key), json=send_batch, timeout=60)
            if r.ok:
                try:
                    payload = r.json()
                    total += len(payload) if isinstance(payload, list) else len(send_batch)
                except Exception:
                    total += len(send_batch)
                break

            # If the failure looks like an unknown column (e.g., 'replies' or 'source'), iteratively drop and retry
            detail_text: Optional[str]
            try:
                detail = r.json()
                detail_text = json.dumps(detail)
            except Exception:
                detail_text = r.text

            unknown_cols = set(_extract_unknown_cols(detail_text))
            if detail_text and ("schema cache" in (detail_text or "").lower() or "does not exist" in (detail_text or "").lower()):
                for candidate in ("replies", "source"):
                    if candidate in (detail_text or "").lower():
                        unknown_cols.add(candidate)

            # If we detected unknown columns, add to drop set and retry (limit attempts to avoid infinite loops)
            if unknown_cols and attempt < 5:
                drop |= unknown_cols
                continue

            # If we reach here, either no fallback or fallback failed
            raise SystemExit(f"Upsert failed: HTTP {r.status_code}: {detail_text}")
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
    ap.add_argument(
        "--meta",
        default=str(BACKROOMS_LOGS_DIR),
        help="Path to JSONL meta file or a directory to scan (default: var/backrooms_logs)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing to Supabase")
    ap.add_argument("--no-clean", action="store_true", help="Skip tiny/missing log cleanup before syncing")
    ap.add_argument("--min-bytes", type=int, default=0, help="Minimum size threshold for logs when cleaning; 0 disables size filter (default: 0)")
    ap.add_argument("--min-replies", type=int, default=1, help="Skip logs with fewer than this many replies (default: 1)")
    args = ap.parse_args()

    url, key = env_keys()
    target = Path(args.meta)
    files = find_meta_files(target)
    if not files:
        raise SystemExit(f"No meta files found at: {target}")

    total_synced = 0
    for meta_path in files:
        # Clean tiny/missing logs and de-duplicate JSONL by default
        if not args.no_clean:
            clean_meta_inplace(meta_path, min_bytes=int(args.min_bytes), delete_logs=True)
        items = read_jsonl(meta_path)
        if not items:
            continue
        rows = to_backrooms_rows(items, min_replies=int(args.min_replies))
        if not rows:
            continue
        count = upsert_rows(url, key, rows, dry_run=args.dry_run)
        print(f"Synced {count} rows from {meta_path}")
        total_synced += count
    print(f"Done. Total rows synced: {total_synced}")


if __name__ == "__main__":
    main()
