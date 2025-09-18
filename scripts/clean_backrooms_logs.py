#!/usr/bin/env python3
"""
Clean var/backrooms_logs and metadata JSONL by removing tiny/empty logs.

- Scans JSONL entries (default: var/backrooms_logs/dreamsim3/dreamsim3_meta.jsonl).
- Flags entries whose `log_file` is missing or smaller than --min-bytes.
- Optionally deletes those tiny/missing logs from disk and rewrites JSONL
  without the bad entries.

Usage examples:
  # Preview what would be removed (no writes)
  python scripts/clean_backrooms_logs.py --meta var/backrooms_logs/dreamsim3/dreamsim3_meta.jsonl

  # Use a stricter size threshold
  python scripts/clean_backrooms_logs.py --min-bytes 128

  # Delete tiny logs on disk and write cleaned JSONL alongside as .cleaned.jsonl
  python scripts/clean_backrooms_logs.py --delete-logs --write-cleaned

  # In-place rewrite of JSONL (backs up as .bak) and delete logs
  python scripts/clean_backrooms_logs.py --delete-logs --inplace
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_META = ROOT / "var" / "backrooms_logs" / "dreamsim3" / "dreamsim3_meta.jsonl"


@dataclass
class Entry:
    raw: Dict
    index: int


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


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def is_tiny_log(p: Path, min_bytes: int) -> bool:
    try:
        return p.exists() and p.is_file() and p.stat().st_size < max(1, min_bytes)
    except Exception:
        return True


def clean(
    meta_path: Path,
    min_bytes: int,
    delete_logs: bool,
    inplace: bool,
    write_cleaned: bool,
) -> Tuple[int, int, int, int]:
    items = read_jsonl(meta_path)
    entries: List[Entry] = [Entry(raw=it, index=i) for i, it in enumerate(items)]

    removed_missing = 0
    removed_tiny = 0

    keep: List[Dict] = []
    seen_logs: set[str] = set()
    duplicates_removed = 0

    for e in entries:
        lf = e.raw.get("log_file")
        if not lf:
            # No file to attach — drop
            removed_missing += 1
            continue
        p = Path(lf)
        if not p.exists():
            removed_missing += 1
            continue
        if is_tiny_log(p, min_bytes):
            removed_tiny += 1
            if delete_logs:
                try:
                    p.unlink(missing_ok=True)
                except Exception:
                    pass
            continue
        # De-dupe identical log_file references; keep the last occurrence
        if lf in seen_logs:
            duplicates_removed += 1
            continue
        seen_logs.add(lf)
        keep.append(e.raw)

    # Write cleaned output
    if inplace:
        backup = meta_path.with_suffix(meta_path.suffix + ".bak")
        try:
            backup.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
        write_jsonl(meta_path, keep)
    elif write_cleaned:
        out = meta_path.with_suffix(".cleaned.jsonl")
        write_jsonl(out, keep)

    return len(entries), len(keep), removed_missing, removed_tiny + duplicates_removed


def main():
    ap = argparse.ArgumentParser(description="Remove tiny/empty logs and clean JSONL metadata")
    ap.add_argument(
        "--meta",
        default=str(DEFAULT_META),
        help="Path to metadata JSONL (default: var/backrooms_logs/dreamsim3/dreamsim3_meta.jsonl)",
    )
    ap.add_argument("--min-bytes", type=int, default=64, help="Minimum file size in bytes to be considered a valid log (default: 64)")
    ap.add_argument("--delete-logs", action="store_true", help="Delete tiny log files on disk")
    where = ap.add_mutually_exclusive_group()
    where.add_argument("--inplace", action="store_true", help="Rewrite the JSONL in-place (backs up as .bak)")
    where.add_argument("--write-cleaned", action="store_true", help="Write cleaned JSONL next to source (.cleaned.jsonl)")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    if not meta_path.exists():
        raise SystemExit(f"Metadata file not found: {meta_path}")

    total, kept, removed_missing, removed_small_or_dupes = clean(
        meta_path=meta_path,
        min_bytes=int(args.min_bytes),
        delete_logs=bool(args.delete_logs),
        inplace=bool(args.inplace),
        write_cleaned=bool(args.write_cleaned),
    )

    print(
        (
            f"Scanned {total} meta entries. Kept {kept}. "
            f"Removed missing={removed_missing}, tiny/dupes={removed_small_or_dupes}."
        )
    )
    if args.delete_logs:
        print("Tiny logs were deleted from disk.")
    if args.inplace:
        print(f"Rewrote JSONL in-place: {meta_path} (backup written as {meta_path}.bak)")
    elif args.write_cleaned:
        print(f"Wrote cleaned JSONL: {meta_path.with_suffix('.cleaned.jsonl')} ")
    else:
        print("No JSONL written (dry-run). Use --inplace or --write-cleaned to save.")


if __name__ == "__main__":
    main()
