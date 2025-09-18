#!/usr/bin/env python3
"""
Index Backrooms logs for a given template into a metadata JSONL.

This scans a logs directory (default: var/backrooms_logs/<template>) for *.txt
transcripts produced by backrooms.py and writes/updates a JSONL file with
minimal metadata used by scripts/sync_backrooms.py.

Fields per row:
- models: [model1, model2] parsed from filename prefix
- template: template name parsed/validated from filename
- start: ISO8601 timestamp parsed from filename suffix (UTC assumed if 'Z' provided; otherwise local naive)
- max_turns: parsed from tail when present (best-effort)
- duration_sec: None (not derivable reliably post hoc)
- log_file: absolute or relative path to the log file
- exit_reason: inferred from tail (early_stop | max_turns | context_budget | unknown)

By default, it appends only new logs not already present in the JSONL based
on exact 'log_file' matches. Use --reindex to rebuild from scratch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paths import BACKROOMS_LOGS_DIR


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
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


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


FILENAME_RE = re.compile(
    r"^(?P<m1>[^_]+)_(?P<m2>[^_]+)_(?P<tmpl>[^_]+)_(?P<ts>\d{8}_\d{6})\.txt$"
)


def parse_filename(p: Path) -> Optional[Tuple[str, str, str, str]]:
    m = FILENAME_RE.match(p.name)
    if not m:
        return None
    return m.group("m1"), m.group("m2"), m.group("tmpl"), m.group("ts")


def ts_to_iso(ts: str) -> str:
    # ts like 20250907_224538 -> assume local time; store as naive ISO
    try:
        dtobj = dt.datetime.strptime(ts, "%Y%m%d_%H%M%S")
        return dtobj.isoformat()
    except Exception:
        return ts


def determine_exit_reason_from_file(p: Path) -> str:
    try:
        tail = p.read_text(encoding="utf-8")[-4000:]
    except Exception:
        return "unknown"
    if "has ended the conversation with ^C^C" in tail:
        return "early_stop"
    if "Reached maximum number of turns" in tail:
        return "max_turns"
    if "Context budget limit reached" in tail:
        return "context_budget"
    return "unknown"


def parse_max_turns_from_tail(p: Path) -> Optional[int]:
    try:
        tail = p.read_text(encoding="utf-8")[-4000:]
    except Exception:
        return None
    m = re.search(r"Reached maximum number of turns \((\d+)\)", tail)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def build_meta_for_file(p: Path, *, default_prompt: Optional[str] = "AI GENERATED") -> Optional[Dict]:
    parsed = parse_filename(p)
    if not parsed:
        return None
    m1, m2, tmpl, ts = parsed
    start_iso = ts_to_iso(ts)
    exit_reason = determine_exit_reason_from_file(p)
    max_turns = parse_max_turns_from_tail(p)
    return {
        "dream_index": None,
        "dream_id": None,
        "date": None,
        "prompt": default_prompt,
        "models": [m1, m2],
        "template": tmpl,
        "max_turns": max_turns,
        "start": start_iso,
        "end": None,
        "duration_sec": None,
        "log_file": str(p),
        "exit_reason": exit_reason,
        "returncode": None,
        "stderr_tail": None,
    }


def index_logs(logs_dir: Path, template: str, existing: Set[str], *, default_prompt: Optional[str]) -> List[Dict]:
    out: List[Dict] = []
    for p in sorted(logs_dir.glob("*.txt")):
        if str(p) in existing:
            continue
        meta = build_meta_for_file(p, default_prompt=default_prompt)
        if not meta:
            continue
        if meta.get("template") != template:
            continue
        out.append(meta)
    return out


def main():
    ap = argparse.ArgumentParser(description="Index var/backrooms_logs/<template> into <template>_meta.jsonl")
    ap.add_argument("--template", required=True, help="Template name to index (e.g., dreamsim4)")
    ap.add_argument("--logs-dir", default="", help="Logs directory (default: var/backrooms_logs/<template>)")
    ap.add_argument("--out", default="", help="Output JSONL (default: var/backrooms_logs/<template>/<template>_meta.jsonl)")
    ap.add_argument("--reindex", action="store_true", help="Rebuild JSONL from scratch (ignore existing entries)")
    ap.add_argument("--prompt", default="AI GENERATED", help="Default prompt to store for each log (default: 'AI GENERATED')")
    args = ap.parse_args()

    template = args.template
    logs_dir = Path(args.logs_dir) if args.logs_dir else BACKROOMS_LOGS_DIR / template
    out_path = Path(args.out) if args.out else BACKROOMS_LOGS_DIR / template / f"{template}_meta.jsonl"

    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows: List[Dict] = []
    existing_set: Set[str] = set()
    if not args.reindex and out_path.exists():
        existing_rows = read_jsonl(out_path)
        existing_set = {str(r.get("log_file")) for r in existing_rows if r.get("log_file")}

    new_rows = index_logs(logs_dir, template, existing_set, default_prompt=args.prompt)

    if args.reindex:
        write_jsonl(out_path, new_rows)
        print(f"Indexed {len(new_rows)} logs into {out_path} (reindex)")
    else:
        all_rows = existing_rows + new_rows
        write_jsonl(out_path, all_rows)
        print(f"Indexed {len(new_rows)} new logs (total {len(all_rows)}) into {out_path}")


if __name__ == "__main__":
    main()
