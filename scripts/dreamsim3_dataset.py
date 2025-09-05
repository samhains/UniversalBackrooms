#!/usr/bin/env python3
"""
Batch runner for DreamSim3 seeded from a local CSV.

Reads `data/dreams_rows.csv` (or a provided CSV) and, for each dream text,
renders `templates/dreamsim3/initiator.history.md` from the template file,
then invokes `backrooms.py` with a configurable list of models and max turns.

For each run, writes a metadata record to a JSONL file including:
- dream text (prompt), models used, template, max turns
- start/end timestamps and duration
- log file path created by backrooms
- exit reason: "max_turns" or "early_stop" (when ^C^C encountered)

Usage examples:
  python scripts/dreamsim3_dataset.py \
      --csv data/dreams_rows.csv \
      --models gpt5,hermes,k2 \
      --max-turns 30

  # Limit number of dreams processed
  python scripts/dreamsim3_dataset.py --max-dreams 20

  # Mixed model pairs (model1 vs model2)
  python scripts/dreamsim3_dataset.py --pairs gpt5:hermes,hermes:k2 --max-turns 30

Notes:
- If --pairs is provided, it runs each pair as (model1, model2).
- Otherwise, --models is used as separate self-dialogue runs (model, model) for each model.
- The script is resilient to early termination runs triggered by "^C^C".
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import List, Optional, Tuple

from model_config import get_model_info

TEMPLATES_DIR = Path("templates/dreamsim3")
INIT_TEMPLATE = TEMPLATES_DIR / "initiator.history.template.md"
INIT_OUTPUT = TEMPLATES_DIR / "initiator.history.md"
BACKROOMS_LOGS = Path("BackroomsLogs")


def read_dreams_from_csv(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # Expect a column named 'content'
        if not reader.fieldnames or "content" not in reader.fieldnames:
            sys.exit("CSV must contain a 'content' column with dream text")
        for row in reader:
            content = (row.get("content") or "").strip()
            if content:
                rows.append(row)
    return rows


def render_initiator(dream_text: str) -> str:
    # Keep CLI arg single-line; escape embedded quotes
    safe = " ".join(dream_text.split()).replace('"', '\\"')
    if INIT_TEMPLATE.exists():
        base = INIT_TEMPLATE.read_text(encoding="utf-8")
        return base.replace("{{DREAM_TEXT}}", safe)
    # Fallback minimal initiator
    return (
        "## assistant\n"
        "simulator@{model2_company}:~/$\n\n"
        "## user\n\n"
        f"./dreamsim.exe \"{safe}\"\n"
    )


def latest_log_for(models: List[str], template: str) -> Optional[Path]:
    pattern = f"{'_'.join(models)}_{template}_"
    candidates = sorted(
        (p for p in BACKROOMS_LOGS.glob("*.txt") if pattern in p.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def determine_exit_reason(log_file: Path) -> str:
    try:
        tail = log_file.read_text(encoding="utf-8")[-2000:]
    except Exception:
        return "unknown"
    if "has ended the conversation with ^C^C" in tail:
        return "early_stop"
    if "Reached maximum number of turns" in tail:
        return "max_turns"
    return "unknown"


def run_one(models_pair: List[str], max_turns: int, template: str) -> subprocess.CompletedProcess:
    # backrooms requires as many models as agents in template (2)
    cmd = [
        sys.executable,
        "backrooms.py",
        "--lm",
        *models_pair,
        "--template",
        template,
        "--max-turns",
        str(max_turns),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def parse_pairs(pairs_str: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not pairs_str:
        return pairs
    # Accept separators ':', '+', 'x'
    for token in [t.strip() for t in pairs_str.split(',') if t.strip()]:
        mid = None
        for sep in (':', '+', 'x'):
            if sep in token:
                mid = sep
                break
        if not mid:
            raise SystemExit(f"Invalid pair format '{token}'. Use model1:model2 (or '+', 'x').")
        a, b = [p.strip() for p in token.split(mid, 1)]
        if not a or not b:
            raise SystemExit(f"Invalid pair '{token}' — both models required.")
        pairs.append((a, b))
    return pairs


def validate_models(aliases: List[str]):
    info = get_model_info()
    unknown = [m for m in aliases if m not in info and m.lower() != 'cli']
    if unknown:
        raise SystemExit(f"Unknown model alias(es): {', '.join(unknown)}")


def main():
    ap = argparse.ArgumentParser(description="Batch runner for DreamSim3 from CSV")
    ap.add_argument("--csv", default="data/dreams_rows.csv", help="Path to CSV with a 'content' column")
    ap.add_argument("--models", default="gpt5,hermes,k2", help="Comma-separated model aliases (each runs as model,model unless --pairs is given)")
    ap.add_argument("--pairs", default="", help="Comma-separated mixed pairs, e.g. 'gpt5:hermes,hermes:k2'")
    ap.add_argument("--max-turns", type=int, default=30, help="Maximum turns per run (default: 30)")
    ap.add_argument("--max-dreams", type=int, default=0, help="Limit number of dreams processed (0 = all)")
    ap.add_argument("--template", default="dreamsim3", help="Template name (default: dreamsim3)")
    ap.add_argument("--out", default="BackroomsLogs/dreamsim3_meta.jsonl", help="Metadata JSONL output path")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    # Resolve runs from either pairs or models
    pairs = parse_pairs(args.pairs)
    if pairs:
        flat = [x for pair in pairs for x in pair]
        validate_models(flat)
    else:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        if not models:
            sys.exit("Provide --pairs or at least one model via --models")
        validate_models(models)
        pairs = [(m, m) for m in models]

    # Prepare metadata sink
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_dreams_from_csv(csv_path)
    if args.max_dreams > 0:
        rows = rows[: args.max_dreams]

    total_runs = len(rows) * len(pairs)
    print(f"Running {total_runs} simulations: {len(rows)} dreams x {len(pairs)} runs")

    completed = 0
    for idx, row in enumerate(rows, start=1):
        dream_text = (row.get("content") or "").strip()
        # Update initiator for this dream
        INIT_OUTPUT.write_text(render_initiator(dream_text), encoding="utf-8")

        for (m1, m2) in pairs:
            models_pair = [m1, m2]
            start = dt.datetime.utcnow().isoformat() + "Z"
            t0 = time.time()

            # Run the conversation
            proc = run_one(models_pair, args.max_turns, args.template)

            # Best-effort locate the log file created by backrooms
            log_path = latest_log_for(models_pair, args.template)
            duration = time.time() - t0
            end = dt.datetime.utcnow().isoformat() + "Z"
            exit_reason = determine_exit_reason(log_path) if log_path else "unknown"

            # Persist metadata
            meta = {
                "dream_index": idx,
                "dream_id": row.get("id"),
                "date": row.get("date"),
                "prompt": dream_text,
                "models": models_pair,
                "template": args.template,
                "max_turns": args.max_turns,
                "start": start,
                "end": end,
                "duration_sec": round(duration, 3),
                "log_file": str(log_path) if log_path else None,
                "exit_reason": exit_reason,
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-1000:] if proc.stderr else None,
            }
            with out_path.open("a", encoding="utf-8") as outf:
                outf.write(json.dumps(meta) + "\n")

            completed += 1
            pair_label = f"{m1}-{m2}"
            print(f"[{completed}/{total_runs}] dream#{idx} pair={pair_label} turns={args.max_turns} exit={exit_reason} time={duration:.1f}s")

    print(f"Done. Wrote metadata to {out_path}")


if __name__ == "__main__":
    main()
