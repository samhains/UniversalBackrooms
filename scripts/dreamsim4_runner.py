#!/usr/bin/env python3
"""
Simple batch runner for DreamSim4.

Runs backrooms.py with the dreamsim4 template for a sequence of
model pairs. You can:

- Cycle through a list of model aliases, running self-dialogue (model, model)
  for N runs (default: 30), or
- Provide explicit pairs like "opus4:sonnet4,k2:gpt5" to run in order.

For each run, the script:
- Invokes backrooms.py with the specified models and template
-, by default, runs for 30 turns
- Writes a metadata JSONL entry to BackroomsLogs/dreamsim4/dreamsim4_meta.jsonl
  including timing and the discovered log file path

Examples:
  # 30 runs, cycling through models (self-dialogue):
  python scripts/dreamsim4_runner.py \
    --models opus4,sonnet4,k2,gpt5,v31 \
    --runs 30 --max-turns 30

  # Explicit pairs (runs each pair once):
  python scripts/dreamsim4_runner.py \
    --pairs opus4:sonnet4,k2:gpt5 \
    --max-turns 30

  # Cycle through models for 10 runs, no streaming to console:
  python scripts/dreamsim4_runner.py \
    --models opus4,sonnet4,k2,gpt5,v31 \
    --runs 10 --no-stream
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import os

# Ensure repository root is importable when running as a script from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_config import get_model_info

BACKROOMS_LOGS = Path("BackroomsLogs")


def validate_models(aliases: List[str]):
    info = get_model_info()
    unknown = [m for m in aliases if m not in info and m.lower() != "cli"]
    if unknown:
        raise SystemExit(f"Unknown model alias(es): {', '.join(unknown)}")


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


def latest_log_for(models: List[str], template: str) -> Optional[Path]:
    pattern = f"{'_'.join(models)}_{template}_"
    tmpl_dir = BACKROOMS_LOGS / template
    search_spaces = []
    if tmpl_dir.exists():
        search_spaces.append(tmpl_dir.rglob("*.txt"))
    search_spaces.append(BACKROOMS_LOGS.glob("*.txt"))  # legacy fallback
    candidates = []
    for it in search_spaces:
        for p in it:
            if pattern in p.name:
                candidates.append(p)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
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
    if "Context budget limit reached" in tail:
        return "context_budget"
    return "unknown"


def run_one(
    models_pair: List[str],
    max_turns: int,
    template: str,
    max_context_frac: float,
    context_window: int,
    stream: bool = True,
) -> subprocess.CompletedProcess:
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
    if max_context_frac and max_context_frac > 0:
        cmd += ["--max-context-frac", str(max_context_frac)]
        if context_window and context_window > 0:
            cmd += ["--context-window", str(context_window)]
    if stream:
        return subprocess.run(cmd)
    else:
        return subprocess.run(cmd, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser(description="Batch runner for DreamSim4")
    ap.add_argument(
        "--models",
        default="opus4,sonnet4,k2,gpt5,v31",
        help="Comma-separated model aliases to cycle through for self-dialogue runs.",
    )
    ap.add_argument(
        "--pairs",
        default="",
        help="Explicit pairs like 'opus4:sonnet4,k2:gpt5'. If provided, overrides --models cycling.",
    )
    ap.add_argument("--runs", type=int, default=30, help="Total number of runs when cycling models (default: 30)")
    ap.add_argument("--max-turns", type=int, default=30, help="Maximum turns per run (default: 30)")
    ap.add_argument("--template", default="dreamsim4", help="Template name (default: dreamsim4)")
    ap.add_argument(
        "--out",
        default="BackroomsLogs/dreamsim4/dreamsim4_meta.jsonl",
        help="Metadata JSONL output path",
    )
    ap.add_argument("--no-stream", action="store_true", help="Do not stream logs; capture output silently")
    ap.add_argument("--max-context-frac", type=float, default=0.0, help="Early-stop when estimated prompt tokens exceed this fraction of the context window (0 disables)")
    ap.add_argument("--context-window", type=int, default=128000, help="Assumed context window for limiting model (default: 128000)")

    args = ap.parse_args()

    template = (args.template or "dreamsim4").strip()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs: List[Tuple[str, str]] = []
    if args.pairs.strip():
        pairs = parse_pairs(args.pairs)
        # Validate
        validate_models([m for pair in pairs for m in pair])
        planned_runs = len(pairs)
        print(f"Running {planned_runs} explicit pair run(s) for template '{template}'")
        run_plan = pairs
    else:
        models = [m.strip() for m in args.models.split(',') if m.strip()]
        if not models:
            raise SystemExit("Provide at least one model via --models or explicit pairs via --pairs")
        validate_models(models)
        runs = max(1, int(args.runs))
        print(f"Cycling through {len(models)} model(s) for {runs} run(s) — template '{template}'")
        run_plan = []
        for i in range(runs):
            m = models[i % len(models)]
            run_plan.append((m, m))

    completed = 0
    total = len(run_plan)
    for (m1, m2) in run_plan:
        models_pair = [m1, m2]
        start = dt.datetime.now(dt.timezone.utc).isoformat()
        t0 = time.time()

        proc = run_one(
            models_pair,
            args.max_turns,
            template,
            args.max_context_frac,
            args.context_window,
            stream=(not args.no_stream),
        )

        log_path = latest_log_for(models_pair, template)
        duration = time.time() - t0
        end = dt.datetime.now(dt.timezone.utc).isoformat()
        exit_reason = determine_exit_reason(log_path) if log_path else "unknown"

        meta = {
            "models": models_pair,
            "template": template,
            "max_turns": args.max_turns,
            "start": start,
            "end": end,
            "duration_sec": round(duration, 3),
            "log_file": str(log_path) if log_path else None,
            "exit_reason": exit_reason,
            "returncode": proc.returncode,
            "stderr_tail": (proc.stderr[-1000:] if (hasattr(proc, "stderr") and proc.stderr) else None),
        }
        with out_path.open("a", encoding="utf-8") as outf:
            outf.write(json.dumps(meta) + "\n")

        completed += 1
        print(f"[{completed}/{total}] pair={m1}-{m2} turns={args.max_turns} exit={exit_reason} time={duration:.1f}s")

    print(f"Done. Wrote metadata to {out_path}")


if __name__ == "__main__":
    main()

