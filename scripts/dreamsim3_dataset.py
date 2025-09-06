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

import random

# Ensure repository root is importable when running as a script from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    # Prefer the template subfolder (backrooms now writes there)
    tmpl_dir = BACKROOMS_LOGS / template
    search_spaces = []
    if tmpl_dir.exists():
        search_spaces.append(tmpl_dir.rglob("*.txt"))
    # Fallback to top-level (for legacy runs)
    search_spaces.append(BACKROOMS_LOGS.glob("*.txt"))

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
    if max_context_frac and max_context_frac > 0:
        cmd += ["--max-context-frac", str(max_context_frac)]
        if context_window and context_window > 0:
            cmd += ["--context-window", str(context_window)]
    if stream:
        # Inherit stdout/stderr so logs stream live to the console
        return subprocess.run(cmd)
    else:
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
    ap.add_argument("--out", default="BackroomsLogs/dreamsim3/dreamsim3_meta.jsonl", help="Metadata JSONL output path")
    ap.add_argument("--mixed", action="store_true", help="Mix models into pairs (see --mixed-mode)")
    ap.add_argument("--mixed-mode", choices=["all", "random"], default="all", help="How to mix models when --mixed is set: 'all' unique pairs, or 'random' per dream")
    ap.add_argument("--runs-per-dream", type=int, default=1, help="When --mixed-mode=random, number of random pairs to run per dream (default: 1)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for mixed shuffling/sampling")
    ap.add_argument("--no-shuffle", action="store_true", help="Do not shuffle dream order (default: shuffled)")
    ap.add_argument("--no-stream", action="store_true", help="Do not stream logs; capture output silently (default: stream)")
    ap.add_argument("--max-context-frac", type=float, default=0.0, help="Early-stop when estimated prompt tokens exceed this fraction of the context window (0 disables)")
    ap.add_argument("--context-window", type=int, default=128000, help="Assumed context window size in tokens for the limiting model (default: 128000)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    # Resolve runs from either pairs or models
    rng = random.Random(args.seed) if args.seed is not None else random
    # Sanitize potential 'pairs=' or 'models=' prefixes inserted by shells/just
    raw_pairs = args.pairs
    if isinstance(raw_pairs, str) and raw_pairs.lower().startswith("pairs="):
        raw_pairs = raw_pairs.split("=", 1)[1]
    explicit_pairs = parse_pairs(raw_pairs)
    models_for_random: List[str] = []
    pairs_static: Optional[List[Tuple[str, str]]] = None
    if explicit_pairs:
        flat = [x for pair in explicit_pairs for x in pair]
        validate_models(flat)
        pairs_static = explicit_pairs
    else:
        raw_models = args.models
        if isinstance(raw_models, str) and raw_models.lower().startswith("models="):
            raw_models = raw_models.split("=", 1)[1]
        models = [m.strip() for m in raw_models.split(",") if m.strip()]
        if not models:
            sys.exit("Provide --pairs or at least one model via --models")
        validate_models(models)
        if args.mixed:
            if args.mixed_mode == "all":
                uniq_pairs: List[Tuple[str, str]] = []
                for i in range(len(models)):
                    for j in range(i + 1, len(models)):
                        uniq_pairs.append((models[i], models[j]))
                rng.shuffle(uniq_pairs)
                pairs_static = uniq_pairs
            else:
                # random per dream; store models for sampling later
                models_for_random = models
                pairs_static = None
        else:
            pairs_static = [(m, m) for m in models]

    # Prepare metadata sink
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = read_dreams_from_csv(csv_path)
    # Shuffle dreams by default to avoid oversampling early rows
    if not args.no_shuffle:
        # Reuse rng from mixed logic for reproducibility
        try:
            rng  # type: ignore  # will exist below; safe to fallback if not
        except NameError:
            rng = random.Random(args.seed) if args.seed is not None else random
        rng.shuffle(rows)
    if args.max_dreams > 0:
        rows = rows[: args.max_dreams]

    if pairs_static is not None:
        total_runs = len(rows) * len(pairs_static)
        print(f"Running {total_runs} simulations: {len(rows)} dreams x {len(pairs_static)} runs")
    else:
        runs_per_dream = max(1, int(args.runs_per_dream))
        total_runs = len(rows) * runs_per_dream
        print(f"Running {total_runs} simulations: {len(rows)} dreams x {runs_per_dream} random pairs")

    completed = 0
    for idx, row in enumerate(rows, start=1):
        dream_text = (row.get("content") or "").strip()
        # Update initiator for this dream
        INIT_OUTPUT.write_text(render_initiator(dream_text), encoding="utf-8")

        # Determine pairs for this dream
        if pairs_static is not None:
            pairs_for_dream = pairs_static
        else:
            runs_per_dream = max(1, int(args.runs_per_dream))
            if len(models_for_random) < 2:
                raise SystemExit("Need at least two models for --mixed-mode=random")

            # When users pass duplicates in --models (e.g., "gpt5,gpt5,hermes,k2"),
            # treat them as weighting, but still ensure a mixed pair (distinct aliases).
            def sample_weighted_distinct_pair() -> Tuple[str, str]:
                # Retry until two sampled entries have different aliases
                # Duplicates in the list increase their selection probability.
                for _ in range(100):  # bounded retries to avoid rare pathological cases
                    a, b = rng.sample(models_for_random, 2)
                    if a != b:
                        return a, b
                # As a last resort (e.g., list accidentally all-identical), fall back to simple sample
                a, b = rng.sample(models_for_random, 2)
                return a, b

            pairs_for_dream = [sample_weighted_distinct_pair() for _ in range(runs_per_dream)]

        # Log which dream is about to run before invoking backrooms (full text, not truncated)
        try:
            total_dreams = len(rows)
        except Exception:
            total_dreams = 0
        meta_bits = []
        if row.get("id"):
            meta_bits.append(f"id={row.get('id')}")
        if row.get("date"):
            meta_bits.append(f"date={row.get('date')}")
        pairs_label = ", ".join([f"{a}-{b}" for (a, b) in pairs_for_dream])
        header = f"=== Dream {idx}"
        if total_dreams:
            header += f"/{total_dreams}"
        if meta_bits:
            header += " (" + ", ".join(meta_bits) + ")"
        print("\n" + header)
        print("Text:")
        print(dream_text)
        print(f"Will run {len(pairs_for_dream)} pair(s): {pairs_label}")

        for (m1, m2) in pairs_for_dream:
            models_pair = [m1, m2]
            start = dt.datetime.now(dt.timezone.utc).isoformat()
            t0 = time.time()

            # Run the conversation
            proc = run_one(
                models_pair,
                args.max_turns,
                args.template,
                args.max_context_frac,
                args.context_window,
                stream=(not args.no_stream),
            )

            # Best-effort locate the log file created by backrooms
            log_path = latest_log_for(models_pair, args.template)
            duration = time.time() - t0
            end = dt.datetime.now(dt.timezone.utc).isoformat()
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
                "stderr_tail": (proc.stderr[-1000:] if (hasattr(proc, "stderr") and proc.stderr) else None),
            }
            with out_path.open("a", encoding="utf-8") as outf:
                outf.write(json.dumps(meta) + "\n")

            completed += 1
            pair_label = f"{m1}-{m2}"
            print(f"[{completed}/{total_runs}] dream#{idx} pair={pair_label} turns={args.max_turns} exit={exit_reason} time={duration:.1f}s")

    print(f"Done. Wrote metadata to {out_path}")


if __name__ == "__main__":
    main()
