#!/usr/bin/env python3
"""
Batch runner for DreamSim3 seeded from Supabase.

Reads dreams from Supabase and, for each dream text, renders
`templates/dreamsim3/initiator.history.md` from the template file,
then invokes `backrooms.py` with a configurable list of models and max turns.

For each run, writes a metadata record to a JSONL file including:
- dream text (prompt), models used, template, max turns
- start/end timestamps and duration
- log file path created by backrooms
- exit reason: "max_turns" or "early_stop" (when ^C^C encountered)

Usage examples:
  python scripts/dreamsim3_dataset.py \
      --models gpt5,hermes,k2 \
      --max-turns 30

  # Limit number of dreams processed
  python scripts/dreamsim3_dataset.py --max-dreams 20

  # Mixed model pairs (model1 vs model2)
  python scripts/dreamsim3_dataset.py --pairs gpt5:hermes,hermes:k2 --max-turns 30

  # Mixed random pairs with weighting (same-model pairs allowed by default)
  python scripts/dreamsim3_dataset.py --models gpt5,gpt5,gpt5,hermes --mixed --mixed-mode=random --runs-per-dream 2

Notes:
- If --pairs is provided, it runs each pair as (model1, model2).
- Otherwise, --models is used as separate self-dialogue runs (model, model) for each model.
- The script is resilient to early termination runs triggered by "^C^C".
 - With --mixed --mixed-mode=all, all index-based unique pairs are generated; duplicates in --models can yield same-alias pairs (e.g., gpt5:gpt5).
 - With --mixed --mixed-mode=random, pairs are sampled per dream using duplicates as weights and may include same-alias pairs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import random
import os
import json

import requests

# Ensure repository root is importable when running as a script from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from model_config import get_model_info
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # optional dependency

# Optional: import sync helpers to upsert each run immediately
try:
    from sync_backrooms import env_keys as _sync_env_keys  # type: ignore
    from sync_backrooms import to_backrooms_rows as _sync_to_rows  # type: ignore
    from sync_backrooms import upsert_rows as _sync_upsert  # type: ignore
except Exception:
    _sync_env_keys = None
    _sync_to_rows = None
    _sync_upsert = None

TEMPLATES_DIR = Path("templates/dreamsim3")
INIT_TEMPLATE = TEMPLATES_DIR / "initiator.history.template.md"
VARS_FILE = TEMPLATES_DIR / "vars.json"
BACKROOMS_LOGS = Path("BackroomsLogs")


# CSV support removed — we rely solely on Supabase now.


# --- Supabase REST helpers (reused and adapted from scripts/seed_dreamsim3.py) ---

def _env_keys() -> Tuple[str, str]:
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
    endpoint = f"{url}/rest/v1/dreams"
    params = {
        # Include source so we can persist it to meta JSONL
        "select": "id,content,date,source",
        "order": "date.desc",
        "limit": str(limit),
    }
    if source and source != "all":
        params["source"] = f"eq.{source}"
    r = requests.get(endpoint, headers=_headers(key), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def search_dreams(url: str, key: str, q: str, limit: int = 50, source: str | None = None):
    """Search dreams via PostgREST filters (no RPC required).

    Uses ilike on content. Adjust this if you want to include other fields.
    """
    endpoint = f"{url}/rest/v1/dreams"
    # PostgREST ilike requires wrapping with *
    q_pat = f"*{q}*"
    params = {
        # Include source so we can persist it to meta JSONL
        "select": "id,content,date,source",
        "content": f"ilike.{q_pat}",
        "order": "date.desc",
        "limit": str(limit),
    }
    if source and source != "all":
        params["source"] = f"eq.{source}"
    r = requests.get(endpoint, headers=_headers(key), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def read_dreams_from_supabase(query: Optional[str], limit: int, source: str = "mine") -> List[dict]:
    url, key = _env_keys()
    try:
        rows = search_dreams(url, key, query, limit, source) if query else fetch_recent(url, key, limit, source)
    except requests.HTTPError as e:
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text if e.response is not None else str(e)
        sys.exit(f"Supabase request failed: {detail}")

    out: List[dict] = []
    for r in rows:
        content = (r.get("content") or "").strip()
        if not content:
            continue
        out.append(
            {
                **r,
                # normalize common fields used later for logging/metadata
                "id": r.get("id") or r.get("dream_id") or r.get("uuid"),
                "date": r.get("date") or r.get("created_at") or r.get("dream_date"),
                "content": content,
            }
        )
    return out


def write_vars(dream_text: str) -> None:
    safe = " ".join(dream_text.split())
    VARS_FILE.write_text(json.dumps({"DREAM_TEXT": safe}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    discord_profile: Optional[str] = None,
    media_preset: Optional[str] = None,
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
    if discord_profile:
        cmd += ["--discord", str(discord_profile)]
    if media_preset:
        cmd += ["--media", str(media_preset)]
    if stream:
        # Inherit stdout/stderr so logs stream live to the console
        return subprocess.run(cmd)
    else:
        return subprocess.run(cmd, capture_output=True, text=True)


def _sync_single_meta(meta: dict) -> None:
    """Best-effort upsert of a single run into Supabase.

    Reuses scripts/sync_backrooms helpers to map JSONL meta -> backrooms row and upsert.
    Designed to be non-fatal on failure so batch runs keep going.
    """
    if not (_sync_env_keys and _sync_to_rows and _sync_upsert):
        # Sync helpers unavailable (e.g., import error) — skip silently.
        return
    try:
        url, key = _sync_env_keys()
    except Exception:
        # Missing env or other issue — skip without interrupting batch
        return
    try:
        rows = _sync_to_rows([meta])
        if not rows:
            return
        _sync_upsert(url, key, rows, dry_run=False)
        print("  ↳ synced to Supabase (on_conflict=log_file)")
    except Exception as e:
        # Do not raise; just note and continue.
        print(f"  ↳ sync failed: {e}")


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
    # Load environment variables from project .env if python-dotenv is available
    if load_dotenv is not None:
        try:
            load_dotenv(ROOT / ".env")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Batch runner for DreamSim3 from Supabase")
    ap.add_argument("--query", default="", help="Fuzzy search query for Supabase (RPC dreams_search). Omit to fetch recent.")
    ap.add_argument("--limit", type=int, default=200, help="Supabase fetch limit for recent/search (default: 200)")
    ap.add_argument("--source", choices=["mine", "rsos", "all"], default="mine", help="Which source of dreams to use (default: mine)")
    ap.add_argument("--models", default="gpt5,hermes,k2", help="Comma-separated model aliases (each runs as model,model unless --pairs is given)")
    ap.add_argument("--pairs", default="", help="Comma-separated mixed pairs, e.g. 'gpt5:hermes,hermes:k2'")
    ap.add_argument("--max-turns", type=int, default=30, help="Maximum turns per run (default: 30)")
    ap.add_argument("--max-dreams", type=int, default=0, help="Limit number of dreams processed (0 = all)")
    ap.add_argument("--template", default="dreamsim3", help="Template name (default: dreamsim3)")
    ap.add_argument("--out", default="BackroomsLogs/dreamsim3/dreamsim3_meta.jsonl", help="Metadata JSONL output path")
    ap.add_argument("--mixed", action="store_true", help="Mix models into pairs (see --mixed-mode)")
    ap.add_argument(
        "--mixed-mode",
        choices=["all", "random"],
        default="all",
        help=(
            "How to mix models when --mixed is set: 'all' unique pairs, or 'random' per dream "
            "(duplicates in --models weight sampling; same-model pairs allowed)"
        ),
    )
    ap.add_argument("--runs-per-dream", type=int, default=1, help="When --mixed-mode=random, number of random pairs to run per dream (default: 1)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for mixed shuffling/sampling")
    ap.add_argument("--no-shuffle", action="store_true", help="Do not shuffle dream order (default: shuffled)")
    ap.add_argument("--no-stream", action="store_true", help="Do not stream logs; capture output silently (default: stream)")
    ap.add_argument("--discord", default="", help="Discord preset/profile name (./discord/<name>.json) passed to backrooms.py")
    ap.add_argument("--media", default="", help="Media preset name (./media/<name>.json) passed to backrooms.py")
    ap.add_argument("--max-context-frac", type=float, default=0.0, help="Early-stop when estimated prompt tokens exceed this fraction of the context window (0 disables)")
    ap.add_argument("--context-window", type=int, default=128000, help="Assumed context window size in tokens for the limiting model (default: 128000)")
    args = ap.parse_args()
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

    q = (args.query or "").strip() or None
    rows = read_dreams_from_supabase(q, int(args.limit), args.source)
    # Guardrail: common pitfall is zero results due to source filter.
    if not rows:
        hint = [
            "No dreams returned for the given parameters.",
            f"query={q!r} limit={args.limit} source={args.source!r}",
            "\nTroubleshooting:",
            "- If you expected matches, try a broader source filter: --source all",
            "  Example: python scripts/dreamsim3_dataset.py --query \"static\" --source all --limit 1000 --models sonnet3 --max-turns 30",
            "- Sanity check your query first: python scripts/search_dreams.py --query \"static\" --limit 200 --source all",
            "- If using OpenRouter models (e.g., sonnet3), ensure OPENROUTER_API_KEY is set.",
        ]
        print("\n".join(hint))
        sys.exit(1)
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
        # Update template vars for this dream (used by initiator.history.template.md)
        write_vars(dream_text)

        # Determine pairs for this dream
        if pairs_static is not None:
            pairs_for_dream = pairs_static
        else:
            runs_per_dream = max(1, int(args.runs_per_dream))

            # Weighted sampling: duplicates in --models increase selection probability.
            # Random mode allows same-model pairs by default (e.g., gpt5:gpt5).
            def sample_weighted_pair() -> Tuple[str, str]:
                if len(models_for_random) >= 2:
                    a, b = rng.sample(models_for_random, 2)
                    return a, b
                elif len(models_for_random) == 1:
                    a = b = models_for_random[0]
                    return a, b
                else:
                    raise SystemExit("Provide at least one model for --mixed-mode=random")

            pairs_for_dream = [sample_weighted_pair() for _ in range(runs_per_dream)]

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
        # Determine source: prefer database value; else fall back to run-level filter (except 'all')
        src_val = row.get("source") or (None if args.source == "all" else args.source)
        if src_val:
            meta_bits.append(f"source={src_val}")
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
                discord_profile=(args.discord or None),
                media_preset=(args.media or None),
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
                # Persist source (db -> run-level)
                "source": src_val,
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

            # Manual sync preferred: set BACKROOMS_AUTO_SYNC=1 to enable immediate upsert
            if os.getenv("BACKROOMS_AUTO_SYNC") == "1":
                _sync_single_meta(meta)

    print(f"Done. Wrote metadata to {out_path}")


if __name__ == "__main__":
    main()
