#!/usr/bin/env python3
"""
Run Backrooms via JSON configs.

Usage:
  python scripts/run_config.py --config configs/<file>.json

Supports two modes:
  - type=single: invoke backrooms.py once with the given template/models/options
  - type=batch:  fetch items (e.g., dreams from Supabase), set template vars per item,
                 and run multiple conversations, writing a metadata JSONL

The config format is intentionally minimal and mirrors existing CLI args.
See examples in ./configs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import getpass

# Ensure repository root is importable when running as a script from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Optional dotenv
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

# Reuse helpers from dreamsim3_dataset where possible
try:
    from scripts.batch_utils import (
        read_dreams_from_supabase,
        parse_pairs,
        validate_models,
        latest_log_for,
        determine_exit_reason,
        sync_single_meta as _sync_single_meta,
    )
except Exception:
    # Fallback stubs for type hints; batch mode will fail if actually used without these
    def read_dreams_from_supabase(query: Optional[str], limit: int, source: str = "mine") -> List[dict]:
        raise SystemExit("Supabase helpers unavailable; cannot run batch mode.")

    def parse_pairs(pairs_str: str):
        raise SystemExit("parse_pairs unavailable; cannot run batch mode.")

    def validate_models(aliases: List[str]):  # type: ignore
        return None

    def latest_log_for(models: List[str], template: str):  # type: ignore
        return None

    def determine_exit_reason(log_file: Path):  # type: ignore
        return "unknown"

    def _sync_single_meta(meta: dict) -> None:  # type: ignore
        return None


def _load_env() -> None:
    if load_dotenv is not None:
        try:
            load_dotenv(ROOT / ".env")
        except Exception:
            pass


def _write_template_vars(template: str, vars_map: Dict[str, Any]) -> None:
    """Write templates/<template>/vars.json with the provided mapping.

    Values are written as-is; backrooms.py will escape braces during formatting.
    """
    base = ROOT / "templates" / template
    base.mkdir(parents=True, exist_ok=True)
    vars_path = base / "vars.json"
    with vars_path.open("w", encoding="utf-8") as f:
        json.dump(vars_map, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _run_backrooms(
    models: List[str],
    template: str,
    max_turns: int,
    *,
    max_context_frac: float = 0.0,
    context_window: int = 128000,
    discord_profile: Optional[Union[str, List[str]]] = None,
    media_preset: Optional[Union[str, List[str]]] = None,
    vars_inline: Optional[Dict[str, str]] = None,
    query_value: Optional[str] = None,
    max_tokens: Optional[int] = None,
    stream: bool = True,
    discord_overrides: Optional[Dict[str, Any]] = None,
    media_overrides: Optional[Dict[str, Any]] = None,
) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "backrooms.py",
        "--lm",
        *models,
        "--template",
        template,
        "--max-turns",
        str(max_turns),
    ]
    if max_context_frac and max_context_frac > 0:
        cmd += ["--max-context-frac", str(max_context_frac)]
        if context_window and context_window > 0:
            cmd += ["--context-window", str(context_window)]
    if max_tokens is not None and max_tokens > 0:
        cmd += ["--max-tokens", str(max_tokens)]
    if discord_profile:
        if isinstance(discord_profile, list):
            for dp in discord_profile:
                cmd += ["--discord", str(dp)]
        else:
            cmd += ["--discord", str(discord_profile)]
    if media_preset:
        if isinstance(media_preset, list):
            for mp in media_preset:
                cmd += ["--media", str(mp)]
        else:
            cmd += ["--media", str(media_preset)]
    if vars_inline:
        for k, v in vars_inline.items():
            if v is None:
                continue
            cmd += ["--var", f"{k}={v}"]
    if query_value:
        cmd += ["--query", query_value]

    env = os.environ.copy()
    mcp_cfg_path = str(ROOT / "mcp.config.json")
    if os.path.exists(mcp_cfg_path):
        env.setdefault("MCP_SERVERS_CONFIG", mcp_cfg_path)
    # Provide per-run Discord overrides to backrooms via env
    if discord_overrides:
        try:
            env["BACKROOMS_DISCORD_OVERRIDES"] = json.dumps(discord_overrides)
        except Exception:
            pass
    # Provide per-run Media overrides to backrooms via env
    if media_overrides:
        try:
            env["BACKROOMS_MEDIA_OVERRIDES"] = json.dumps(media_overrides)
        except Exception:
            pass
    run_kwargs = {"cwd": str(ROOT), "env": env}
    if stream:
        return subprocess.run(cmd, **run_kwargs)
    else:
        return subprocess.run(cmd, capture_output=True, text=True, **run_kwargs)


def _unique_index_pairs(models: List[str]) -> List[Tuple[str, str]]:
    """Return all index-based unique pairs i<j.
    Duplicate aliases can yield same-alias pairs (e.g., gpt5:gpt5).
    """
    out: List[Tuple[str, str]] = []
    n = len(models)
    for i in range(n):
        for j in range(i + 1, n):
            out.append((models[i], models[j]))
    return out


def run_single(cfg: Dict[str, Any]) -> None:
    template = cfg.get("template") or ""
    models = cfg.get("models") or []
    if not template or not models:
        raise SystemExit("single config requires 'template' and 'models' (list)")

    max_turns = int(cfg.get("max_turns", 30))
    max_context_frac = float(cfg.get("max_context_frac", 0.0))
    context_window = int(cfg.get("context_window", 128000))
    # Accept integrations via either top-level keys or an 'integrations' map (parity with batch mode)
    integrations = cfg.get("integrations") or {}
    discord_profile = cfg.get("discord") or integrations.get("discord")
    media_preset = cfg.get("media") or integrations.get("media")
    # Optional per-run Discord overrides at config level
    discord_overrides = (
        cfg.get("discord_overrides")
        or integrations.get("discord_overrides")
        or integrations.get("discord_options")
    )
    # Optional per-run Media overrides at config level
    media_overrides = (
        cfg.get("media_overrides")
        or integrations.get("media_overrides")
        or integrations.get("media_options")
    )
    # Simpler knobs: allow `integrations.post_transcript` and `integrations.transcript_channel`
    simple_overrides = {}
    if "post_transcript" in integrations:
        try:
            simple_overrides["post_transcript"] = bool(integrations.get("post_transcript"))
        except Exception:
            pass
    if "transcript_channel" in integrations and integrations.get("transcript_channel"):
        simple_overrides["transcript_channel"] = str(integrations.get("transcript_channel"))
    if simple_overrides:
        if not isinstance(discord_overrides, dict) or discord_overrides is None:
            discord_overrides = {}
        discord_overrides.update(simple_overrides)

    # Normalize to list or string; also allow comma-separated strings
    if isinstance(discord_profile, list):
        discord_val: Optional[Union[str, List[str]]] = [str(x) for x in discord_profile]
    elif isinstance(discord_profile, str) and "," in discord_profile:
        discord_val = [x.strip() for x in discord_profile.split(",") if x.strip()]
    else:
        discord_val = discord_profile

    if isinstance(media_preset, list):
        media_val: Optional[Union[str, List[str]]] = [str(x) for x in media_preset]
    elif isinstance(media_preset, str) and "," in media_preset:
        media_val = [x.strip() for x in media_preset.split(",") if x.strip()]
    else:
        media_val = media_preset
    vars_inline = cfg.get("vars") or None
    query_value = cfg.get("query") or None
    max_tokens = cfg.get("max_tokens")

    print(f"Running single: template='{template}', models={models}, turns={max_turns}")
    start = dt.datetime.now(dt.timezone.utc).isoformat()
    t0 = time.time()
    proc = _run_backrooms(
        models,
        template,
        max_turns,
        max_context_frac=max_context_frac,
        context_window=context_window,
        discord_profile=discord_val,
        media_preset=media_val,
        vars_inline=vars_inline,
        query_value=query_value,
        max_tokens=int(max_tokens) if isinstance(max_tokens, (int, str)) and str(max_tokens).isdigit() else None,
        stream=True,
        discord_overrides=discord_overrides if isinstance(discord_overrides, dict) else None,
        media_overrides=media_overrides if isinstance(media_overrides, dict) else None,
    )
    duration = time.time() - t0
    end = dt.datetime.now(dt.timezone.utc).isoformat()

    # Determine latest log file for this run
    log_path = latest_log_for(models, template)
    exit_reason = determine_exit_reason(log_path) if log_path else "unknown"

    # Where to write meta JSONL (default per-template path)
    out_cfg = cfg.get("output") or {}
    out_path = Path(out_cfg.get("meta_jsonl") or f"BackroomsLogs/{template}/{template}_meta.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "models": models,
        "template": template,
        "max_turns": max_turns,
        "start": start,
        "end": end,
        "duration_sec": round(duration, 3),
        "log_file": str(log_path) if log_path else None,
        "exit_reason": exit_reason,
        "returncode": proc.returncode,
        # Single-run configs may optionally include a prompt/query
        "prompt": (query_value or None),
    }
    _append_meta_jsonl(out_path, meta)

    # Optional immediate sync (if enabled like in batch mode)
    auto_sync = bool(cfg.get("auto_sync", False))
    if auto_sync:
        try:
            _sync_single_meta(meta)
        except Exception as e:
            print(f"  ↳ sync failed: {e}")


def run_batch(cfg: Dict[str, Any]) -> None:
    template = (cfg.get("template") or "").strip()
    if not template:
        raise SystemExit("batch config requires 'template'")

    max_turns = int(cfg.get("max_turns", 30))
    max_context_frac = float(cfg.get("max_context_frac", 0.0))
    context_window = int(cfg.get("context_window", 128000))
    max_tokens = cfg.get("max_tokens")

    # Integrations
    integrations = cfg.get("integrations") or {}
    discord_profile = integrations.get("discord")
    # Allow list or comma-separated profiles
    if isinstance(discord_profile, list):
        discord_val: Optional[Union[str, List[str]]] = [str(x) for x in discord_profile]
    elif isinstance(discord_profile, str) and "," in discord_profile:
        discord_val = [x.strip() for x in discord_profile.split(",") if x.strip()]
    else:
        discord_val = discord_profile
    media_preset = integrations.get("media")
    # Optional per-run Media overrides at config level
    media_overrides = (
        cfg.get("media_overrides")
        or integrations.get("media_overrides")
        or integrations.get("media_options")
    )
    # Optional per-run Discord overrides at config level
    discord_overrides = (
        cfg.get("discord_overrides")
        or integrations.get("discord_overrides")
        or integrations.get("discord_options")
    )
    # Simpler knobs: allow `integrations.post_transcript` and `integrations.transcript_channel`
    simple_overrides = {}
    if "post_transcript" in integrations:
        try:
            simple_overrides["post_transcript"] = bool(integrations.get("post_transcript"))
        except Exception:
            pass
    if "transcript_channel" in integrations and integrations.get("transcript_channel"):
        simple_overrides["transcript_channel"] = str(integrations.get("transcript_channel"))
    if simple_overrides:
        if not isinstance(discord_overrides, dict) or discord_overrides is None:
            discord_overrides = {}
        discord_overrides.update(simple_overrides)
    # Allow list of media presets
    if isinstance(media_preset, list):
        media_val: Optional[Union[str, List[str]]] = [str(x) for x in media_preset]
    elif isinstance(media_preset, str) and "," in media_preset:
        media_val = [x.strip() for x in media_preset.split(",") if x.strip()]
    else:
        media_val = media_preset

    # Data source
    ds = cfg.get("data_source") or {}
    kind = (ds.get("kind") or "supabase").strip().lower()
    rows: List[dict] = []
    source = None  # used only for Supabase metadata
    if kind in ("supabase",):
        query = ds.get("query") or ""
        limit = int(ds.get("limit", 200))
        source = ds.get("source", "mine")
        rows = read_dreams_from_supabase(query=query or None, limit=limit, source=source)

        # Shuffle/limit — default to shuffle when using a search query
        shuffle_flag = cfg.get("shuffle")
        if shuffle_flag is None:
            shuffle_flag = bool(query)
        if shuffle_flag:
            seed = cfg.get("seed")
            rng = random.Random(seed) if seed is not None else random
            rng.shuffle(rows)
        max_items = int(cfg.get("max_items", 0))
        if max_items > 0:
            rows = rows[:max_items]

        if not rows:
            print("No items returned from data source; nothing to do.")
            return
    elif kind in ("none", "local", "static", ""):  # no external rows; pure sequence
        pass
    else:
        raise SystemExit(f"Unsupported data_source.kind '{kind}' (supported: 'supabase', 'none')")

    # Pair/run plan
    pairs_cfg = cfg.get("pairs") or []
    models_cfg = cfg.get("models") or []
    mixed_cfg = cfg.get("mixed") or None

    pairs_static: Optional[List[Tuple[str, str]]] = None
    models_for_random: List[str] = []

    if pairs_cfg:
        pairs_str = ",".join(pairs_cfg)
        pairs_static = parse_pairs(pairs_str)
        validate_models([m for (a, b) in pairs_static for m in (a, b)])
    elif mixed_cfg is not None:
        if not models_cfg:
            raise SystemExit("When 'mixed' is set, provide a non-empty 'models' list")
        models_for_random = [str(m).strip() for m in models_cfg if str(m).strip()]
        validate_models(models_for_random)
        mode = (mixed_cfg.get("mode") or "all").strip().lower()
        if mode == "all":
            pairs_static = _unique_index_pairs(models_for_random)
        elif mode == "random":
            pairs_static = None  # computed per-item below
        else:
            raise SystemExit("mixed.mode must be 'all' or 'random'")
    else:
        # Self-dialogue per model
        if not models_cfg:
            raise SystemExit("Provide 'models' or 'pairs' or 'mixed'")
        models_list = [str(m).strip() for m in models_cfg if str(m).strip()]
        validate_models(models_list)
        pairs_static = [(m, m) for m in models_list]

    out = cfg.get("output") or {}
    out_path = Path(out.get("meta_jsonl") or f"BackroomsLogs/{template}/{template}_meta.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tvars_map = cfg.get("template_vars_from_item") or {"DREAM_TEXT": "content"}
    auto_sync = bool(cfg.get("auto_sync", False))

    # RNG for random mixed pairs per item
    seed = cfg.get("seed")
    rng = random.Random(seed) if seed is not None else random

    def sample_weighted_pair(src_models: List[str]) -> Tuple[str, str]:
        if len(src_models) >= 2:
            a, b = rng.sample(src_models, 2)
            return a, b
        elif len(src_models) == 1:
            return src_models[0], src_models[0]
        else:
            raise SystemExit("Provide at least one model for mixed random")

    total_runs = 0
    # Sequence-only mode (no rows): build a run plan and execute
    if kind in ("none", "local", "static", ""):
        sequence = cfg.get("sequence") or {}
        runs = int(sequence.get("runs", 30))
        # Build plan
        run_plan: List[Tuple[str, str]] = []
        if pairs_static is not None:
            run_plan = list(pairs_static)
        elif mixed_cfg is not None:
            mode = (mixed_cfg.get("mode") or "all").strip().lower()
            if mode == "all":
                # Unique pairs once
                run_plan = _unique_index_pairs(models_for_random)
            elif mode == "random":
                rpi = int((mixed_cfg or {}).get("runs_per_item", runs))
                rpi = max(1, rpi)
                for _ in range(rpi):
                    run_plan.append(sample_weighted_pair(models_for_random))
            else:
                raise SystemExit("mixed.mode must be 'all' or 'random'")
        else:
            # Cycle through models for N runs (self-dialogue)
            ml = [str(m).strip() for m in models_cfg if str(m).strip()]
            if not ml:
                raise SystemExit("Provide 'models' for sequence runs or use 'pairs'/'mixed'")
            for i in range(runs):
                m = ml[i % len(ml)]
                run_plan.append((m, m))

        print(f"Running {len(run_plan)} simulations — template '{template}'")
        completed = 0
        for (m1, m2) in run_plan:
            start = dt.datetime.now(dt.timezone.utc).isoformat()
            t0 = time.time()
            proc = _run_backrooms(
                [m1, m2],
                template,
                max_turns,
                max_context_frac=max_context_frac,
                context_window=context_window,
                discord_profile=discord_val,
                media_preset=media_val,
                vars_inline=None,
                query_value=None,
                max_tokens=int(max_tokens) if isinstance(max_tokens, (int, str)) and str(max_tokens).isdigit() else None,
                stream=True,
                media_overrides=media_overrides if isinstance(media_overrides, dict) else None,
            )
            log_path = latest_log_for([m1, m2], template)
            duration = time.time() - t0
            end = dt.datetime.now(dt.timezone.utc).isoformat()
            exit_reason = determine_exit_reason(log_path) if log_path else "unknown"
            meta = {
                "models": [m1, m2],
                "template": template,
                "max_turns": max_turns,
                "start": start,
                "end": end,
                "duration_sec": round(duration, 3),
                "log_file": str(log_path) if log_path else None,
                "exit_reason": exit_reason,
                "returncode": proc.returncode,
                "stderr_tail": (proc.stderr[-1000:] if (hasattr(proc, "stderr") and proc.stderr) else None),
            }
            _append_meta_jsonl(out_path, meta)
            completed += 1
            print(f"[{completed}/{len(run_plan)}] pair={m1}-{m2} turns={max_turns} exit={exit_reason} time={duration:.1f}s")
        print(f"Done. Wrote metadata to {out_path}")
        return

    # Supabase-backed mode below
    if pairs_static is not None:
        total_runs = len(rows) * len(pairs_static)
        print(f"Running {total_runs} simulations: {len(rows)} items x {len(pairs_static)} runs")
    else:
        runs_per_item = int((mixed_cfg or {}).get("runs_per_item", 1))
        runs_per_item = max(1, runs_per_item)
        total_runs = len(rows) * runs_per_item
        print(f"Running {total_runs} simulations: {len(rows)} items x {runs_per_item} random pairs")

    completed = 0
    for idx, row in enumerate(rows, start=1):
        # Build template vars from item fields
        vars_map: Dict[str, Any] = {}
        for var_name, field_name in tvars_map.items():
            vars_map[var_name] = row.get(field_name)
        _write_template_vars(template, vars_map)

        # Determine pairs for this item
        if pairs_static is not None:
            pairs_for_item = pairs_static
        else:
            runs_per_item = int((mixed_cfg or {}).get("runs_per_item", 1))
            runs_per_item = max(1, runs_per_item)
            pairs_for_item = [sample_weighted_pair(models_for_random) for _ in range(runs_per_item)]

        # Info header
        meta_bits = []
        if row.get("id"):
            meta_bits.append(f"id={row.get('id')}")
        if row.get("date"):
            meta_bits.append(f"date={row.get('date')}")
        src_val = row.get("source") or (None if source == "all" else source)
        if src_val:
            meta_bits.append(f"source={src_val}")
        pairs_label = ", ".join([f"{a}-{b}" for (a, b) in pairs_for_item])
        header = f"=== Item {idx}/{len(rows)}"
        if meta_bits:
            header += " (" + ", ".join(meta_bits) + ")"
        print("\n" + header)
        print("Text:")
        print((row.get("content") or "").strip())
        print(f"Will run {len(pairs_for_item)} pair(s): {pairs_label}")

        for (m1, m2) in pairs_for_item:
            models_pair = [m1, m2]
            start = dt.datetime.now(dt.timezone.utc).isoformat()
            t0 = time.time()

            proc = _run_backrooms(
                models_pair,
                template,
                max_turns,
                max_context_frac=max_context_frac,
                context_window=context_window,
                discord_profile=discord_profile,
                media_preset=media_preset,
                vars_inline=None,
                query_value=None,
                max_tokens=int(max_tokens) if isinstance(max_tokens, (int, str)) and str(max_tokens).isdigit() else None,
                stream=True,
                discord_overrides=discord_overrides if isinstance(discord_overrides, dict) else None,
                media_overrides=media_overrides if isinstance(media_overrides, dict) else None,
            )

            log_path = latest_log_for(models_pair, template)
            duration = time.time() - t0
            end = dt.datetime.now(dt.timezone.utc).isoformat()
            exit_reason = determine_exit_reason(log_path) if log_path else "unknown"

            meta = {
                "item_index": idx,
                "dream_index": idx,  # alias for compatibility
                "dream_id": row.get("id"),
                "date": row.get("date"),
                "prompt": (row.get("content") or "").strip(),
                "source": src_val,
                "models": models_pair,
                "template": template,
                "max_turns": max_turns,
                "start": start,
                "end": end,
                "duration_sec": round(duration, 3),
                "log_file": str(log_path) if log_path else None,
                "exit_reason": exit_reason,
                "returncode": proc.returncode,
                "stderr_tail": (proc.stderr[-1000:] if (hasattr(proc, "stderr") and proc.stderr) else None),
            }
            _append_meta_jsonl(out_path, meta)

            completed += 1
            print(f"[{completed}/{total_runs}] pair={m1}-{m2} turns={max_turns} exit={exit_reason} time={duration:.1f}s")

            if auto_sync:
                try:
                    _sync_single_meta(meta)
                except Exception as e:
                    print(f"  ↳ sync failed: {e}")

    print(f"Done. Wrote metadata to {out_path}")


def main():
    _load_env()
    ap = argparse.ArgumentParser(description="Run Backrooms via JSON configs")
    ap.add_argument("--config", required=True, help="Path to config JSON file")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    cfg_type = (cfg.get("type") or "single").strip().lower()
    if cfg_type == "single":
        run_single(cfg)
    elif cfg_type == "batch":
        run_batch(cfg)
    else:
        raise SystemExit("Config 'type' must be 'single' or 'batch'")


def _append_meta_jsonl(path: Path, meta: Dict[str, Any]) -> None:
    """Append a JSON object to a JSONL file with robust fallback on permission errors.

    If the target file exists but is not writable (e.g., created by another user),
    write to a per-user fallback file alongside it, and print a concise notice.
    """
    line = json.dumps(meta)
    try:
        with path.open("a", encoding="utf-8") as outf:
            outf.write(line + "\n")
        print(f"Wrote meta to {path}")
        return
    except PermissionError as e:
        user = getpass.getuser() or "unknown"
        alt = path.with_name(f"{path.stem}.{user}.jsonl")
        try:
            with alt.open("a", encoding="utf-8") as outf:
                outf.write(line + "\n")
            print(f"Meta not writable ({path}): {e}. Wrote to {alt}")
            return
        except Exception as e2:
            print(f"Failed to write meta to fallback {alt}: {e2}")
            raise
    except Exception as e:
        print(f"Failed to write meta to {path}: {e}")
        raise


if __name__ == "__main__":
    main()
