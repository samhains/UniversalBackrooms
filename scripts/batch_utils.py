#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import requests

# Repo root import path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_config import get_model_info


# --- Supabase helpers ---

def env_keys() -> Tuple[str, str]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL and/or SUPABASE_ANON_KEY (or SERVICE_ROLE).")
    return url.rstrip("/"), key


def headers(key: str) -> Dict[str, str]:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def fetch_recent(url: str, key: str, limit: int = 50, source: str | None = None):
    endpoint = f"{url}/rest/v1/dreams"
    params = {
        "select": "id,content,date,source",
        "order": "date.desc",
        "limit": str(limit),
    }
    if source and source != "all":
        params["source"] = f"eq.{source}"
    r = requests.get(endpoint, headers=headers(key), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def search_dreams(url: str, key: str, q: str, limit: int = 50, source: str | None = None):
    endpoint = f"{url}/rest/v1/dreams"
    q_pat = f"*{q}*"
    params = {
        "select": "id,content,date,source",
        "content": f"ilike.{q_pat}",
        "order": "date.desc",
        "limit": str(limit),
    }
    if source and source != "all":
        params["source"] = f"eq.{source}"
    r = requests.get(endpoint, headers=headers(key), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def read_dreams_from_supabase(query: Optional[str], limit: int, source: str = "mine") -> List[dict]:
    url, key = env_keys()
    try:
        rows = search_dreams(url, key, query, limit, source) if query else fetch_recent(url, key, limit, source)
    except requests.HTTPError as e:
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text if e.response is not None else str(e)
        raise SystemExit(f"Supabase request failed: {detail}")

    out: List[dict] = []
    for r in rows:
        content = (r.get("content") or "").strip()
        if not content:
            continue
        out.append(
            {
                **r,
                "id": r.get("id") or r.get("dream_id") or r.get("uuid"),
                "date": r.get("date") or r.get("created_at") or r.get("dream_date"),
                "content": content,
            }
        )
    return out


# --- Template vars ---

def write_template_vars(template: str, vars_map: Dict[str, Any]) -> None:
    base = ROOT / "templates" / template
    base.mkdir(parents=True, exist_ok=True)
    vars_path = base / "vars.json"
    with vars_path.open("w", encoding="utf-8") as f:
        json.dump(vars_map, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_dreamsim3_vars(dream_text: str) -> None:
    safe = " ".join((dream_text or "").split())
    write_template_vars("dreamsim3", {"DREAM_TEXT": safe})


# --- Backrooms log helpers ---

BACKROOMS_LOGS = Path("BackroomsLogs")


def latest_log_for(models: List[str], template: str) -> Optional[Path]:
    pattern = f"{'_'.join(models)}_{template}_"
    tmpl_dir = BACKROOMS_LOGS / template
    search_spaces = []
    if tmpl_dir.exists():
        search_spaces.append(tmpl_dir.rglob("*.txt"))
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


# --- Model helpers ---

def parse_pairs(pairs_str: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not pairs_str:
        return pairs
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


# --- Optional sync helper ---

try:
    from scripts.sync_backrooms import env_keys as _sync_env_keys  # type: ignore
    from scripts.sync_backrooms import to_backrooms_rows as _sync_to_rows  # type: ignore
    from scripts.sync_backrooms import upsert_rows as _sync_upsert  # type: ignore
except Exception:
    _sync_env_keys = None
    _sync_to_rows = None
    _sync_upsert = None


def sync_single_meta(meta: dict) -> None:
    if not (_sync_env_keys and _sync_to_rows and _sync_upsert):
        return
    try:
        url, key = _sync_env_keys()
    except Exception:
        return
    try:
        rows = _sync_to_rows([meta])
        if not rows:
            return
        _sync_upsert(url, key, rows, dry_run=False)
        print("  ↳ synced to Supabase (on_conflict=log_file)")
    except Exception as e:
        print(f"  ↳ sync failed: {e}")

