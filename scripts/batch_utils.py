#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Iterable, Union

import requests

# Repo root import path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model_config import get_model_info
from paths import BACKROOMS_LOGS_DIR


# --- Supabase helpers ---

def env_keys() -> Tuple[str, str]:
    url = os.getenv("SUPABASE_URL")
    # Accept common key names: ANON, SERVICE_ROLE, or legacy SUPABASE_KEY
    key = (
        os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
    )
    if not url or not key:
        raise SystemExit(
            "Missing SUPABASE_URL and/or SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY)."
        )
    return url.rstrip("/"), key


def headers(key: str) -> Dict[str, str]:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _normalize_select(select: Optional[Union[str, Iterable[str]]], default: str) -> str:
    if select is None:
        return default
    if isinstance(select, str):
        return select
    parts = [str(col).strip() for col in select if str(col).strip()]
    return ",".join(parts) if parts else default


def fetch_recent(
    url: str,
    key: str,
    *,
    table: str = "dreams",
    limit: int = 50,
    source: str | None = None,
    select: Optional[Union[str, Iterable[str]]] = None,
    order_column: str = "date",
    order_desc: bool = True,
    source_column: Optional[str] = "source",
):
    endpoint = f"{url}/rest/v1/{table}"
    params: Dict[str, str] = {
        "select": _normalize_select(select, "id,content,date,source"),
        "limit": str(limit),
    }
    if order_column:
        direction = "desc" if order_desc else "asc"
        params["order"] = f"{order_column}.{direction}"
    if source and source != "all" and source_column:
        params[source_column] = f"eq.{source}"
    r = requests.get(endpoint, headers=headers(key), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def search_dreams(
    url: str,
    key: str,
    q: str,
    *,
    table: str = "dreams",
    limit: int = 50,
    source: str | None = None,
    select: Optional[Union[str, Iterable[str]]] = None,
    search_column: str = "content",
    order_column: str = "date",
    order_desc: bool = True,
    source_column: Optional[str] = "source",
):
    endpoint = f"{url}/rest/v1/{table}"
    q_pat = f"*{q}*"
    params: Dict[str, str] = {
        "select": _normalize_select(select, "id,content,date,source"),
        search_column: f"ilike.{q_pat}",
        "limit": str(limit),
    }
    if order_column:
        direction = "desc" if order_desc else "asc"
        params["order"] = f"{order_column}.{direction}"
    if source and source != "all" and source_column:
        params[source_column] = f"eq.{source}"
    r = requests.get(endpoint, headers=headers(key), params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_by_ids(
    url: str,
    key: str,
    ids: Iterable[object],
    source: str | None = None,
    chunk_size: int = 50,
    *,
    table: str = "dreams",
    select: Optional[Union[str, Iterable[str]]] = None,
    id_column: str = "id",
    source_column: Optional[str] = "source",
):
    ids_list = [str(i).strip() for i in ids if str(i).strip()]
    if not ids_list:
        return []

    endpoint = f"{url}/rest/v1/{table}"
    out_rows: List[dict] = []
    for start in range(0, len(ids_list), chunk_size):
        chunk = ids_list[start : start + chunk_size]
        values: List[str] = []
        for val in chunk:
            if val.isdigit():
                values.append(val)
            else:
                escaped = val.replace("\"", "\\\"")
                values.append(f'"{escaped}"')
        params: Dict[str, str] = {
            "select": _normalize_select(select, "id,content,date,source"),
            id_column: f"in.({','.join(values)})",
            "limit": str(len(chunk)),
        }
        if source and source != "all" and source_column:
            params[source_column] = f"eq.{source}"
        r = requests.get(endpoint, headers=headers(key), params=params, timeout=30)
        r.raise_for_status()
        out_rows.extend(r.json())

    # Preserve the original ordering of IDs
    order_map = {str(idx): pos for pos, idx in enumerate(ids_list)}
    out_rows.sort(key=lambda row: order_map.get(str(row.get("id")), len(order_map)))
    return out_rows

def read_dreams_from_supabase(
    query: Optional[str],
    limit: int,
    source: str = "mine",
    ids: Optional[Iterable[object]] = None,
) -> List[dict]:
    return read_items_from_supabase(query, limit, source, ids)


def read_items_from_supabase(
    query: Optional[str],
    limit: int,
    source: str = "mine",
    ids: Optional[Iterable[object]] = None,
    *,
    table: str = "dreams",
    select: Optional[Union[str, Iterable[str]]] = None,
    search_column: str = "content",
    order_column: str = "date",
    order_desc: bool = True,
    source_column: Optional[str] = "source",
    id_column: str = "id",
    content_field: Optional[str] = "content",
    content_fallback_fields: Optional[List[str]] = None,
    date_fields: Optional[List[str]] = None,
    require_content: bool = True,
) -> List[dict]:
    url, key = env_keys()
    try:
        if ids is not None:
            unique_ids = []
            seen = set()
            for raw in ids:
                s = str(raw).strip()
                if not s or s in seen:
                    continue
                seen.add(s)
                unique_ids.append(s)
            if not unique_ids:
                rows = []
            else:
                if limit and limit > 0:
                    unique_ids = unique_ids[:limit]
                rows = fetch_by_ids(
                    url,
                    key,
                    unique_ids,
                    source,
                    table=table,
                    select=select,
                    id_column=id_column,
                    source_column=source_column,
                )
        elif query:
            rows = search_dreams(
                url,
                key,
                query,
                table=table,
                limit=limit,
                source=source,
                select=select,
                search_column=search_column,
                order_column=order_column,
                order_desc=order_desc,
                source_column=source_column,
            )
        else:
            rows = fetch_recent(
                url,
                key,
                table=table,
                limit=limit,
                source=source,
                select=select,
                order_column=order_column,
                order_desc=order_desc,
                source_column=source_column,
            )
    except requests.HTTPError as e:
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text if e.response is not None else str(e)
        raise SystemExit(f"Supabase request failed: {detail}")

    content_candidates: List[str] = []
    if content_field:
        content_candidates.append(content_field)
    if content_fallback_fields:
        for field in content_fallback_fields:
            if field and field not in content_candidates:
                content_candidates.append(field)
    for fallback in ("content", "dream_text", "text"):
        if fallback not in content_candidates:
            content_candidates.append(fallback)

    date_candidates: List[str] = []
    if date_fields:
        date_candidates.extend([f for f in date_fields if f])
    for fallback in ("date", "created_at", "dream_date"):
        if fallback not in date_candidates:
            date_candidates.append(fallback)

    out: List[dict] = []
    for r in rows:
        raw_content = ""
        for cand in content_candidates:
            value = r.get(cand)
            if value:
                raw_content = str(value)
                break
        content = raw_content.strip()
        if require_content and not content:
            continue

        row_out = {**r}
        if content_field and content_field != "content" and content_field not in row_out:
            row_out[content_field] = raw_content
        if content:
            row_out["content"] = content
        elif require_content:
            row_out["content"] = content

        row_out.setdefault("id", r.get("id") or r.get("dream_id") or r.get("uuid"))
        for cand in date_candidates:
            date_val = r.get(cand)
            if date_val:
                row_out["date"] = date_val
                break

        out.append(row_out)
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

BACKROOMS_LOGS = BACKROOMS_LOGS_DIR


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
