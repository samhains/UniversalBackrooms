#!/usr/bin/env python3
"""
Import RSOS dream data (date + content only) from a TSV into Supabase `public.dreams`.

Reads data/rsos_dream_data.tsv and inserts rows with:
  - content: from `text_dream`
  - date: parsed from `dream_date` when possible (YYYY-MM-DD); else NULL or YYYY-01-01 for year-only
  - source: 'rsos'
  - source_ref: 'rsos:<dream_id>' (for idempotent upserts)

Usage:
  python scripts/import_rsos_dreams.py --file data/rsos_dream_data.tsv --limit 0

Environment:
  - SUPABASE_URL
  - SUPABASE_SERVICE_ROLE_KEY (preferred for writes) or SUPABASE_ANON_KEY

Notes:
  - Requires unique index on (source, source_ref) so upsert works without duplicates.
  - Uses on_conflict=source,source_ref for idempotency.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
from typing import Optional, List, Dict

import requests


def _env_keys() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY/ANON_KEY")
    return url.rstrip("/"), key


def _headers(key: str):
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def parse_date(raw: str) -> Optional[str]:
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    # Strip trailing '?'
    if s.endswith('?'):
        s = s[:-1].strip()
    # Year-only -> YYYY-01-01
    if len(s) == 4 and s.isdigit():
        try:
            dt.date(int(s), 1, 1)
            return f"{s}-01-01"
        except Exception:
            return None
    # Try common formats
    fmts = ["%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y", "%m/%Y", "%Y-%m"]
    for fmt in fmts:
        try:
            d = dt.datetime.strptime(s, fmt)
            # If format lacks day, default to 01
            year = d.year
            month = d.month if "%m" in fmt else 1
            day = d.day if "%d" in fmt else 1
            dt.date(year, month, day)
            return f"{year:04d}-{month:02d}-{day:02d}"
        except Exception:
            continue
    return None


def batch_upsert(url: str, key: str, rows: List[Dict], chunk_size: int = 500) -> int:
    endpoint = f"{url}/rest/v1/dreams"
    total = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        params = {"on_conflict": "source,source_ref"}
        r = requests.post(endpoint, headers=_headers(key), params=params, data=json.dumps(chunk), timeout=60)
        if r.status_code not in (200, 201, 204):
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            raise SystemExit(f"Upsert failed at batch {i//chunk_size}: {detail}")
        total += len(chunk)
    return total


def main():
    ap = argparse.ArgumentParser(description="Import RSOS dreams (date + content) into public.dreams")
    ap.add_argument("--file", default="data/rsos_dream_data.tsv", help="Path to RSOS TSV file")
    ap.add_argument("--limit", type=int, default=0, help="Limit rows to import (0 = all)")
    ap.add_argument("--dry-run", action="store_true", help="Parse and report, but do not write to DB")
    args = ap.parse_args()

    url, key = _env_keys()
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"dream_id", "dream_date", "text_dream"}
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise SystemExit(f"Missing columns in TSV: {', '.join(missing)}")
        rows: List[Dict] = []
        for idx, rec in enumerate(reader, start=1):
            if args.limit and len(rows) >= args.limit:
                break
            raw_text = (rec.get("text_dream") or "").strip()
            if not raw_text:
                continue
            dream_id = (rec.get("dream_id") or "").strip()
            raw_date = (rec.get("dream_date") or "").strip()
            iso_date = parse_date(raw_date)
            rows.append({
                "content": raw_text,
                "date": iso_date,
                "source": "rsos",
                "source_ref": f"rsos:{dream_id}" if dream_id else None,
            })

    print(f"Prepared {len(rows)} RSOS rows (limit={args.limit})")
    if args.dry_run:
        # Show a small sample for verification
        for s in rows[:3]:
            print(json.dumps(s, ensure_ascii=False)[:300])
        return

    inserted = batch_upsert(url, key, rows)
    print(f"Upserted {inserted} rows into public.dreams (source='rsos')")


if __name__ == "__main__":
    main()

