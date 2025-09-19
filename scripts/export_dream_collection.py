#!/usr/bin/env python3
"""
Fetch dreams from Supabase and write a single Obsidian-friendly Markdown
collection for quick review, including related backrooms transcripts.

Designed for ad-hoc reading sessions instead of long-term archival.

Example:
  python scripts/export_dream_collection.py \
    --contains "harry" \
    --name "harry-dreams" \
    --title "Dreams about Harry"

This writes obsidian/Dreams/harry-dreams.md (unless --subdir is changed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import requests

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # optional dependency
    load_dotenv = None


ROOT = Path(__file__).resolve().parents[1]


def _env_keys() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL and SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY)")
    return url.rstrip("/"), key


def _headers(key: str) -> dict[str, str]:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _slugify(text: str, default: str = "collection", maxlen: int = 80) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return (text[:maxlen] or default).strip("-") or default


def fetch_dreams(
    url: str,
    key: str,
    *,
    contains: Optional[str] = None,
    dream_ids: Optional[Sequence[int]] = None,
    source: Optional[str] = None,
    limit: int = 200,
) -> List[dict]:
    endpoint = f"{url}/rest/v1/dreams"
    params: dict[str, str] = {
        "select": "id,content,date,source,source_ref",
        "order": "id.asc",
        "limit": str(limit),
    }
    if contains:
        params["content"] = f"ilike.*{contains}*"
    if source and source != "all":
        params["source"] = f"eq.{source}"
    if dream_ids:
        id_list = ",".join(str(int(i)) for i in dream_ids)
        params["id"] = f"in.({id_list})"
        # Keep limit large so PostgREST doesn't truncate
        params["limit"] = str(max(limit, len(dream_ids)))

    response = requests.get(endpoint, headers=_headers(key), params=params, timeout=60)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        return []

    cleaned: List[dict] = []
    for row in rows:
        content = (row.get("content") or "").strip()
        if not content:
            continue
        cleaned.append(
            {
                "id": row.get("id"),
                "content": content,
                "date": row.get("date"),
                "source": row.get("source"),
                "source_ref": row.get("source_ref"),
            }
        )
    return cleaned


def fetch_transcripts(
    url: str,
    key: str,
    *,
    dream_ids: Sequence[int],
    limit: int = 500,
) -> Dict[int, List[dict]]:
    if not dream_ids:
        return {}

    endpoint = f"{url}/rest/v1/backrooms"
    id_list = ",".join(str(int(i)) for i in dream_ids)
    params: dict[str, str] = {
        "select": "id,dream_id,created_at,models,exit_reason,transcript,prompt,template,log_file,duration_sec,max_turns",
        "dream_id": f"in.({id_list})",
        # Fetch newest transcripts first so each dream's list is most-recent-first
        "order": "created_at.desc",
        "limit": str(max(limit, len(dream_ids))),
    }

    response = requests.get(endpoint, headers=_headers(key), params=params, timeout=60)
    response.raise_for_status()
    rows = response.json()
    if not isinstance(rows, list):
        return {}

    grouped: Dict[int, List[dict]] = {}
    for row in rows:
        did = row.get("dream_id")
        if did is None:
            continue
        try:
            did_int = int(did)
        except (TypeError, ValueError):
            continue
        transcript_text = (row.get("transcript") or "").strip()
        if not transcript_text:
            continue
        entry = {
            "id": row.get("id"),
            "dream_id": did_int,
            "created_at": row.get("created_at"),
            "models": row.get("models"),
            "exit_reason": row.get("exit_reason"),
            "template": row.get("template"),
            "prompt": row.get("prompt"),
            "transcript": transcript_text,
            "log_file": row.get("log_file"),
            "duration_sec": row.get("duration_sec"),
            "max_turns": row.get("max_turns"),
        }
        grouped.setdefault(did_int, []).append(entry)

    return grouped


def build_markdown(
    rows: Sequence[dict],
    *,
    title: str,
    contains: Optional[str],
    dream_ids: Iterable[int],
    source: Optional[str],
    transcripts_by_dream: Dict[int, List[dict]],
    transcripts_total: int,
) -> str:
    dream_ids_list = sorted({int(i) for i in dream_ids})
    header = title or "Dream Collection"

    frontmatter_lines = ["---", "type: dream_collection", f"title: {header}"]
    frontmatter_lines.append(f"generated_at: {dt.datetime.utcnow().isoformat(timespec='seconds')}Z")
    if contains:
        frontmatter_lines.append(f"contains: {contains}")
    if source and source != "all":
        frontmatter_lines.append(f"source: {source}")
    if dream_ids_list:
        frontmatter_lines.append("dream_ids:")
        for did in dream_ids_list:
            frontmatter_lines.append(f"  - {did}")
        frontmatter_lines.append(f"count: {len(dream_ids_list)}")
    else:
        frontmatter_lines.append(f"count: {len(rows)}")
    frontmatter_lines.append(f"transcripts_total: {transcripts_total}")
    frontmatter_lines.append("---\n")

    body_lines: List[str] = [f"# {header}\n"]

    if rows:
        body_lines.append("## Dream Index\n")
        for row in rows:
            did_value = row.get("id")
            if did_value is None:
                continue
            first_sentence = (row.get("content") or "").split("\n", 1)[0][:80]
            preview = first_sentence + ("…" if len(first_sentence) == 80 else "")
            try:
                did_int = int(did_value)
            except (TypeError, ValueError):
                did_int = None
            transcript_count = len(transcripts_by_dream.get(did_int, [])) if did_int is not None else 0
            index_line = f"- [[#Dream {did_value}|Dream {did_value}]]"
            if transcript_count:
                index_line += f" ({transcript_count} transcript{'s' if transcript_count != 1 else ''})"
            if preview:
                index_line += f" — {preview}"
            body_lines.append(index_line)
        body_lines.append("")

    for row in rows:
        did_value = row.get("id")
        heading = f"## Dream {did_value}" if did_value is not None else "## Dream"
        if row.get("date"):
            heading += f" ({row['date']})"
        body_lines.append(heading)

        meta_bits = []
        if row.get("source"):
            meta_bits.append(f"Source: `{row['source']}`")
        if row.get("source_ref"):
            meta_bits.append(f"Source Ref: `{row['source_ref']}`")
        if meta_bits:
            body_lines.append("  " + " · ".join(meta_bits))

        body_lines.append("")
        body_lines.append(row.get("content", "").rstrip() + "\n")

        try:
            did_int = int(did_value) if did_value is not None else None
        except (TypeError, ValueError):
            did_int = None
        transcripts = transcripts_by_dream.get(did_int, []) if did_int is not None else []
        if transcripts:
            body_lines.append("### Transcripts\n")
            for idx, tr in enumerate(transcripts, 1):
                tr_id = tr.get("id")
                heading = f"#### Transcript {tr_id}" if tr_id is not None else f"#### Transcript {idx}"
                created = tr.get("created_at")
                if created:
                    heading += f" — {created}"
                body_lines.append(heading)

                meta_bits: List[str] = []
                models = tr.get("models")
                if isinstance(models, list) and models:
                    meta_bits.append("models: " + ", ".join(str(m) for m in models))
                elif isinstance(models, str) and models:
                    meta_bits.append(f"models: {models}")
                if tr.get("template"):
                    meta_bits.append(f"template: {tr['template']}")
                if tr.get("exit_reason"):
                    meta_bits.append(f"exit: {tr['exit_reason']}")
                if tr.get("duration_sec"):
                    meta_bits.append(f"duration_sec: {tr['duration_sec']}")
                if tr.get("max_turns"):
                    meta_bits.append(f"max_turns: {tr['max_turns']}")
                if tr.get("log_file"):
                    meta_bits.append(f"log: {tr['log_file']}")
                if meta_bits:
                    body_lines.append("  " + " · ".join(meta_bits))

                prompt_text = (tr.get("prompt") or "").strip()
                if prompt_text:
                    body_lines.append("")
                    body_lines.append("_Prompt:_\n")
                    body_lines.append(prompt_text)

                body_lines.append("")
                transcript_text = tr.get("transcript", "")
                body_lines.append(transcript_text.rstrip() + "\n")

    return "\n".join(frontmatter_lines + body_lines)


def ensure_output_path(base: Path, subdir: str, name: str) -> Path:
    collection_dir = base / subdir
    collection_dir.mkdir(parents=True, exist_ok=True)
    filename = name if name.endswith(".md") else f"{name}.md"
    return collection_dir / filename


def main() -> None:
    if load_dotenv:
        try:
            load_dotenv(ROOT / ".env")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Export a filtered dream collection to Obsidian Markdown")
    ap.add_argument("--contains", help="Case-insensitive substring filter for dream content")
    ap.add_argument("--ids", nargs="*", type=int, help="Explicit dream IDs to include (overrides --contains)")
    ap.add_argument("--source", default="mine", help="Dream source filter (default: mine, use 'all' to disable)")
    ap.add_argument("--limit", type=int, default=200, help="Maximum dreams to fetch (default: 200)")
    ap.add_argument("--transcript-limit", type=int, default=500, help="Maximum transcripts to fetch (default: 500)")
    ap.add_argument("--vault", default=str(ROOT / "obsidian"), help="Destination Obsidian folder")
    ap.add_argument(
        "--subdir",
        default="Dreams",
        help="Subdirectory inside the vault for the generated collection (default: Dreams)",
    )
    ap.add_argument("--name", default="", help="Output filename (without extension). Defaults to slugified query")
    ap.add_argument("--title", default="", help="Override the document title")
    ap.add_argument("--no-overwrite", action="store_true", help="Fail if the output file already exists")
    ap.add_argument("--no-transcripts", action="store_true", help="Skip fetching backrooms transcripts")
    args = ap.parse_args()

    if not args.contains and not args.ids:
        ap.error("Provide --contains or at least one explicit --ids value")

    url, key = _env_keys()

    dream_rows = fetch_dreams(
        url,
        key,
        contains=args.contains,
        dream_ids=args.ids,
        source=args.source,
        limit=args.limit,
    )

    if not dream_rows:
        print("No dreams found for the given parameters.")
        return

    dream_rows.sort(key=lambda r: (r.get("id") or 0))
    dream_ids: List[int] = []
    for r in dream_rows:
        try:
            if r.get("id") is None:
                continue
            dream_ids.append(int(r.get("id")))
        except (TypeError, ValueError):
            continue

    if args.title:
        title = args.title
    elif args.contains:
        title = f"Dreams containing '{args.contains}'"
    elif args.ids:
        title = "Dreams " + ", ".join(str(i) for i in args.ids)
    else:
        title = "Dream Collection"

    name_source = args.name or args.title or (args.contains or "dream-collection")
    slug = _slugify(name_source)
    output_path = ensure_output_path(Path(args.vault), args.subdir, slug)

    if args.no_overwrite and output_path.exists():
        raise SystemExit(f"Refusing to overwrite existing file: {output_path}")

    transcripts_by_dream: Dict[int, List[dict]] = {}
    transcripts_total = 0
    if dream_ids and not args.no_transcripts:
        transcripts_by_dream = fetch_transcripts(
            url,
            key,
            dream_ids=dream_ids,
            limit=args.transcript_limit,
        )
        transcripts_total = sum(len(v) for v in transcripts_by_dream.values())

    markdown = build_markdown(
        dream_rows,
        title=title,
        contains=args.contains,
        dream_ids=dream_ids,
        source=args.source,
        transcripts_by_dream=transcripts_by_dream,
        transcripts_total=transcripts_total,
    )

    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {len(dream_rows)} dreams to {output_path}")


if __name__ == "__main__":
    main()
