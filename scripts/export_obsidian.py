#!/usr/bin/env python3
"""
Export Backrooms transcripts from Supabase to an Obsidian-friendly folder.

Writes one Markdown note per transcript with YAML frontmatter for Dataview:
  - type: transcript
  - dream_id, models, template, created_at, duration_sec, max_turns, exit_reason
  - model_pair (joined), prompt_title (truncated), prompt_hash (stable), source_log

Also optionally writes a simple Index.md and per-prompt pages to browse by
dream_id inside Obsidian using Dataview blocks.

Usage examples:
  # Full export into ./obsidian (default) with transcripts in Transcripts/
  python scripts/export_obsidian.py

  # Only new items since a date
  python scripts/export_obsidian.py --since 2025-09-01

  # Filter by dream_id or by prompt substring
  python scripts/export_obsidian.py --dream-id 123e4567-e89b-12d3-a456-426614174000
  python scripts/export_obsidian.py --prompt-contains "rain-soaked museum"

  # Write index pages
  python scripts/export_obsidian.py --write-index

Env:
  SUPABASE_URL, SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY)
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional

import requests

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # optional
    load_dotenv = None


ROOT = Path(__file__).resolve().parents[1]


def _env_keys() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit("Missing SUPABASE_URL and SUPABASE_ANON_KEY (or SUPABASE_SERVICE_ROLE_KEY)")
    return url.rstrip("/"), key


def _headers(key: str) -> dict:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def iso_date(s: str) -> str:
    try:
        # Accept YYYY-MM-DD or full ISO
        if len(s) == 10:
            return dt.datetime.fromisoformat(s).strftime("%Y-%m-%dT00:00:00Z")
        # Pass-through if already ISO-like
        dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        return s
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid date: {s}")


def _slugify(s: str, maxlen: int = 50) -> str:
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:maxlen] or "untitled"


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def _select_fields() -> str:
    # Keep this in sync with scripts/sync_backrooms.py mapping
    cols = [
        "id",
        "dream_id",
        "prompt",
        "models",
        "template",
        "max_turns",
        "created_at",
        "duration_sec",
        "log_file",
        "exit_reason",
        "transcript",
    ]
    return ",".join(cols)


def fetch_backrooms(
    url: str,
    key: str,
    *,
    since: Optional[str] = None,
    dream_id: Optional[str] = None,
    prompt_contains: Optional[str] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    endpoint = f"{url}/rest/v1/backrooms"
    params: Dict[str, str] = {
        "select": _select_fields(),
        "order": "created_at.desc",
        "limit": str(limit),
    }
    if since:
        params["created_at"] = f"gte.{since}"
    if dream_id:
        params["dream_id"] = f"eq.{dream_id}"
    if prompt_contains:
        params["prompt"] = f"ilike.*{prompt_contains}*"

    r = requests.get(endpoint, headers=_headers(key), params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()
    if not isinstance(rows, list):
        return []
    return rows


def fetch_dreams(
    url: str,
    key: str,
    *,
    source: Optional[str] = None,
    contains: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    endpoint = f"{url}/rest/v1/dreams"
    params: Dict[str, str] = {
        "select": "id,content,date,source,source_ref",
        "order": "date.desc.nullslast,id.desc",
        "limit": str(limit),
    }
    if source and source != "all":
        params["source"] = f"eq.{source}"
    if contains:
        params["content"] = f"ilike.*{contains}*"

    r = requests.get(endpoint, headers=_headers(key), params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()
    if not isinstance(rows, list):
        return []

    cleaned: List[Dict[str, Any]] = []
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


def ensure_dirs(base: Path) -> dict[str, Path]:
    transcripts_dir = base / "Transcripts"
    prompts_dir = base / "Prompts"
    models_dir = base / "Models"
    dreams_dir = base / "Dreams"
    base.mkdir(parents=True, exist_ok=True)
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    dreams_dir.mkdir(parents=True, exist_ok=True)
    return {
        "base": base,
        "transcripts": transcripts_dir,
        "prompts": prompts_dir,
        "models": models_dir,
        "dreams": dreams_dir,
    }


def to_frontmatter(row: Dict[str, Any]) -> Dict[str, Any]:
    prompt_text = (row.get("prompt") or "").strip()
    models = row.get("models") or []
    model_pair = "-".join(models) if isinstance(models, list) else str(models)
    # Normalize individual model fields and an order-independent set label
    model_a = models[0] if isinstance(models, list) and len(models) > 0 else None
    model_b = models[1] if isinstance(models, list) and len(models) > 1 else None
    model_set = None
    if isinstance(models, list) and models:
        model_set = "-".join(sorted(models))
    prompt_title = prompt_text[:80].replace("\n", " ") if prompt_text else ""
    fm: Dict[str, Any] = {
        "type": "transcript",
        "dream_id": row.get("dream_id"),
        "source": row.get("source"),
        "template": row.get("template"),
        "models": models,
        "model_pair": model_pair,
        "model_a": model_a,
        "model_b": model_b,
        "model_set": model_set,
        "created_at": row.get("created_at"),
        "duration_sec": row.get("duration_sec"),
        "max_turns": row.get("max_turns"),
        "exit_reason": row.get("exit_reason"),
        "source_log": row.get("log_file"),
        "prompt_title": prompt_title,
        "prompt_hash": _short_hash(prompt_text) if prompt_text else None,
    }
    # Remove empty keys
    return {k: v for k, v in fm.items() if v not in (None, "", [])}


def write_markdown(transcripts_dir: Path, row: Dict[str, Any], overwrite: bool = False) -> Path:
    prompt_text = (row.get("prompt") or "").strip()
    transcript = row.get("transcript") or ""
    created = (row.get("created_at") or "").replace(":", "-")
    created_safe = created[:19] if created else ""
    dream_id = row.get("dream_id") or "unknown"
    models = row.get("models") or []
    model_pair = "-".join(models) if isinstance(models, list) else str(models)
    title_bits = [b for b in [created_safe, dream_id, model_pair] if b]
    base_name = "_".join(title_bits) if title_bits else f"transcript_{row.get('id') or _short_hash(prompt_text)}"
    base_name = re.sub(r"[^A-Za-z0-9_\-]+", "_", base_name)
    path = transcripts_dir / f"{base_name}.md"

    if path.exists() and not overwrite:
        return path

    fm = to_frontmatter(row)
    # YAML frontmatter
    yaml_lines = ["---"]
    for k, v in fm.items():
        if isinstance(v, list):
            yaml_lines.append(f"{k}:")
            for item in v:
                yaml_lines.append(f"  - {item}")
        else:
            yaml_lines.append(f"{k}: {v}")
    yaml_lines.append("---\n")

    # Body
    chunks: List[str] = []
    if prompt_text:
        chunks.append("## Prompt\n")
        chunks.append(prompt_text + "\n")
    if transcript:
        chunks.append("## Transcript\n")
        chunks.append(transcript if transcript.endswith("\n") else transcript + "\n")

    body = "\n".join(chunks)
    path.write_text("\n".join(yaml_lines) + body, encoding="utf-8")
    return path


def write_dream_markdown(
    dreams_dir: Path,
    row: Dict[str, Any],
    *,
    groups: Optional[Iterable[str]] = None,
    search_contains: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    content = (row.get("content") or "").strip()
    dream_id = row.get("id")
    id_part = str(dream_id) if dream_id is not None else _short_hash(content or "")
    preview = content.splitlines()[0][:60] if content else ""
    slug = _slugify(preview, maxlen=60)
    fname = f"{id_part}--{slug}.md" if slug else f"{id_part}.md"
    path = dreams_dir / fname

    if path.exists() and not overwrite:
        return path

    group_list = [g for g in (groups or []) if g]
    # Deduplicate while preserving order
    seen: dict[str, None] = {}
    for g in group_list:
        if g not in seen:
            seen[g] = None
    yaml_lines = ["---"]
    fm: Dict[str, Any] = {
        "type": "dream",
        "dream_id": dream_id,
        "date": row.get("date"),
        "source": row.get("source"),
    }
    if row.get("source_ref"):
        fm["source_ref"] = row.get("source_ref")
    if search_contains:
        fm["search_contains"] = search_contains
    if seen:
        fm["groups"] = list(seen.keys())

    for k, v in fm.items():
        if v in (None, "", []):
            continue
        if isinstance(v, list):
            yaml_lines.append(f"{k}:")
            for item in v:
                yaml_lines.append(f"  - {item}")
        else:
            yaml_lines.append(f"{k}: {v}")
    yaml_lines.append("---\n")

    body_lines: List[str] = ["## Dream"]
    if content:
        body_lines.append("")
        body_lines.append(content if content.endswith("\n") else content + "\n")
    body = "\n".join(body_lines)
    if not body.endswith("\n"):
        body += "\n"
    path.write_text("\n".join(yaml_lines) + body, encoding="utf-8")
    return path


def write_prompt_page(prompts_dir: Path, dream_id: str, sample_prompt: str, overwrite: bool = True) -> Path:
    # Canonical filename: <id>--<slugified-prompt>.md
    id_slug = _slugify(dream_id) or 'unknown'
    preview_slug = _slugify((sample_prompt or '').splitlines()[0][:60]) if sample_prompt else ''
    fname = f"{id_slug}--{preview_slug}.md" if preview_slug else f"{id_slug}.md"
    path = prompts_dir / fname
    if path.exists() and not overwrite:
        return path
    title = (sample_prompt or "").splitlines()[0][:100]
    fm = {"type": "prompt_view", "dream_id": dream_id, "title": title}
    yaml = ["---"] + [f"{k}: {v}" for k, v in fm.items()] + ["---\n"]
    # Basic table (all transcripts of this prompt)
    dv_all = (
        "```dataview\n"
        "table file.link as Transcript, model_pair, created_at, duration_sec, exit_reason\n"
        "from \"Transcripts\"\n"
        "where type = \"transcript\" and dream_id = this.dream_id\n"
        "sort created_at desc\n"
        "```\n"
    )
    header_line = f"# Prompt {dream_id} — {title}" if title else f"# Prompt {dream_id}"
    body = (
        header_line + "\n\n" + (sample_prompt + "\n\n" if sample_prompt else "") +
        "## All Runs\n\n" + dv_all + "\n"
    )
    path.write_text("\n".join(yaml) + body, encoding="utf-8")
    # Best-effort: detect legacy duplicates that share the same dream_id
    try:
        dupes: list[Path] = []
        for p in prompts_dir.glob("*.md"):
            if p == path:
                continue
            try:
                head = p.read_text(encoding="utf-8")[:600]
            except Exception:
                continue
            if f"dream_id: {dream_id}" in head:
                dupes.append(p)
        if dupes:
            print(f"[prompt-dupes] {dream_id}: canonical={path.name} duplicates={[d.name for d in dupes]}")
    except Exception:
        pass
    return path


def write_model_page(models_dir: Path, model: str, overwrite: bool = True) -> Path:
    slug = _slugify(model)
    path = models_dir / f"{slug}.md"
    if path.exists() and not overwrite:
        return path
    fm = {"type": "model_view", "model": model}
    yaml = ["---"] + [f"{k}: {v}" for k, v in fm.items()] + ["---\n"]
    # Use DataviewJS for robust filtering on array membership and grouping by prompt
    dvjs = """```dataviewjs
const M = dv.current().model;
const pages = dv.pages('"Transcripts"')
  .where(p => p.type === 'transcript' && Array.isArray(p.models) && p.models.includes(M))
  .sort(p => p.created_at, 'desc');
const groups = pages.groupBy(p => (p.dream_id ? p.dream_id : 'unknown'));
function previewFor(group) {
  let t = '';
  for (const r of group.rows) { if (r.prompt_title) { t = r.prompt_title; break; } }
  if (!t) {
    const pr = dv.pages('"Prompts"').where(x => x.dream_id === group.key).first();
    if (pr && pr.title) t = pr.title;
  }
  return t;
}
for (const g of groups) {
  const label = previewFor(g);
  dv.header(3, 'Prompt ' + g.key + (label ? ' - ' + label : ''));
  const pr = dv.pages('"Prompts"').where(x => x.dream_id === g.key).first();
  if (pr) dv.paragraph('Prompt page: ' + pr.file.link);
  dv.table(['Transcript','ModelPair','Date','Exit'],
    g.rows.map(p => [p.file.link, (p.model_pair || ''), (p.created_at || ''), (p.exit_reason || '')])
  );
}
```"""
    body = f"# Model {model}\n\n" + dvjs
    path.write_text("\n".join(yaml) + body, encoding="utf-8")
    return path


def write_dream_group_page(
    dreams_dir: Path,
    group_label: str,
    *,
    contains: Optional[str] = None,
    source: Optional[str] = None,
    overwrite: bool = True,
) -> Path:
    slug = _slugify(group_label) or "group"
    path = dreams_dir / f"group-{slug}.md"
    if path.exists() and not overwrite:
        return path

    fm: Dict[str, Any] = {"type": "dream_group", "group": group_label}
    if contains:
        fm["contains"] = contains
    if source and source != "all":
        fm["source"] = source
    yaml = ["---"] + [f"{k}: {v}" for k, v in fm.items() if v not in (None, "", [])] + ["---\n"]

    group_js = group_label.replace("\\", "\\\\").replace("'", "\\'")
    dvjs = (
        "```dataviewjs\n"
        "const group = '" + group_js + "';\n"
        "const pages = dv.pages('\\\"Dreams\\\"')\n"
        "  .where(p => p.type === 'dream' && Array.isArray(p.groups) && p.groups.includes(group))\n"
        "  .sort(p => p.date ?? '', 'desc');\n"
        "dv.table(['Dream','Date','Source'], pages.map(p => [p.file.link, p.date ?? '', p.source ?? '']));\n"
        "```"
    )

    lines: List[str] = [f"# Dream Group: {group_label}"]
    meta_lines: List[str] = []
    if contains:
        meta_lines.append(f"- contains: `{contains}`")
    if source and source != "all":
        meta_lines.append(f"- source: `{source}`")
    if meta_lines:
        lines.append("\n".join(meta_lines))
    lines.append("")
    lines.append(dvjs)
    body = "\n\n".join(lines) + "\n"
    path.write_text("\n".join(yaml) + body, encoding="utf-8")
    return path


def write_models_index(models_dir: Path, models: Iterable[str], overwrite: bool = True) -> Path:
    path = models_dir / "Index.md"
    if path.exists() and not overwrite:
        return path
    list_items = "\n".join([f"- [[{_slugify(m)}|{m}]]" for m in sorted(set(models))])
    contents = f"""
---
type: models_index
---

# Models

{list_items}

""".lstrip()
    path.write_text(contents, encoding="utf-8")
    return path


def write_index(base: Path, overwrite: bool = True) -> Path:
    path = base / "Index.md"
    # Overwrite when requested so fixes propagate
    if path.exists() and not overwrite:
        return path
    contents = """
---
type: index
---

# Backrooms Transcripts Index

## By Prompt (dream_id)

```dataviewjs
// Group transcripts by prompt id and link to the corresponding prompt page
const tx = dv.pages('\"Transcripts\"').where(p => p.type === 'transcript');
const groups = tx.groupBy(p => p.dream_id ?? 'unknown').sort(g => g.rows.length, 'desc');
function promptLink(did) {
  const pr = dv.pages('\"Prompts\"').where(p => p.dream_id === did).first();
  return pr ? pr.file.link : did;
}
dv.table(['Prompt','Count'], groups.map(g => [promptLink(g.key), g.rows.length]));
```

## Latest Transcripts

```dataview
table file.link as Transcript, model_pair, dream_id, created_at, duration_sec
from "Transcripts"
where type = "transcript"
sort created_at desc
limit 200
```

## Unknown Prompts

```dataviewjs
// Transcripts with no dream_id present
const pages = dv.pages('\"Transcripts\"').where(p => p.type === 'transcript' && !p.dream_id);
dv.table(['Transcript','ModelPair','Date'], pages.map(p => [p.file.link, (p.model_pair || ''), (p.created_at || '')]));
```
""".lstrip()
    path.write_text(contents, encoding="utf-8")
    return path


def write_unknown_prompts_page(prompts_dir: Path, overwrite: bool = True) -> Path:
    path = prompts_dir / "unknown.md"
    if path.exists() and not overwrite:
        return path
    fm = {"type": "prompt_view", "dream_id": "", "title": "Unknown"}
    yaml = ["---"] + [f"{k}: {v}" for k, v in fm.items()] + ["---\n"]
    dvjs = """```dataviewjs
const pages = dv.pages('"Transcripts"')
  .where(p => p.type === 'transcript' && !p.dream_id)
  .sort(p => p.created_at, 'desc');
dv.table(['Transcript','ModelPair','Date','Exit'],
  pages.map(p => [p.file.link, (p.model_pair || ''), (p.created_at || ''), (p.exit_reason || '')])
);
```"""
    body = "# Unknown Prompts\n\n" + dvjs
    path.write_text("\n".join(yaml) + body, encoding="utf-8")
    return path


def main():
    if load_dotenv:
        try:
            load_dotenv(ROOT / ".env")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Export Supabase backrooms to Obsidian Markdown")
    ap.add_argument("--vault", default=str(ROOT / "obsidian"), help="Destination Obsidian folder (default: ./obsidian)")
    ap.add_argument("--since", type=str, default=None, help="Only export rows with created_at >= date (YYYY-MM-DD or ISO)")
    ap.add_argument("--dream-id", default="", help="Filter rows by a specific dream_id")
    ap.add_argument("--prompt-contains", default="", help="Filter rows where prompt ilike *substring*")
    ap.add_argument("--limit", type=int, default=1000, help="Fetch limit (default: 1000)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing transcript files")
    ap.add_argument("--write-index", action="store_true", help="Write Index.md and per-prompt pages")
    ap.add_argument("--export-dreams", action="store_true", help="Export matching dreams into Dreams/")
    ap.add_argument("--dream-source", default="mine", help="Dream source filter (default: mine; use 'all' to disable)")
    ap.add_argument("--dream-contains", default="", help="Filter dreams whose content ilike *substring*")
    ap.add_argument("--dream-limit", type=int, default=200, help="Dream fetch limit when exporting (default: 200)")
    ap.add_argument("--dream-group", default="", help="Assign a group label to fetched dreams and write a helper page")
    args = ap.parse_args()

    url, key = _env_keys()

    rows: List[Dict[str, Any]] = []
    since_val = iso_date(args.since) if args.since else None
    if args.limit != 0:
        rows = fetch_backrooms(
            url,
            key,
            since=since_val,
            dream_id=args.dream_id or None,
            prompt_contains=args.prompt_contains or None,
            limit=args.limit,
        )
        if not rows:
            print("No transcripts fetched for the given parameters.")
    else:
        print("Skipping transcript export (--limit 0).")

    paths = ensure_dirs(Path(args.vault))
    tx_dir = paths["transcripts"]
    prompts_dir = paths["prompts"]
    models_dir = paths["models"]

    # Write transcripts
    written = []
    for r in rows:
        p = write_markdown(tx_dir, r, overwrite=args.overwrite)
        written.append(p)

    # Optional index and per-prompt pages
    if args.write_index and rows:
        write_index(paths["base"]) 
        # Use the first prompt text we see for each dream_id
        seen: dict[str, str] = {}
        model_names: set[str] = set()
        for r in rows:
            did = str(r.get("dream_id") or "")
            if not did:
                continue
            if did not in seen:
                seen[did] = (r.get("prompt") or "").strip()
        for did, sample in seen.items():
            write_prompt_page(prompts_dir, did, sample, overwrite=True)
        # Unknown prompts page (missing dream_id)
        write_unknown_prompts_page(prompts_dir, overwrite=True)
        # Per-model pages and models index
        for r in rows:
            mods = r.get("models") or []
            for m in mods:
                if isinstance(m, str) and m:
                    model_names.add(m)
        for m in sorted(model_names):
            write_model_page(models_dir, m)
        if model_names:
            write_models_index(models_dir, model_names)

    if written:
        print(f"Exported {len(written)} transcripts to {tx_dir}")

    export_dreams = args.export_dreams or bool(args.dream_contains or args.dream_group)
    dream_written: List[Path] = []
    if export_dreams:
        dream_rows = fetch_dreams(
            url,
            key,
            source=args.dream_source or None,
            contains=args.dream_contains or None,
            limit=args.dream_limit,
        )
        if not dream_rows:
            print("No dreams fetched for the given dream filters.")
        else:
            groups = [args.dream_group] if args.dream_group else []
            for row in dream_rows:
                dp = write_dream_markdown(
                    paths["dreams"],
                    row,
                    groups=groups,
                    search_contains=args.dream_contains or None,
                    overwrite=args.overwrite,
                )
                dream_written.append(dp)
            if args.dream_group:
                write_dream_group_page(
                    paths["dreams"],
                    args.dream_group,
                    contains=args.dream_contains or None,
                    source=args.dream_source or None,
                    overwrite=True,
                )

    if dream_written:
        print(f"Exported {len(dream_written)} dreams to {paths['dreams']}")


if __name__ == "__main__":
    main()
