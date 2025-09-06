Obsidian Sync for Backrooms Transcripts

- Script: `scripts/export_obsidian.py`
- Output: Markdown notes under `./obsidian/Transcripts/` (plus optional `Prompts/` and `Index.md`)
- Requirements: `requests`, `python-dotenv` (already in `requirements.txt`)
- Env: `SUPABASE_URL`, `SUPABASE_ANON_KEY` (or `SUPABASE_SERVICE_ROLE_KEY`)

Quick start
- Put your Obsidian vault path or use the repo-local `./obsidian` folder.
- Export all transcripts:
  - `python scripts/export_obsidian.py --write-index`
- Export only new since a date:
  - `python scripts/export_obsidian.py --since 2025-09-01 --write-index`
- Filter to one dream prompt (by id):
  - `python scripts/export_obsidian.py --dream-id <uuid> --write-index`
- Filter by prompt substring:
  - `python scripts/export_obsidian.py --prompt-contains "rain-soaked museum"`

Dataview examples
- Latest transcripts:
```
table file.link as Transcript, model_pair, dream_id, created_at, duration_sec
from "Transcripts"
where type = "transcript"
sort created_at desc
limit 200
```

- All transcripts for a particular prompt (`dream_id`):
```
table file.link as Transcript, model_pair, created_at, duration_sec
from "Transcripts"
where type = "transcript" and dream_id = "<uuid>"
sort created_at desc
```

- Group by prompt to scan coverage:
```
table rows.file.link.length as Count
from "Transcripts"
where type = "transcript"
group by dream_id
sort Count desc
```

Frontmatter written per transcript
- `type: transcript`
- `dream_id: <uuid>`
- `models: [model1, model2]`, plus `model_a`, `model_b`, and order-independent `model_set`
- `model_pair: model1-model2`
- `template`, `created_at`, `duration_sec`, `max_turns`, `exit_reason`, `source_log`
- `prompt_title` (truncated) and `prompt_hash`

Per-prompt pages
- When `--write-index` is set, the script creates `obsidian/Index.md` and `obsidian/Prompts/<dream_id>.md` pages.
- Each prompt page includes:
  - All runs table
  - Grouped-by-model table (by `model_pair`)
  - Optional single-model filter via DataviewJS: set `model_filter: <model>` in the page frontmatter

Per-model pages
- The exporter also creates `obsidian/Models/<model>.md` and `obsidian/Models/Index.md`.
- Each model page lists transcripts for that model, grouped by `dream_id`, so you can compare a single model across prompts.

Prompt page filenames
- Prompt files include a short preview of the prompt and a short id for scanability, e.g.: `Prompts/rain-soaked-museum--a1b2c3d4.md`.

Tips for readability
- Use Obsidian “Readable line length”.
- Fold headings to collapse sections.
- If your transcripts contain code/tool blocks, they render best with triple backticks already in the transcript logs.
