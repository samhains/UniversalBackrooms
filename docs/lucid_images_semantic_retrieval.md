Lucid Image Posting — Semantic Retrieval Refactor (Status)

Overview
- Script: `scripts/post_lucid_images.py`
- Goal: Post N images from the Eagle DB to Discord (#lucid) based on a short, poetic prompt without relying on LLMs.
- New default: Embedding-based semantic search (OpenAI embeddings + pgvector). Falls back to deterministic FTS+trigram when needed. LLM-driven SQL is now optional via `--use-llm`.

What changed
- Embedding-first retrieval path:
  - Computes an embedding for the prompt (OpenAI `text-embedding-3-small` by default).
  - Executes a single SQL over `eagle_images` ordering by vector distance: `ORDER BY e.<vector-column> <-> '[...]'::vector ASC, created_at DESC LIMIT N`.
  - Configurable vector column via `--vector-column` (default `caption_embedding`).
- Deterministic fallback (no LLM):
  - Fused FTS + trigram + small recency, no WHERE, always returns rows.
  - Used automatically when embeddings aren’t available or semantic path fails.
- Discord posting made robust:
  - Sanitizes URLs to a single image link before posting.
  - Uses `mediaUrl`, `imageUrl`, and `attachments` keys; falls back to plain text if attachments fail.
  - Logs each post attempt into the run log.
- Result parsing more tolerant:
  - Accepts `imageUrl`, `image_url`, or `imageurl` fields from MCP Supabase rows.
- Top‑off to exactly N images:
  - If fewer than N images are found/posted, fetches the latest images via Supabase REST (deduped) and posts only the extras.
- Logging and traceability:
  - Records the executed SQL for semantic and FTS paths in `BackroomsLogs/oneoff/*.txt`.
  - Records each Discord post attempt and fallback.

CLI usage
- Default (semantic preferred when OPENAI_API_KEY is set):
  - `python scripts/post_lucid_images.py "WORLD IN A DREAM SPIRE" --n 3`
- Explicitly force semantic:
  - `--semantic`
- Specify the pgvector column name (if not `caption_embedding`):
  - `--vector-column clip_text_embedding`
- Choose embedding model:
  - `--embed-model text-embedding-3-large` (if your DB vector dimension matches)
- No LLM / deterministic only:
  - `--no-llm`
- LLM-driven SQL (not recommended):
  - `--use-llm --model haiku3`

Environment requirements
- `.env` must include:
  - `OPENAI_API_KEY` (for semantic embeddings; optional if using only FTS)
  - `SUPABASE_URL` and either `SUPABASE_ANON_KEY` or `SUPABASE_SERVICE_ROLE_KEY`
- DB requirements:
  - `pgvector` installed and a vector column on `eagle_images` with dimension matching the embedding model used (default 1536 for `text-embedding-3-small`).
  - If the column name differs, pass via `--vector-column`.

Known issues (as of 2025‑09‑12)
- “Only one image posted” in some runs despite `--n > 1`:
  - Possible causes: SQL yields < N rows; Discord MCP tool refuses subsequent posts; or our REST top‑off was skipped due to missing Supabase key.
  - Mitigation added: After initial posting, fetch latest images via REST to top off; robust Discord posting with fallbacks; log executed SQL and URL counts.
- If the pgvector column is the wrong dimension for the selected embedding model, the semantic query will error and the script will silently fall back to FTS. Check the run log for the “SQL (semantic)” section or errors.

How to debug quickly
1) Run the command and open the log under `BackroomsLogs/oneoff/` for that timestamp.
2) Look for:
   - `### SQL (semantic) ###` or `### SQL (fts+trgm, …) ###` — this shows the exact SQL.
   - A `URLs: N` line after semantic SQL.
   - One or more `### Discord Post ###` sections.
   - If fewer than N were posted, look for a second set of `Discord Post` blocks from the top‑off.
3) If semantic section is missing, check `OPENAI_API_KEY` is set.
4) If top‑off didn’t happen, verify `SUPABASE_URL` and one of the keys (`SUPABASE_ANON_KEY` or `SUPABASE_SERVICE_ROLE_KEY`).

Next steps / TODO
- Auto-detect available vector columns on `eagle_images` (e.g., try `caption_embedding`, `clip_text_embedding`, `openai_text_embedding`) by introspecting `information_schema`.
- Add a dimension check by inspecting `pg_typeof(<col>)` or a `limit 1` select to confirm compatibility with the chosen embedding model.
- Optionally support local embedding models to remove external dependency.
- Consider batching Discord posts into one message with multiple attachments if the MCP server supports it; otherwise stick to one post per image.

