## Changelog

### 2025-09-12 (later)
- scripts/post_lucid_images.py: switch to embedding-first retrieval (OpenAI embeddings + pgvector) with automatic fallback to deterministic FTS+trigram; LLM path only via `--use-llm`.
- Add CLI options: `--semantic`, `--vector-column`, `--embed-model`; prefer semantic when `OPENAI_API_KEY` present.
- Add robust Discord posting helper: sanitizes image URLs, posts via `mediaUrl`/`imageUrl`/`attachments`, and falls back to plain text with URL.
- Improve result parsing to accept `imageUrl`, `image_url`, or `imageurl` from Supabase MCP responses.
- Log executed SQL for semantic and FTS paths in the per-run log for traceability.
- Add top-off after non-LLM path posting to ensure exactly N images get posted (fetch latest images via REST, excluding already-posted URLs).
- Known issue: in some runs only 1 image is posted despite N>1; investigating whether the Discord MCP tool ignores additional attachments/messages or our SQL yields < N rows. Logging added to verify counts and SQL used.
  - Added batch post attempt (single message with multiple attachments) before falling back to per-image posts. This may resolve Discord MCP limitations around rapid sequential posts.

### 2025-09-12
- scripts/post_lucid_images.py: add `--n` option to request an exact number of images; `--top-k` is deprecated and mapped to `--n`.
- Ensure exactly N images are posted: if the media agent returns fewer than N, top off via Supabase REST with the latest images (deduped).
- Update system prompt dynamically to reflect N (replaces LIMIT 3/THREE with LIMIT N and adds explicit instruction).
- REST fallback limits now respect `--n` rather than a fixed 3.
