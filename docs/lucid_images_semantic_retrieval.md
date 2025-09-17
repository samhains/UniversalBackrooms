Lucid Image Posting — Semantic Retrieval

Overview
- Script: `scripts/post_lucid_images.py`
- Goal: Post N images from the Eagle DB to Discord (#lucid) from a short prompt via semantic search.
- Default: Embedding-based semantic search (OpenAI embeddings + pgvector). If fewer than N URLs are produced, the script tops off from the most recent images to reach N.

What it does
- Computes an embedding for the prompt (`text-embedding-3-small`).
- Calls the database RPC to retrieve similar images.
- Builds a direct URL from `image_url` or `storage_path`.
- Posts each image individually to Discord using the MCP `discord` server and `send-message` tool.
- Prints similarity for semantic results in the console: `sim: 0.xxx (semantic)`; top-off rows display `sim: n/a (topoff)`.

CLI usage
- Default (min similarity 0.0):
  - `python scripts/post_lucid_images.py "WORLD IN A DREAM SPIRE" --n 3`
- Options:
  - `--n`: exact number of images to post (default 3)
  - `--min-similarity`: threshold 0.0–1.0 (default 0.0)
  - `--channel`: Discord channel (default `lucid`)
  - `--folders`: optional Eagle folder filters
  - `--dry-run`: print URLs (and similarities) without posting

Environment requirements
- `.env` must include:
  - `OPENAI_API_KEY`
  - `SUPABASE_URL` and one of `SUPABASE_KEY` / `SUPABASE_ANON_KEY` / `SUPABASE_SERVICE_ROLE_KEY`

Notes
- The script uses only semantic search plus a recent-images top-off. It does not include LLM-generated SQL or FTS fallbacks.
- Ensure the `discord` MCP server is configured in `mcp.config.json` and exposes a `send-message` tool.

Future improvements
- Optional log file under `var/backrooms_logs/oneoff/` summarizing query, results, similarities, and post outcomes.
