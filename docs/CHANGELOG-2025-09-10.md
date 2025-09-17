Backrooms Media + Kie.ai Integration – 2025-09-10

Summary
- Fixed Kie.ai edit posts “freezing” by ensuring status polling returns the edited image URL and not the echoed input URL. Discord now reliably receives edit images in the logs channel.

Changes
- Media Agent
  - Ignore echoed input URL in edit mode; keep polling for the real edited URL.
  - Broaden status parsing to include: outputUrl(s), resultUrl(s), arrays of dicts, and nested api_response/local_task shapes.
  - Final regex fallback to capture any image URL present in status text.
  - Status polling sends both task_id and taskId; optional params wrapper supported for status tool.
  - Increased status heartbeat length in logs for better forensics.

- Presets / Config
  - media/kieai_edit_chain.json: set discord_channel to logs; enable placeholders for robustness; tuned poll settings.
  - mcp.config.json (kie-ai-mcp-server): added KIE_AI_DB_PATH to a shared SQLite file to persist tasks across calls.
  - .env: added KIE_AI_DB_PATH for consistency when running the MCP server manually.

- Logging
  - var/backrooms_logs updated with latest runs confirming edit posting.

- Discord
  - Added `pre_message` banner/separator to `discord/dream_header.json` and profile handling so a header can post before any content each dream. Uses the same channel/server defaults and respects chunking limits.

Notes
- Kie MCP server relies on a persistent SQLite DB for task tracking. Using KIE_AI_DB_PATH ensures get_task_status can observe completion within ~5–20s.
- If desired, we can add an early_post_after_seconds option to post best-available media after a fixed time.
