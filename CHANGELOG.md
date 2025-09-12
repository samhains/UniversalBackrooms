## 2025-09-10

- Discord presets
  - Added `discord/transcripts.json` preset for transcript-only posting. No summary; posts each round's verbatim entries to a transcripts channel.
  - Discord agent now supports `disable_summary` in a profile to skip summary posting (used by the transcripts preset).
  - Docs updated to recommend using the `transcripts` preset instead of setting transcript toggles in two places; config-level overrides remain supported.

- Dream header + posting flow
  - Added `discord/dream_header.json`: one-shot, maximalist ASCII banner posted at the start of each dream (infers short title + embeds 2–3 poetic lines). Defaults to `#logs` channel.
  - `backrooms.py`: supports `post_once_at_start: true` and `run_on: "first"|"start"` in Discord profiles to post only in the first round; logs loaded Discord profiles and warns when a preset is missing.
  - `configs/posting_dreams.json`: enabled `dream_header` alongside `simple_director` and `transcripts` so each dream opens with the header.
  - Discord agent: added `pre_message` support in profiles/presets to post a banner/separator before any content (useful between dreams). Text is chunked to respect Discord limits and posted with the same channel/server defaults.

- Media (ComfyUI edit chain)
  - `configs/posting_dreams.json`: added `"media": "comfyui_edit_chain"` to pair the ComfyUI chain (initial T2I then iterative edits) with posting runs.
  - Uses existing `media/comfyui_edit_chain.json` which posts generated/edited images to the `#media` channel via its `discord_tool` defaults.

- Media (Kie.ai)
  - `media/kieai_edit_chain_logonly.json`: disabled `discord_dry_run` and set the Discord channel to `logs` so images are actually posted to `#logs`.
  - No new config required; existing `configs/batch_dreamsim3_query_kie_chain.json` continues to work and now posts images to `#logs`.

- Docs
  - README and `docs/configs.md` updated: transcript preset usage, included presets list, and clarified per-run overrides.

## 2025-09-09

- Kie.ai MCP integration and Discord improvements (branch: feature/discord-transcripts):
  - Media Agent handles async Kie.ai tasks by polling `get_task_status` until an image URL is available; enriches results with a standard image content element and logs concise heartbeats.
  - Batch dataset runner executes `backrooms.py` from the repo root and sets `MCP_SERVERS_CONFIG` so MCP servers (Discord, ComfyUI, Kie.ai) load reliably.
  - Added Kie.ai media preset (`media/kieai.json`) with polling config, and documentation (`docs/kie_ai_mcp.md`).
  - Documentation updates: `docs/dataset_runner.md`, README notes for batch usage and MCP servers.
- MCP client/CLI enhancements:
  - `mcp_client.py` supports `cwd` in server config; `mcp_cli.py` reads optional `cwd`.
  - Media Agent supports `wrap_params` for servers that expect either direct args or `{params: ...}` envelopes.

- Added config-based runner (`scripts/run_config.py`) supporting:
  - Single runs mapping to `backrooms.py`
  - Batch runs (DreamSim3 via Supabase, DreamSim4 sequences) with metadata JSONL
  - Reused helpers for Supabase fetch, pairing, logs, and sync
- Added multiple example configs under `configs/` for DreamSim3 and DreamSim4
- Media system:
  - Support multiple media presets per round (sequential execution)
  - `post_image_to_discord` toggle in media presets
  - Per-media Discord channel/server overrides via `discord_channel` / `discord_server`
  - Added `media/kieai_no_discord.json` example
- Discord agent allows per-call channel/server override (used by media posting)
- Refactor: extracted shared batch helpers to `scripts/batch_utils.py`
- Refactor: made `scripts/dreamsim3_dataset.py` use shared helpers
- Justfile cleanup: added config-based presets and a `run` target
- README: updated with Quickstart, config runner usage, and media agent improvements
## 2025-09-12

- K2 posting presets
  - Added `configs/posting_dreams_k2.json`: identical to `configs/posting_dreams.json` but forced to K2 for all runs.
  - Added `configs/posting_dreams_k2_ascii.json`: K2-only batch config using Discord presets `dream_header`, `ascii_director`, and `transcripts`.

- Discord ASCII director
  - Added `discord/ascii_director.json`: an ASCII-art-focused variant of the simple director that outputs a compact `prompt:` line followed by a readable ASCII panel (single fenced code block, clear layout, no meta/emojis).

- Media (Eagle DB top 3)
  - Added `media/eagle_top3.json`: prompts a single SQL statement (via K2) to fetch up to 3 images from `eagle_images` and returns `imageUrl` fields; posts all results to Discord.
  - Enabled the preset in posting configs: `configs/posting_dreams.json`, `configs/posting_dreams_k2.json`, and `configs/posting_dreams_k2_ascii.json` via `"media": "eagle_top3"`.
  - Routed image posts to `#lucid` using `integrations.media_overrides: { "discord_channel": "lucid" }` so only media goes to that channel; text updates remain unchanged.

- Media agent improvements
  - Supports posting multiple media URLs from a single tool result using `post_top_k` in media presets.
  - Extracts multiple URLs robustly from common MCP result shapes (content arrays, response.data, local_task, etc.).
  - Maintains duplicate-post protection for single-image results and preserves dry-run behavior.

- MCP servers
  - Added a `cv-art` server entry to `mcp.config.json` (points to `../cv-art-mcp`); no runtime coupling yet, prepared for future captioning/vibe prompts per image.

- Discord ASCII director
  - `discord/ascii_director.json`: removed leading "prompt:" text. Now emits only a single fenced code block with ASCII art (no prose outside the block).

- One-off Eagle → Discord poster
  - Added `scripts/post_lucid_images.py`: a simple CLI to take a one-off prompt (e.g., "LAKE DOOR THRESHOLD"), retrieve up to 3 images from the `eagle_images` DB via the `media/eagle_top3.json` preset (SQL crafted by the selected LLM, default Opus 4), and post them to Discord (default channel `#lucid`).
  - Usage: `python scripts/post_lucid_images.py "LAKE DOOR THRESHOLD" --channel lucid --model opus4 --caption "optional text" --config eagle_top3`
  - Writes a run log to `BackroomsLogs/oneoff/` and respects MCP server configuration from `mcp.config.json`.
