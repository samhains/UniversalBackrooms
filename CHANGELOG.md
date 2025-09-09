## 2025-09-10

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
