## 2025-09-10

- Discord presets
  - Added `discord/transcripts.json` preset for transcript-only posting. No summary; posts each round's verbatim entries to a transcripts channel.
  - Discord agent now supports `disable_summary` in a profile to skip summary posting (used by the transcripts preset).
  - Docs updated to recommend using the `transcripts` preset instead of setting transcript toggles in two places; config-level overrides remain supported.

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
