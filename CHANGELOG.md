## 2025-09-10

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

