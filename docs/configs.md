# Configs Guide

This doc summarizes the JSON config formats used by `scripts/run_config.py` and the related integrations. Use files under `configs/` as working examples.

## Top-Level Shape

- `type`: `"single"` | `"batch"`
- `template`: Template name under `templates/` (e.g., `dreamsim3`, `roleplay`).
- `models` / `pairs` / `mixed`: Choose how models are selected for each run (see below).
- `max_turns`: Integer, default `30`.
- `max_context_frac`: Float, default `0.0` (disabled). When > 0, budget the context and optionally set `context_window`.
- `context_window`: Integer, default `128000`. Used with `max_context_frac`.
- `integrations`: Optional integrations to run per-round.
  - `discord`: Profile name (string) or list of profile names under `discord/`.
  - `media`: Preset name (string) or list under `media/`.
- `output` (batch only):
  - `meta_jsonl`: Path for metadata, default `var/backrooms_logs/<template>/<template>_meta.jsonl`.

## Model Selection

- `models`: list of aliases (e.g., `["sonnet3", "gpt5"]`). Runs self-dialogue per model if no `pairs`/`mixed`.
- `pairs`: list of strings `"modelA:modelB"` (also supports `+` or `x` separators). Runs each explicit pair.
- `mixed`: random or all pairs from a base set of `models`.
  - `mode`: `"all"` (unique index pairs once) | `"random"` (sample per-item).
  - `runs_per_item`: integer, used when `mode = "random"`.

Validation for model aliases uses `model_config.py`. Unknown aliases will error.

## Single Mode (`type = "single"`)

- `vars`: object map of template variables to values.
- `query`: shorthand that sets both `QUERY` and `DREAM_TEXT`.

Example:

```json
{
  "type": "single",
  "template": "roleplay",
  "models": ["hermes"],
  "integrations": { "discord": "chronicle_fp", "media": "cli" },
  "vars": { "TOPIC": "alchemy" }
}
```

## Batch Mode (`type = "batch"`)

- `data_source`:
  - `kind`: `"supabase"` | `"none"` | `"local"` | `"static"`. Use `"supabase"` to fetch items from your DB.
  - `query`: string. When set and `shuffle` is `null`/unspecified, rows are shuffled by default.
  - `limit`: integer, number of rows to fetch.
  - `source`: `"mine"` | `"rsos"` | `"all"` (optional filter in Supabase queries).
- `template_vars_from_item`: map template var -> item field (default `{ "DREAM_TEXT": "content" }`).
- `shuffle`: `true` | `false` | `null`. If `null` or absent and `query` is present, shuffles by default.
- `seed`: integer for deterministic shuffle.
- `max_items`: integer, truncate list after shuffle.
- `auto_sync`: boolean. When `true`, upserts per-run metadata to Supabase using `scripts/sync_backrooms.py`.

Sequence-only batch (no `supabase`) uses:

```json
{
  "type": "batch",
  "template": "dreamsim3",
  "data_source": { "kind": "none" },
  "models": ["sonnet3", "gpt5"],
  "sequence": { "runs": 30 }
}
```

## Integrations

### Discord (text)

Provide a profile name under `discord/` via `integrations.discord`. Each profile:

- `model`: API model preference, e.g., `"same-as-lm1"` or a known alias.
- `system_prompt`: System prompt for the summarizer.
- `user_template`: May reference:
  - `{latest_round_transcript}`: last round as "- actor: text" lines.
  - `{transcript}`: recent transcript window as "- actor: text" lines.
  - `{last_actor}`, `{last_text}`: convenience values.
- `transcript_window`: integer; number of recent messages to include in `{transcript}` (0 disables).
- `tool`: where to post the summary
  - `server`: usually `"discord"`
  - `name`: usually `"send-message"`
  - `defaults.channel`: the channel name (e.g., `"backrooms"`)
- `post_transcript`: boolean; if true, also post a verbatim transcript of the last round. Default: `false`.
- `transcript_channel`: channel for transcripts; falls back to `tool.defaults.channel`.

Notes:
- Empty string `"server": ""` is ignored. Omit it or set a valid guild/server name if your MCP server expects it.
- Transcripts are posted per-entry (one Discord message per actor’s message) to maximize readability.

Transcript posting (recommended):
- Use a dedicated preset `transcripts` so you configure it once. Example usage in a config:

```json
{
  "integrations": { "discord": ["status_feed", "transcripts"] }
}
```

Advanced (optional/back‑compat): You may override transcript behavior per run using `integrations.post_transcript`, `integrations.transcript_channel`, or `integrations.transcript_tool`, or pass arbitrary fields via `integrations.discord_overrides`. The preset approach is simpler and avoids duplication.

Backwards compatibility:
- `{latest_round_bullets}` and `{context_bullets}` are still supported but deprecated in favor of `{latest_round_transcript}` and `{transcript}`.

### Media (images)

Provide a preset under `media/` via `integrations.media`. Each preset defines:

- `model`: generation model alias
- `system_prompt`: image prompt generator’s system prompt
- `mode`: e.g., `"t2i"`
- Discord posting:
  - `post_image_to_discord`: boolean
  - `discord_channel`: channel name
  - `post_caption_to_discord`: boolean (default true). When false, posts images without a text caption.

## Environment

For Supabase-backed batches:
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY` or `SUPABASE_SERVICE_ROLE_KEY` (fallback `SUPABASE_KEY` supported)

For Discord MCP server, configure your bot token and any server/guild details in your MCP config file (e.g., `mcp.config.json`).
