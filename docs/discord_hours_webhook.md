# Discord Hours Webhook (experimental)

This branch adds a minimal HTTP server that can receive Discord interaction webhooks and
turn them into prompt updates for the new `hours` template.

## Features

- `/system` — Update the active system prompt for the channel. Clears stored history.
- `/gen` — Capture a scene prompt and append it to the channel history. Each call creates a
  JSON job in `data/discord_jobs/` with the ready-to-run template variables.
- `/reset` — Drop the stored session and fall back to the default system prompt.

The server keeps per-channel state in `data/discord_sessions/state.json` so restart safety is
covered without a database. Jobs capture a snapshot of the system prompt, history, and the
current prompt so an offline worker can run the actual generation when convenient.

## 1. Discord application setup

1. Create (or reuse) a Discord application and bot user.
2. In the *Interactions Endpoint URL* field, point to your running server (e.g.,
   `https://example.com/discord/interactions`).
3. Register three **Slash Commands** for the application:
   - `system` with one required string option named `prompt`.
   - `gen` with one required string option named `prompt` (or `scene`).
   - `reset` with no options.
   Discord slash commands must be lower-case without punctuation; you can describe them in the
   command description as `!system`/`!gen` if you prefer that format in documentation.
4. Copy the application public key and set it locally as `DISCORD_PUBLIC_KEY`.

## 2. Running the webhook server

```bash
pip install -r requirements.txt
export DISCORD_PUBLIC_KEY="<your-public-key>"
uvicorn discord_webhook_server:app --host 0.0.0.0 --port 8000
```

During local development you can bypass signature checks by exporting
`DISCORD_SKIP_SIGNATURE_CHECK=1`. **Do not use that flag in production.**

The server exposes:

- `POST /discord/interactions` — Discord callback endpoint.
- `GET /discord/sessions` — Debug snapshot of stored channel state.
- `GET /healthz` — Simple health probe.

## 3. Consuming queued jobs

Each `/gen` call writes a file like `data/discord_jobs/1695323456789-ab12cd34.json`. The schema:

```json
{
  "template": "hours",
  "channel_id": "12345",
  "vars": {
    "HOURS_SYSTEM_PROMPT": "…",
    "HOURS_SCENE_PROMPT": "Las Vegas Graffiti"
  },
  "session": {
    "history": ["…"]
  }
}
```

You can point the main runner at the new template with the saved variables:

```bash
python backrooms.py \
  --template hours \
  --lm opus opus \
  --var HOURS_SYSTEM_PROMPT="$(jq -r '.vars.HOURS_SYSTEM_PROMPT' job.json)" \
  --var HOURS_SCENE_PROMPT="$(jq -r '.vars.HOURS_SCENE_PROMPT' job.json)"
```

(Replace `job.json` with the actual queued file. The job file also includes the interaction ID/token
in case you want to post results back to Discord later.)

## 4. Template assets

The `templates/hours/` directory contains the operator/collaborator system prompts used when
processing queued jobs. Defaults live in `templates/hours/vars.json`; the webhook overrides those
values dynamically per `/system` and `/gen` call.

## Next steps

- Build a worker that watches `data/discord_jobs/` and kicks off `backrooms.py` runs.
- Post completion messages or generated media back to Discord using the stored interaction token.
- Extend the history persistence if you want longer conversation memory or multi-channel contexts.
