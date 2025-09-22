# Discord Hours Playbook (setup & run checklist)

_Last updated: 2025-09-22_

Use this when we return to wire the new webhook into an actual Discord workflow. It assumes
we keep the FastAPI server from `discord_webhook_server.py` and the MCP bot (`../discordmcp`)
side-by-side.

---

## 1. Discord application prep
- Confirm the bot is in the target server: https://discord.com/oauth2/authorize?client_id=<ID>&scope=bot%20applications.commands&permissions=<mask>
- Permissions we need at minimum: `Send Messages`, `Use Slash Commands`, optionally `Embed Links`.
- In the Developer Portal, register the slash commands (global or per guild):
  - `system` (string option `prompt` required)
  - `gen` (string option `prompt` required)
  - `reset` (no options)
- Copy the `Public Key` → set as `DISCORD_PUBLIC_KEY` env var.
- Optional: set default command descriptions so collaborators understand `system` (updates Hours
  persona) vs `gen` (creates a job).

## 2. Local configuration
- `pip install -r requirements.txt`
- Decide runtime host (local tunnel via `cloudflared`/`ngrok` or deploy). For local tests:
  ```bash
  export DISCORD_PUBLIC_KEY=...  # from portal
  export DISCORD_SKIP_SIGNATURE_CHECK=1  # only if not hitting Discord yet
  uvicorn discord_webhook_server:app --host 0.0.0.0 --port 8000
  ```
- If exposing to Discord, remove the skip flag and provide a public URL (e.g., `https://mydomain/discord/interactions`).
- Persist `data/discord_sessions/state.json` somewhere durable if we deploy remotely.

## 3. MCP bot for outbound messages (optional but recommended)
- In `../discordmcp/.env`, set `DISCORD_TOKEN` to the same bot token.
- `npm install && npm run build` (already done once, re-run if deps changed).
- Launch via MCP inspector or Claude to test `send-message` works in the target server.
- Note the channel naming expectations (`general` vs channel ID) so the worker knows what to send back.

## 4. Wire webhook + bot together
- Interactions hit FastAPI → job file lands in `data/discord_jobs/`.
- Create a small watcher script tomorrow (Python or Justfile target):
  - Watch directory, load JSON, run `backrooms.py --template hours ...` with vars.
  - On success, use MCP `send-message` tool (or Discord REST) to post results referencing `channel_id` + maybe follow-up interactions.
  - Optionally delete or archive processed job files once posted.
- Keep `state.json` under `data/discord_sessions/` for per-channel history; consider syncing/backup if we deploy.

## 5. Validation checklist
- [ ] Slash commands respond inside Discord within 3s.
- [ ] Job file created with correct vars and session snapshot.
- [ ] Worker reads job and runs generation template without extra flags.
- [ ] Output posted back to same channel with mention of requester.
- [ ] `reset` clears session, reverts to default system prompt.
- [ ] MCP server continues to function for other tooling (no token conflicts).

## 6. TODO backlog for future iteration
- Accept `--template` override via `/gen` option (default stays `hours`).
- Support multi-turn memory longer than 50 entries with pruning.
- Log job outcomes (`queued`/`processing`/`posted`) for observability.
- Consider storing session/jobs in SQLite or Supabase to share across hosts.
- Add Discord follow-up message support using the saved interaction token.
- Provide a `/status` command to echo the active system prompt + recent jobs.

Bring this checklist tomorrow when we flesh out the worker + posting loop.
