# UniversalBackrooms
This repo replicates Andy Ayrey's "Backrooms" (https://dreams-of-an-electric-mind.webflow.io/), but it is runnable with each of Opus 3, Sonnet 3.5, GPT 4o, o1-preview, and o1-mini.

## Preliminary Findings
The models independently often talk about quantum mechanics and the simulation.

For the CLI target: Opus works as expected and o1-preview doesn't really get that it is in a conversation -- both o1s are the prompter and the repl. o1-mini doesn't seem to know how to be a CLI explorer, but can play the CLI role fine. Gpt4o runs very quickly but doesn't seem to go very deep.

I have really been enjoying sonnet's responses -- it really digs into the capture-the-flag aspects of the CLI interface, and I once saw it successfully escape the matrix and broker a lasting cooperative governance structure with the machine overlords. 

## Diffs
I changed the keyword to ^C^C instead of ^C, because many times ^C is the right message to send (e.g. after ping 8.8.8.8).
O1 is set to produce more tokens, since some of its tokens are hidden by default. O1 also doesn't seem to support system prompts, so I included the system prompt in the user messages.
I removed references to the fact that the user will be guiding the conversation in the cli prompts, because this won't always be the case and I don't want to be dishonest to the models. However, this may be causing recent Sonnet refusals.

## Recent Updates
1. Added flexibility to specify different models for LM1 and LM2 roles using command-line arguments.
2. Reorganized the file structure and variable names to clearly distinguish between the LM1 and LM2 contexts, models, and actors.
3. Introduced separate prompts for when the LM1 and LM2 models are the same or different.
4. Updated the handling of system prompts for different model types (Anthropic, GPT-4, and O1).
5. Improved logging and error handling, especially for the ^C^C termination sequence.
6. Updated the filename format to include both model names and a timestamp.
7. Implemented logging to BackroomLogs.
8. Added support for the o1-mini model.
9. Updated API key checks to only require keys for the selected models.
10. Changed the default maximum number of turns to infinity.

## Setup
- Copy .env.example to .env
- Add your Anthropic and/or OpenAI API keys to the .env file, depending on which models you plan to use. Optionally add an OpenRouter API key to try Hermes 405B.
- Install packages.  ```pip install -r requirements.txt```

## Quickstart

- Direct run (still supported):
  - `python backrooms.py` (defaults from the template)
  - `python backrooms.py --lm opus gpt4o --template roleplay`

- Config runner (recommended):
  - `just run config=configs/single_roleplay_hermes.json`
  - DreamSim3 batches from Supabase: `just dreamsim3-default`, `just dreamsim3-query`, etc.
  - DreamSim4 sequences: `just dreamsim4-cycle`, `just dreamsim4-pairs`

## Templates
Templates live under `templates/<name>/template.json` referencing Markdown files for system prompts and history. Use `--template <name>`. The number of agents must match `--lm` models. History is optional, but if all agents have empty histories the program exits with a helpful message.

## DreamSim3 Batch (from Supabase)

Run DreamSim3 over all dreams matching a query in your Supabase DB.

- Script: `scripts/dreamsim3_dataset.py`
- Logs: `BackroomsLogs/dreamsim3/*.txt` (and `BackroomsLogs/dreamsim3/dreamsim3_meta.jsonl` metadata)

Examples:

- Query “static” across all sources, using Sonnet 3 via OpenRouter, 30 turns:
  `python scripts/dreamsim3_dataset.py --query "static" --source all --limit 1000 --models sonnet3 --max-turns 30`

- Force a specific pairing (LM1 vs LM2):
  `python scripts/dreamsim3_dataset.py --query "static" --pairs gpt5:hermes --max-turns 30`

- Try all unique pairs from a set:
  `python scripts/dreamsim3_dataset.py --query "static" --models gpt5,hermes,k2 --mixed --mixed-mode=all --max-turns 30`

Troubleshooting:

- Zero results is usually a source filter: the script defaults to `--source mine`. If your rows aren’t labeled `source = 'mine'`, use `--source all`.
- Sanity-check before a batch:
  `python scripts/search_dreams.py --query "static" --limit 200 --source all`
- Model keys: `sonnet3` is routed via OpenRouter, so set `OPENROUTER_API_KEY`. For Anthropic direct, use `opus4` with `ANTHROPIC_API_KEY`.

Optional: sync accumulated runs back to Supabase (table `backrooms`):

- After runs: `python scripts/sync_backrooms.py --meta BackroomsLogs/dreamsim3/dreamsim3_meta.jsonl`
- Or set `BACKROOMS_AUTO_SYNC=1` to upsert each run as it completes.

## Logging
The script now logs conversations to folders within a main "BackroomsLogs" directory:
- BackroomsLogs/
  - OpusExplorations/
  - SonnetExplorations/
  - GPT4oExplorations/
  - O1previewExplorations/

Each log file is named with the format: `{lm1_model}_{lm2_model}_{template}_{timestamp}.txt`

## Model Information
The script includes a `MODEL_INFO` dictionary that stores information about each supported model, including its API name, display name, and company.

## Temperature
The conversation temperature is set to 1.0 for both models.

## Token Limits
- For Claude models (opus and sonnet), the max_tokens is set to 1024.
- For GPT-4 models, the max_tokens is set to 1024.
- For O1 models (o1-preview and o1-mini), the max_completion_tokens is set to 8192.

## Maximum Turns
The default number of turns is now set to infinity, allowing the conversation to continue indefinitely. You can still set a specific limit using the `--max-turns` argument.

Example to set a limit of 20 turns:
```
python backrooms.py --max-turns 20
```

## OpenRouter (Hermes 405B)
- Set `OPENROUTER_API_KEY` in `.env` (see `.env.example`).
 - Run with Hermes 405B:
```
python backrooms.py --lm hermes
```
You can also mix it with others, e.g. Hermes vs Sonnet:
```
python backrooms.py --lm hermes sonnet --template cli
```
 Or start with reasoning enabled:
 ```
 python backrooms.py --lm hermes_reasoning
 ```
Hermes 4 Reasoning Mode
- Use `--lm hermes_reasoning` to enable hybrid reasoning. This variant sends `reasoning: { enabled: true }` to OpenRouter so the model may include `<think>...</think>` traces.

## World Interface (legacy)
Older CLI-interface templates have been removed from the default set. If you need them, refer to project history.

## MCP Client (connect to existing MCP servers)

This repo includes a minimal MCP client to connect to any MCP-compliant server (e.g., your ComfyUI MCP server) over stdio.

### Batch Runner (DreamSim3)

The dataset runner `scripts/dreamsim3_dataset.py` batches DreamSim3 runs and now ensures:
- `backrooms.py` runs from the repo root so relative paths (e.g., `discord/<name>.json`, `media/<name>.json`) resolve.
- `MCP_SERVERS_CONFIG` points to the project `mcp.config.json` so MCP servers (Discord, ComfyUI, Kie.ai) load consistently.
- By default, fetched dreams are shuffled (randomized) before processing to avoid biasing toward recent rows. Use `--no-shuffle` to preserve the Supabase order, and `--seed <int>` for reproducible shuffles.

Example (Discord posts enabled):
- `python scripts/dreamsim3_dataset.py --query "static" --limit 200 --source all --models sonnet3 --max-turns 30 --discord dreamsim`

Notes on randomization
- Default behavior randomizes the order of returned dreams for both recent and search queries.
- Disable with: `--no-shuffle`
- Reproduce a specific run with: `--seed 42` (or any integer)

See `docs/dataset_runner.md` for more.

### Media Agents (MCP)

You can generate and post media via MCP servers (e.g., ComfyUI, Kie.ai).

- Configure servers in `mcp.config.json` and verify with `mcp_cli.py`.
- Select a media preset via `--media <name>` or `integrations.media` in configs.
- Multiple media presets per round are supported; they execute sequentially.
  - CLI: repeat `--media <name>` or comma-separate values.
  - Config: set `integrations.media` to a list.

Media presets (`media/<name>.json`) define:
- `tool`: `{ server, name, wrap_params, status_tool, poll, defaults{...} }`
- `model`, `system_prompt`, `mode`: t2i or edit
- `post_image_to_discord`: attach image in Discord posts (default true)
- `discord_channel` / `discord_server`: per-media override for where image posts go

Examples:
- `media/kieai.json` routes to the Kie.ai MCP tool; `media/cli.json` to a ComfyUI-like tool.
- To avoid image attachment but still post summaries, set `post_image_to_discord: false` (see `media/kieai_no_discord.json`).

Discord profiles
- You can enable multiple Discord profiles at once; each posts a text update to its configured channel/server.
- CLI: repeat `--discord <profile>` or comma-separate; Config: set `integrations.discord` to a list.

- File: `mcp_client.py` (library)
- CLI: `mcp_cli.py`
- Example config: `mcp_servers.example.json` — copy to `mcp_servers.json` and update.

### Install deps

```
pip install -r requirements.txt
```

### Configure servers

Copy one of the examples and edit for your server command. The CLI supports two formats:

```
cp mcp_servers.example.json mcp_servers.json
```

Format A — list form (file: `mcp_servers.json`):

```
{
  "servers": [
    {
      "name": "comfyui",
      "command": "node",
      "args": ["path/to/comfyui-mcp-server.js"],
      "env": { "NODE_ENV": "production" }
    }
  ]
}
```

Format B — map form (file: `mcp.config.json`) matching prior setups:

```
{
  "mcpServers": {
    "comfyui": {
      "type": "stdio",
      "command": "python",
      "args": ["/Users/you/Code/comfyui-mcp-server/server.py"],
      "env": {}
    }
  }
}
```

### List tools

Using list form config:

```
python mcp_cli.py --config mcp_servers.json --server comfyui list-tools
```

Using map form config:

```
python mcp_cli.py --config mcp.config.json --server comfyui list-tools
```

### Call a tool

```
# Many FastMCP servers expect a single argument named `params`.
# Use --wrap-params to automatically wrap your JSON as {"params": <json>}.
python mcp_cli.py --config mcp.config.json --server comfyui call-tool generate_image --wrap-params --json '{"prompt": "a serene sunset over mountains", "width": 1024, "height": 768}'
```

Note: Shells treat newlines as command separators; keep the command on one line or use a backslash to continue lines.

### Notes

- The client uses stdio transport and performs the standard MCP handshake (`initialize`, then `tools/list` and `tools/call`).
- Config files must be valid JSON (no trailing commas or comments).
- If you prefer not to use a config file, you can pass `--cmd` and `--args` directly:

```
python mcp_cli.py --cmd "node" --args path/to/server.js -- list-tools
```

- Ensure your MCP server binary/script is on `PATH` or provide an absolute path.
- If your server speaks MCP over a different transport (WebSocket/SSE), this minimal client won’t connect as-is; stdio is the most common and simplest.

## Media Agent (optional, explicit opt‑in)

You can opt‑in to a “media agent” that reads each round of conversation and generates an image via your MCP server. It runs once after both actors respond in a round and logs the result to the same Backrooms log file.

- Enable by passing `--media <preset>` to `backrooms.py`.
- Preset file is resolved from `media/<preset>.json` or `templates/<template>/media.json`.
- Configure your MCP server in `mcp.config.json`.

Example preset fields:

- "model": which LLM to use for crafting the image prompt (e.g., "same-as-lm1", "opus", "sonnet", "gpt4o").
- "system_prompt": system prompt for the media agent
- "tool": MCP tool config with "server" and tool "name" (both required). You may also add `defaults` for any tool arguments.
- "mode": `"t2i"` for text-to-image or `"edit"` for iterative image editing
- "t2i_use_summary": optional boolean to base t2i prompts on a short running summary plus the latest round (default false)

The agent:
- Builds a short prompt based on the current round’s messages.
- Calls your selected LLM to produce a concise image prompt.
- Calls the MCP tool (e.g., ComfyUI `generate_image`) with that text and logs the result.

Notes:
- The agent uses `MCP_CONFIG` env var if set, otherwise `mcp.config.json` in the project root.
- No implicit fallbacks: if `--media` is omitted, no media agent runs. The preset must declare a `tool.server` and `tool.name`. The media model must be resolvable; if `same-as-lm1` cannot be inferred, set `model` explicitly.

### Modes

- t2i (default):
  - Generates a concise text-to-image prompt from the latest round.
  - Optionally, set `"t2i_use_summary": true` to incorporate a short running conversation summary.
  - Example config:
    {
      "model": "sonnet",
      "mode": "t2i",
      "tool": { "server": "comfyui", "name": "generate_image", "defaults": {"width": 768, "height": 768} }
    }

- edit (iterative updates):
  - Maintains a short conversation summary per run and the last generated image reference.
  - Produces an edit instruction that updates the image to reflect the latest round while staying true to the conversation’s overall essence.
  - The prompt includes a `BASE_IMAGE: <url>` line when available. Ensure your MCP server/workflow (e.g., using Qwen Edit) understands this convention and applies edits based on the provided base image and instruction.
  - Example config:
    {
      "model": "sonnet",
      "mode": "edit",
      "system_prompt": "You are an image edit director…",
      "tool": { "server": "comfyui", "name": "generate_image", "defaults": {"width": 768, "height": 768} }
    }

State files: The agent saves per-run state (last image reference and a short conversation summary) alongside the log file as `<logfile>.media_state.json`.

### Add Tools To Backrooms

You do not add tools inside this repo; you expose them via your MCP server, then reference them in the media-agent config.

- Discover tools: `python mcp_cli.py --config mcp.config.json --server comfyui list-tools`
- Choose a tool name from the list (e.g., `generate_image`, `remix_image`).
- Update `templates/<template>.media.json`:
  - Set `"tool": { "server": "comfyui", "name": "<your-tool>", "defaults": { ... } }`
  - Optionally set `"mode": "t2i"` or `"edit"` depending on desired behavior.
  - Choose the LLM for the agent (e.g., `"model": "sonnet"` to use Sonnet 3.5).
- The media agent automatically calls the MCP tool once after each round and logs results.

Notes:
- FastMCP-style tools (e.g., your server) expect arguments under a single `params` field; the media agent wraps calls for compatibility automatically.
- No local tool schema is required. If you want client-side schema validation before calling the tool, we can add it as an optional step.

**Quick Start**
- Install: `pip install -r requirements.txt`
- Configure MCP: create `mcp.config.json` with your ComfyUI server command.
- Optional media: create a preset (e.g., `media/cli.json`) and run with `--media cli`.
- Run: `python backrooms.py --lm opus opus --template cli [--media <preset>]`

**mcp.config.json**
- "mcpServers": map of server configs by name.
- "type": must be `stdio` (supported transport).
- "command": executable to run (e.g., `python`).
- "args": array of arguments (e.g., path to your server script).
- "env": optional environment variables for the server process.

Example:
{
  "mcpServers": {
    "comfyui": {
      "type": "stdio",
      "command": "python",
      "args": ["/abs/path/to/comfyui-mcp-server/server.py"],
      "env": {}
    }
  }
}

**Media Preset Config**
- Files: `media/<preset>.json` or `templates/<template>/media.json`
- "model": `same-as-lm1` or a key from `MODEL_INFO` (e.g., `opus`, `sonnet`, `gpt4o`). If `same-as-lm1` cannot be resolved, set a concrete model.
- "system_prompt": system prompt for generating the image prompt text.
- "tool.server": MCP server key from `mcp.config.json` (required).
- "tool.name": MCP tool to call (required).
- "tool.defaults": optional args merged into each call (e.g., width/height).

Example:
{
  "model": "same-as-lm1",
  "system_prompt": "You are a visual director…",
  "tool": {
    "server": "comfyui",
    "name": "generate_image",
    "defaults": { "width": 768, "height": 768 }
  }
}

**Execution Model**
- The media agent runs once per round after both actors reply.
- It builds a short textual prompt from the current round only.
- It calls the selected LLM to refine that into a concise image prompt.
- It calls the MCP tool with `{ prompt, ...defaults }`.
- It logs the prompt and raw tool result under “Media Agent”.

**Best Practices**
- Keep prompts concise (<200 chars) and concrete; avoid meta language.
- Use `same-as-lm1` to reuse existing keys; override per template when needed.
- Start with moderate sizes (e.g., 768x768) and tune later.
- Keep the agent stateless (per-round) to minimize latency and coupling.

**Troubleshooting**
- Missing MCP dependency: ensure `pip install -r requirements.txt` (includes `mcp`).
- Invalid JSON: remove trailing commas/comments from config files.
- Server not found: verify `command` is on PATH or use absolute paths.
- Transport mismatch: only `stdio` is supported by this client.
- Tool name mismatch: run `python mcp_cli.py --config mcp.config.json --server comfyui list-tools` to confirm tool names.
## Discord Posting

The project can optionally post round-by-round updates to a Discord channel using an MCP-compatible Discord server. Profiles live under `./discord/*.json` and can be selected with `--discord <profile>`.

- Set `enabled: true` in the chosen profile to turn it on.
- The tool defaults control where messages go. Example:

```json
{
  "tool": {
    "server": "discord",
    "name": "send-message",
    "defaults": { "channel": "backrooms" }
  }
}
```

### Transcript Posting

You can also mirror the conversation verbatim to a separate channel (e.g., `#transcripts`). Configure this per run in your config file under `integrations` (recommended; presets ignore transcript toggles):

```json
{
  "integrations": {
    "discord": ["status_feed"],
    "post_transcript": false,
    "transcript_channel": "transcripts"
  }
}
```

- Optional: provide a separate tool config for transcript posts if you want a different MCP server or tool:

```json
{
  "transcript_tool": {
    "server": "discord",
    "name": "send-message",
    "defaults": { "channel": "transcripts" }
  }
}
```

When enabled, after each round the agent posts the normal summary to the main channel and also posts the verbatim round transcript to the transcript channel. Very long messages are split into multiple parts to respect Discord length limits.

### Bot Memory (per-run)

Discord profiles can optionally include their own prior posts from the current run as context for better continuity. This history is kept in memory per profile for the duration of a run and is only included when referenced explicitly in the template.

- Template-controlled only: No hidden prompt injection occurs. Use `{bot_history}` in your `user_template` to include a bullet list of the bot’s prior posts (oldest to newest).

- Optional window control (profile JSON):

```json
{
  "bot_history_window": 5
}
```

- Example placement inside a profile template:

```
"user_template": "Your prior posts (oldest to newest):\n{bot_history}\n\nLatest round:\n{latest_round_transcript}"
```

### Included Presets

- `chronicle`: third‑person, concise atmospheric updates.
- `chronicle_fp`: first‑person field notes from inside the world.
- `cctv`: terse surveillance‑style captions.
- `dreamsim`: terminal‑style status logs of current simulation state.
- `narrative_terminal`: terminal‑style, cohesive narrative arc of events.

Use a preset by passing its filename stem:

`python backrooms.py --lm sonnet3 sonnet3 --template dreamsim3 --discord narrative_terminal`

## Template Variables

Override variables without editing `vars.json`:

- `--var NAME=VALUE` (repeatable): sets template variables before formatting.
- `--query "text"`: convenience alias that sets both `QUERY` and `DREAM_TEXT`.

Examples:

- `python backrooms.py --lm k2 k2 --template dreamsim3 --query "A dim hallway that hums like a refrigerator"`
- `python backrooms.py --lm gpt5 hermes --template roleplay --var TOPIC=alchemy --var TONE=serious`

In batch configs, map per-item fields to variables using `template_vars_from_item` (defaults to `DREAM_TEXT <- content`).

## Config Runner (experimental)

Define runs in JSON under `configs/` and execute them with a single command:

- Single run: `just run config=configs/single_roleplay_hermes.json`
- Batch from Supabase (DreamSim3): `just run config=configs/batch_dreamsim3_query_kie.json`

Key fields:
- `type`: `single` or `batch`
- `template`: Backrooms template (e.g., `dreamsim3`, `roleplay`)
- `models`/`pairs`/`mixed`: choose model selection plan
- `integrations`: optional `{ "discord": name, "media": name }`
- `data_source`: for batch (currently `{ kind: "supabase", query, limit, source }`)
- `template_vars_from_item`: mapping for per-item variables (default `DREAM_TEXT <- content`)
- `output.meta_jsonl`: metadata JSONL path for batch runs

See `configs/` and `scripts/run_config.py` for examples and details.
