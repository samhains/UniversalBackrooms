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
- Add your Anthropic and/or OpenAI API keys to the .env file, depending on which models you plan to use.
- Install packages.  ```pip install -r requirements.txt```

## To Run
For a default conversation using Opus for both roles:
```
python backrooms.py
```

For a conversation between different models:
```
python backrooms.py --lm opus gpt4o
```

You can mix and match any combination of models for the LM roles:
- opus
- sonnet
- gpt4o
- o1-preview
- o1-mini

If you don't specify models, it defaults to using two Opus models. You can specify as many models as you want for n-way conversations, as long as your chosen template supports it.

## Templates
Templates are JSON specs pointing to Markdown files for reusable prompts and per‑agent chat history.

- Spec: `templates/<name>.json`
- Prompts: `prompts/.../*.md` (Markdown)
- Chat history: `chat_history/.../*.md` (Markdown)

Pick a template with `--template <name>`. The CLI auto-discovers available templates from `templates/*.json`.

Notes:
- A template’s `agents` list must match the number of `--lm` models.
- History files are optional; an empty file means no initial history. If all agents have empty history, the program exits with a helpful message.

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

## World Interface (legacy)
Older CLI-interface templates have been removed from the default set. If you need them, refer to project history.

## MCP Client (connect to existing MCP servers)

This repo includes a minimal MCP client to connect to any MCP-compliant server (e.g., your ComfyUI MCP server) over stdio.

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

## Media Agent (optional image generation per round)

You can enable a simple “media agent” that reads each round of conversation and generates an image via your MCP ComfyUI server. It runs once after both actors respond in a round and logs the result to the same Backrooms log file.

- Config file: `templates/<template>.media.json`
- Example provided: `templates/cli.media.json`
- Enable by setting `"enabled": true` and configuring your MCP server in `mcp.config.json`.

Example `templates/cli.media.json` fields:

- "enabled": whether to run the agent each round
- "model": which LLM to use for crafting the image prompt (e.g., "same-as-lm1", "opus", "sonnet", "gpt4o"). To use Sonnet 3.5 set `"model": "sonnet"`.
- "system_prompt": system prompt for the media agent
- "tool": MCP tool config with "server" (e.g., "comfyui"), tool "name" (e.g., "generate_image"), and default args
- "mode": `"t2i"` for text-to-image or `"edit"` for iterative image editing
- "t2i_use_summary": optional boolean to base t2i prompts on a short running summary plus the latest round (default false)

The agent:
- Builds a short prompt based on the current round’s messages.
- Calls your selected LLM to produce a concise image prompt.
- Calls the MCP tool (e.g., ComfyUI `generate_image`) with that text and logs the result.

Note: The agent uses `MCP_CONFIG` env var if set, otherwise `mcp.config.json` in the project root.

### Modes

- t2i (default):
  - Generates a concise text-to-image prompt from the latest round.
  - Optionally, set `"t2i_use_summary": true` to incorporate a short running conversation summary.
  - Example config:
    {
      "enabled": true,
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
      "enabled": true,
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
- Enable media: ensure `templates/<template>.media.json` exists with `"enabled": true`.
- Run: `python backrooms.py --lm opus opus --template cli`

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

**Template Media Config**
- File: `templates/<template>.media.json`
- "enabled": set to true to activate.
- "model": `same-as-lm1` or a key from `MODEL_INFO` (e.g., `opus`, `sonnet`, `gpt4o`).
- "system_prompt": system prompt for generating the image prompt text.
- "tool.server": MCP server key from `mcp.config.json` (e.g., `comfyui`).
- "tool.name": MCP tool to call (e.g., `generate_image`).
- "tool.defaults": default args merged into each call (e.g., width/height).

Example:
{
  "enabled": true,
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
