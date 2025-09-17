Kie.ai MCP Server Integration

This project integrates the Kie.ai MCP server to generate images (and videos) and return URLs that can be posted or logged by UniversalBackrooms, similar to the ComfyUI flow.

Prerequisites
- Node.js 18+
- Kie.ai API key

Configuration (mcp.config.json)
- Recommended: set working directory so the server’s SQLite DB (tasks.db) stays under `var/tasks/`.

  "mcpServers": {
    "kie-ai-mcp-server": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@andrewlwn77/kie-ai-mcp-server"],
      "cwd": "/Users/samhains/Code/UniversalBackrooms",
      "env": { "KIE_AI_API_KEY": "YOUR_KEY" }
    }
  }

- Alternative: keep current CWD but write the DB into this repo:

  "mcpServers": {
    "kie-ai-mcp-server": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@andrewlwn77/kie-ai-mcp-server"],
      "env": {
        "KIE_AI_API_KEY": "YOUR_KEY",
        "KIE_AI_DB_PATH": "/Users/samhains/Code/UniversalBackrooms/var/tasks/tasks.db"
      }
    }
  }

- Zero-disk option: set an in-memory DB:

  "env": { "KIE_AI_API_KEY": "YOUR_KEY", "KIE_AI_DB_PATH": ":memory:" }

Media preset
- Use `media/kieai.json` to route prompts to Kie.ai’s `generate_nano_banana` tool. It sends arguments directly (no FastMCP param wrapper).
- For a generate→edit flow, use `media/kieai_edit_chain.json` which chains `generate_nano_banana` and `edit_nano_banana`. The edit step passes the previous URL via `image_urls: ["..."]`.
- The server is asynchronous. Presets include a status tool (`get_task_status`) and polling config so the Media Agent waits for a resulting URL and injects it into the result for Discord posting.

Testing via the included MCP client
- List tools:
  python mcp_cli.py --config mcp.config.json --server kie-ai-mcp-server list-tools
- Generate an image:
  python mcp_cli.py --config mcp.config.json --server kie-ai-mcp-server call-tool generate_nano_banana --json '{"prompt":"A liminal hallway lit by buzzing fluorescents"}'

Using in a run
- Pass the media preset name:
  python backrooms.py --media kieai
- The media agent generates a concise image prompt, calls Kie.ai, and logs the returned URL. The Discord agent can then post this URL if configured.

- Or, for the generate→edit chain that updates the image across rounds:
  python backrooms.py --media kieai_edit_chain
  The first round generates an image; subsequent rounds produce edit instructions and call `edit_nano_banana` with the prior image URL.

Notes
- Kie.ai endpoints require HTTP/HTTPS URLs for editing (no local file paths).
- If you prefer to run a local build instead of npx, set `command: node` and `args: ["/path/to/kie-ai-mcp-server/dist/index.js"]` with the same env and (optional) cwd.
