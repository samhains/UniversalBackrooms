DreamSim3 Dataset Runner Notes

The dataset runner `scripts/dreamsim3_dataset.py` batches DreamSim3 runs by reading dreams from Supabase and invoking `backrooms.py` for each. It now:

- Runs `backrooms.py` from the repository root (`cwd` pinned) so relative paths resolve.
- Sets `MCP_SERVERS_CONFIG` to the absolute path of `mcp.config.json` so MCP servers (e.g., Discord) load consistently.

Typical usage
- List recent dreams from all sources, run Sonnet3 pairs, post to Discord:
  python scripts/dreamsim3_dataset.py --query "static" --limit 200 --source all --models sonnet3 --max-turns 30 --discord dreamsim

Randomization
- The runner shuffles (randomizes) the dream order by default for both recent and search queries.
- Disable shuffling with `--no-shuffle`.
- Provide a `--seed <int>` to make the randomized order reproducible.

Discord requirements
- `discord/<profile>.json` must exist and have `"enabled": true`.
- `mcp.config.json` must include a `discord` MCP server entry with a valid token.
- Any LLM model used by the Discord agent (e.g., OpenRouter Sonnet) must have its API key in the environment.

Troubleshooting
- No Discord output? Ensure the above requirements and that `mcp.config.json` is readable.
- If running from another directory, the runner pins `cwd` to repo root and passes `MCP_SERVERS_CONFIG`, avoiding path issues.
