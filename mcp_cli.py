import argparse
import json
import os
from typing import Any, Dict, List

from mcp_client import MCPServerConfig, list_tools, call_tool, MCPClientError


def load_server_config(config_path: str, server_name: str) -> MCPServerConfig:
    with open(config_path, "r") as f:
        data = json.load(f)

    # Shape 1: { "servers": [ { name, command, args, env } ] }
    servers_list: List[Dict[str, Any]] = data.get("servers", [])
    for s in servers_list:
        if s.get("name") == server_name:
            return MCPServerConfig(
                command=s["command"], args=s.get("args", []), env=s.get("env", {})
            )

    # Shape 2: { "mcpServers": { key: { type, command, args, env } } }
    mcp_servers: Dict[str, Dict[str, Any]] = data.get("mcpServers", {})
    if server_name in mcp_servers:
        s = mcp_servers[server_name]
        # For now we only support stdio servers
        server_type = s.get("type", "stdio")
        if server_type != "stdio":
            raise ValueError(
                f"Unsupported server type '{server_type}' for '{server_name}'; only 'stdio' supported."
            )
        return MCPServerConfig(
            command=s["command"], args=s.get("args", []), env=s.get("env", {})
        )

    raise KeyError(f"Server '{server_name}' not found in {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Minimal MCP client (stdio)")

    parser.add_argument(
        "--config",
        default=os.getenv("MCP_SERVERS_CONFIG", "mcp_servers.json"),
        help="Path to servers config JSON (default: mcp_servers.json)",
    )
    parser.add_argument(
        "--server",
        required=False,
        help="Server name in config. If omitted, --cmd must be provided.",
    )
    parser.add_argument(
        "--cmd",
        help="Server command (overrides config). Example: 'node server.js'",
    )
    parser.add_argument("--args", nargs=argparse.REMAINDER, help="Args after --")

    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list-tools", help="List available tools")

    call_p = sub.add_parser("call-tool", help="Call a tool with JSON args")
    call_p.add_argument("name", help="Tool name")
    call_p.add_argument(
        "--json", required=False, default="{}", help="JSON string of arguments"
    )
    call_p.add_argument(
        "--wrap-params",
        action="store_true",
        help="Wrap provided JSON as {\"params\": <json>} for FastMCP-style tools",
    )

    args = parser.parse_args()

    if args.cmd:
        cfg = MCPServerConfig(command=args.cmd.split()[0], args=args.args or [])
    elif args.server:
        cfg = load_server_config(args.config, args.server)
    else:
        parser.error("Provide --server from config or --cmd with optional --args")

    try:
        if args.action == "list-tools":
            tools = list_tools(cfg)
            print(json.dumps(tools, indent=2, default=lambda o: getattr(o, 'model_dump', lambda **_: str(o))()))
        elif args.action == "call-tool":
            payload = json.loads(args.json) if args.json else {}
            if args.wrap_params:
                payload = {"params": payload}
            result = call_tool(cfg, args.name, payload)
            print(json.dumps(result, indent=2, default=lambda o: getattr(o, 'model_dump', lambda **_: str(o))()))
        else:
            parser.error("Unknown action")
    except MCPClientError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
