import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    # MCP Python SDK (https://github.com/modelcontextprotocol/python-sdk)
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except Exception as e:  # pragma: no cover
    ClientSession = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    stdio_client = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@dataclass
class MCPServerConfig:
    command: str
    args: Optional[List[str]] = None
    env: Optional[Mapping[str, str]] = None


class MCPClientError(RuntimeError):
    pass


async def _connect_stdio(cfg: MCPServerConfig) -> Tuple[ClientSession, Any]:
    if _IMPORT_ERROR:
        raise MCPClientError(
            f"Missing MCP client dependency: {_IMPORT_ERROR}. Install the 'mcp' package."
        )

    params = StdioServerParameters(
        command=cfg.command,
        args=cfg.args or [],
        env=cfg.env or {},
    )

    client_cm = stdio_client(params)
    read, write = await client_cm.__aenter__()
    session = ClientSession(read, write)
    await session.__aenter__()
    # Initialize handshake with minimal client metadata
    await session.initialize(
        client_name="UniversalBackrooms",
        client_version="0.1.0",
        capabilities={},
    )
    return session, client_cm


async def _disconnect(session: ClientSession, client_cm: Any) -> None:
    # Close session and transport cleanly
    try:
        await session.close()
    finally:
        try:
            await session.__aexit__(None, None, None)
        finally:
            try:
                await client_cm.__aexit__(None, None, None)
            except Exception:
                pass


async def list_tools_async(cfg: MCPServerConfig) -> List[Dict[str, Any]]:
    session, client_cm = await _connect_stdio(cfg)
    try:
        tools = await session.list_tools()
        # Tools returned as objects with name/description/parameters
        return [
            {
                "name": t.name,
                "description": getattr(t, "description", None),
                "inputSchema": getattr(t, "inputSchema", None),
            }
            for t in tools
        ]
    finally:
        await _disconnect(session, client_cm)


async def call_tool_async(
    cfg: MCPServerConfig, name: str, arguments: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    session, client_cm = await _connect_stdio(cfg)
    try:
        result = await session.call_tool(name, arguments=arguments or {})
        # Convert result to plain dict form if SDK returns objects
        return {
            "content": getattr(result, "content", None) or result.get("content"),
            "isError": getattr(result, "isError", None)
            if hasattr(result, "isError")
            else result.get("isError"),
            "model": getattr(result, "model", None)
            if hasattr(result, "model")
            else result.get("model"),
        }
    finally:
        await _disconnect(session, client_cm)


def list_tools(cfg: MCPServerConfig) -> List[Dict[str, Any]]:
    return asyncio.run(list_tools_async(cfg))


def call_tool(cfg: MCPServerConfig, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return asyncio.run(call_tool_async(cfg, name, arguments))
