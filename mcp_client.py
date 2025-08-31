import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    # MCP Python SDK (https://github.com/modelcontextprotocol/python-sdk)
    from mcp.client.session import ClientSession
    from mcp import StdioServerParameters, types as mcp_types
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


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of SDK objects to JSON-serializable data."""
    # Pydantic BaseModel in MCP types
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return obj.model_dump(by_alias=True)
        except Exception:
            pass
    # Dataclasses or mappings
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    # Simple types
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Fallback to string
    return str(obj)


def _build_params(cfg: MCPServerConfig) -> StdioServerParameters:
    if _IMPORT_ERROR:
        raise MCPClientError(
            f"Missing MCP client dependency: {_IMPORT_ERROR}. Install the 'mcp' package."
        )
    return StdioServerParameters(
        command=cfg.command,
        args=cfg.args or [],
        env=cfg.env or {},
    )


async def list_tools_async(cfg: MCPServerConfig) -> List[Dict[str, Any]]:
    params = _build_params(cfg)
    async with stdio_client(params) as (read, write):
        async with ClientSession(
            read, write, client_info=mcp_types.Implementation(name="UniversalBackrooms", version="0.1.0")
        ) as session:
            await session.initialize()
            result = await session.list_tools()
            # Normalize possible return shapes
            if hasattr(result, "tools"):
                tools_list = result.tools  # type: ignore[attr-defined]
            elif isinstance(result, dict) and "tools" in result:
                tools_list = result["tools"]
            elif isinstance(result, tuple) and len(result) >= 1 and isinstance(result[0], list):
                tools_list = result[0]
            else:
                tools_list = result  # may already be a list

            norm: list[dict[str, Any]] = []
            for t in tools_list:
                if isinstance(t, dict):
                    name = t.get("name")
                    desc = t.get("description") or t.get("desc")
                    schema = t.get("inputSchema") or t.get("input_schema") or t.get("schema")
                else:
                    name = getattr(t, "name", None)
                    desc = getattr(t, "description", None)
                    schema = getattr(t, "inputSchema", None)
                norm.append({"name": name, "description": desc, "inputSchema": schema})
            return norm


async def call_tool_async(
    cfg: MCPServerConfig, name: str, arguments: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    params = _build_params(cfg)
    async with stdio_client(params) as (read, write):
        async with ClientSession(
            read, write, client_info=mcp_types.Implementation(name="UniversalBackrooms", version="0.1.0")
        ) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments=arguments or {})
            # Normalize to plain JSON-serializable dict
            content = None
            is_error = None
            model = None
            if hasattr(result, "content"):
                content = getattr(result, "content")
                is_error = getattr(result, "isError", None)
                model = getattr(result, "model", None)
            elif isinstance(result, dict):
                content = result.get("content")
                is_error = result.get("isError")
                model = result.get("model")
            return {
                "content": _jsonable(content),
                "isError": _jsonable(is_error),
                "model": _jsonable(model),
            }


def list_tools(cfg: MCPServerConfig) -> List[Dict[str, Any]]:
    return asyncio.run(list_tools_async(cfg))


def call_tool(cfg: MCPServerConfig, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return asyncio.run(call_tool_async(cfg, name, arguments))
