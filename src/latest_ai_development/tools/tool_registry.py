from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import import_module
from inspect import isawaitable, signature
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool

from latest_ai_development.config.settings import (
    get_agents_config,
    get_graph_config,
    get_tools_config,
    resolve_tool_config,
)

REGISTERED_TOOLS: dict[str, Any] = {}
_CONNECTION_MANAGER: Any | None = None
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def register_tool(name: str, tool: Any) -> None:
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("Tool name must not be empty")
    REGISTERED_TOOLS[normalized_name] = tool


def get_tools_by_names(tool_names: Sequence[str] | None) -> list[Any]:
    resolved_tools: list[Any] = []
    seen_names: set[str] = set()

    for tool_name in tool_names or []:
        if not isinstance(tool_name, str):
            continue

        normalized_name = tool_name.strip()
        if not normalized_name or normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)
        tool = REGISTERED_TOOLS.get(normalized_name)
        if tool is None:
            tool = _load_configured_tool(normalized_name)
        if tool is not None:
            resolved_tools.append(tool)

    return resolved_tools


def _load_configured_tool(tool_name: str) -> Any:
    tool_definitions = get_tools_config().get("tools", {})
    tool_config = tool_definitions.get(tool_name) if isinstance(tool_definitions, Mapping) else None
    if not isinstance(tool_config, Mapping):
        return None

    factory_reference = tool_config.get("factory")
    if not isinstance(factory_reference, str):
        return None
    module_name, separator, attribute_name = factory_reference.partition(":")
    if not separator:
        raise ValueError(
            f"Tool factory '{factory_reference}' must use '<module>:<attribute>' format"
        )
    factory: Any = import_module(module_name)
    for part in attribute_name.split("."):
        factory = getattr(factory, part)
    if not callable(factory):
        raise TypeError(f"Configured tool factory '{factory_reference}' is not callable")
    factory_kwargs = tool_config.get("factory_kwargs", {})
    if not isinstance(factory_kwargs, Mapping):
        raise ValueError(f"Tool '{tool_name}' factory_kwargs must be a mapping")
    if "connection_name" in tool_config and "tool_config" in factory_kwargs:
        raise ValueError(f"Tool '{tool_name}' must not override its named connection")
    resolved_kwargs = dict(factory_kwargs)
    parameters = signature(factory).parameters
    resolved_config = dict(tool_config)
    if "connection_manager" in parameters:
        manager = _get_connection_manager()
        from latest_ai_development.tools.connection import (
            resolve_tool_config as resolve_connection_tool_config,
        )

        resolved_config = resolve_connection_tool_config(
            tool_name,
            resolved_config,
            manager.configurations,
        )
        resolved_kwargs.setdefault("connection_manager", manager)
    else:
        resolved_config = resolve_tool_config(tool_name, resolved_config)
    if "tool_name" in parameters:
        resolved_kwargs.setdefault("tool_name", tool_name)
    if "tool_config" in parameters:
        resolved_kwargs.setdefault("tool_config", resolved_config)
    tool = _adapt_tool(factory(**resolved_kwargs))
    register_tool(tool_name, tool)
    return tool


def _get_connection_manager() -> Any:
    global _CONNECTION_MANAGER
    if _CONNECTION_MANAGER is None:
        from latest_ai_development.tools.connection import ConnectionManager

        from latest_ai_development.secrets_manager import SecretsManager

        _CONNECTION_MANAGER = ConnectionManager(
            CONFIG_DIR,
            SecretsManager().get_secret,
        )
    return _CONNECTION_MANAGER


def _adapt_tool(implementation: Any) -> Any:
    if isinstance(implementation, BaseTool):
        return implementation
    execute = getattr(implementation, "execute", None)
    args_schema = getattr(implementation, "args_schema", None)
    if not callable(execute) or args_schema is None:
        raise TypeError("Tool factories must return a native tool or an executable implementation")
    return StructuredTool.from_function(
        func=execute,
        name=implementation.name,
        description=implementation.description,
        args_schema=args_schema,
        tags=getattr(implementation, "tags", []),
        metadata=getattr(implementation, "metadata", {}),
    )


def get_tools_for_node(node_id: str) -> list[Any]:
    """Resolve tools declared for an agent-backed graph node."""
    nodes = get_graph_config().get("nodes", {})
    node_config = nodes.get(node_id) if isinstance(nodes, Mapping) else None
    if not isinstance(node_config, Mapping):
        return []

    agent_id = node_config.get("agent")
    agents = get_agents_config().get("agents", {})
    agent_config = agents.get(agent_id) if isinstance(agents, Mapping) else None
    if not isinstance(agent_config, Mapping):
        return []

    tool_names = agent_config.get("tools", [])
    if not isinstance(tool_names, Sequence) or isinstance(tool_names, (str, bytes)):
        return []
    return get_tools_by_names([str(name) for name in tool_names])


async def close_registered_tools() -> None:
    """Close tool-owned resources once when the application shuts down."""
    tools = list({id(tool): tool for tool in REGISTERED_TOOLS.values()}.values())
    REGISTERED_TOOLS.clear()
    for tool in tools:
        close = getattr(tool, "aclose", None) or getattr(tool, "close", None)
        if not callable(close):
            continue
        result = close()
        if isawaitable(result):
            await result
    global _CONNECTION_MANAGER
    manager, _CONNECTION_MANAGER = _CONNECTION_MANAGER, None
    if manager is not None:
        manager.close()
