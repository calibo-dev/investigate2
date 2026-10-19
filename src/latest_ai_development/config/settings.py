from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = BASE_DIR / "src" / "latest_ai_development" / "config"

DEFAULT_OUTPUT_DIR = BASE_DIR / "output"

PROVIDER_ALIASES = {
    "anthropic": "anthropicai",
    "bedrock": "bedrockai",
    "gemini": "geminiai",
    "microsoftfoundry": "microsoftfoundryai",
}


def normalize_api_context(raw_context: str | None) -> str:
    """Return a stable URL prefix like '/testing' or ''."""
    if not raw_context:
        return ""

    normalized = raw_context.strip()
    if not normalized or normalized == "/":
        return ""

    return "/" + normalized.strip("/")


def normalize_provider_name(provider: str | None) -> str:
    """Return the canonical provider key used in models.yaml."""
    provider_key = (provider or "").strip().lower()
    return PROVIDER_ALIASES.get(provider_key, provider_key)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "latest-ai-development"
    app_env: str = Field(default="dev", alias="APP_ENV")
    debug: bool = Field(default=False, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_api_key_secret: str = Field(default="", alias="OPENAI_API_KEY_SECRET")
    anthropicai_api_key: str = Field(default="", alias="ANTHROPICAI_API_KEY")
    anthropicai_api_key_secret: str = Field(default="", alias="ANTHROPICAI_API_KEY_SECRET")
    geminiai_api_key: str = Field(default="", alias="GEMINIAI_API_KEY")
    geminiai_api_key_secret: str = Field(default="", alias="GEMINIAI_API_KEY_SECRET")
    bedrock_ai_api_key: str = Field(default="", alias="BEDROCK_AI_API_KEY")
    bedrock_ai_api_key_secret: str = Field(default="", alias="BEDROCK_AI_API_KEY_SECRET")
    microsoft_foundry_ai_api_key: str = Field(
        default="", alias="MICROSOFT_FOUNDRY_AI_API_KEY"
    )
    microsoft_foundry_ai_api_key_secret: str = Field(
        default="", alias="MICROSOFT_FOUNDRY_AI_API_KEY_SECRET"
    )
    microsoft_foundry_ai_endpoint: str = Field(
        default="", alias="MICROSOFT_FOUNDRY_AI_ENDPOINT"
    )

    cloud_provider: str = Field(default="", alias="CLOUD_PROVIDER")
    azure_key_vault_url: str = Field(default="", alias="AZURE_KEY_VAULT_URL")
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")

    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_api_key_secret: str = Field(default="", alias="LANGSMITH_API_KEY_SECRET")
    langsmith_endpoint: str = Field(default="", alias="LANGSMITH_ENDPOINT")
    langsmith_project: str = Field(default="", alias="LANGSMITH_PROJECT")
    tracing_backend: str = Field(default="NONE", alias="TRACING_BACKEND")

    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_root_path: str = Field(default="", alias="API_ROOT_PATH")
    port: int = Field(default=8089, alias="PORT")

    context: str = Field(default="/", alias="CONTEXT")

    report_output_file: str = Field(default="report.md", alias="REPORT_OUTPUT_FILE")
    output_dir: str = Field(default=str(DEFAULT_OUTPUT_DIR), alias="OUTPUT_DIR")

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_value(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on", "debug"}:
                return True
            if normalized in {"0", "false", "no", "off", "release", "prod", "production", ""}:
                return False

        return bool(value)

    @field_validator("langsmith_tracing", mode="before")
    @classmethod
    def parse_langsmith_tracing_value(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off", ""}:
                return False

        return bool(value)

    @property
    def normalized_api_root_path(self) -> str:
        return normalize_api_context(self.api_root_path or self.context)

    @property
    def output_dir_path(self) -> Path:
        return Path(self.output_dir).expanduser()

    @property
    def report_output_path(self) -> Path:
        return self.output_dir_path / self.report_output_file

    @property
    def models_config_path(self) -> Path:
        return CONFIG_DIR / "models.yaml"

    @property
    def graph_config_path(self) -> Path:
        return CONFIG_DIR / "graph_config.yaml"

    @property
    def agents_config_path(self) -> Path:
        return CONFIG_DIR / "agents.yaml"

    @property
    def tools_config_path(self) -> Path:
        return CONFIG_DIR / "tools.yaml"

    @property
    def connections_config_path(self) -> Path:
        return CONFIG_DIR / "connection.yml"

    @property
    def mcp_config_path(self) -> Path:
        return CONFIG_DIR / "mcp.yaml"


def ensure_runtime_dirs(settings: Settings) -> None:
    settings.output_dir_path.mkdir(parents=True, exist_ok=True)


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML object: {path}")

    return data


def normalize_graph_config(config: dict[str, Any]) -> dict[str, Any]:
    """Translate the design-aligned YAML graph into the runtime representation."""
    graph = config.get("graph")
    if not isinstance(graph, dict):
        return config

    final_output_field = graph.get("final_output_field")
    if not final_output_field:
        output_mapping = graph.get("output_mapping")
        if isinstance(output_mapping, dict) and len(output_mapping) == 1:
            final_output_field = next(iter(output_mapping.values()))
        else:
            output_schema = graph.get("output_schema")
            if isinstance(output_schema, dict) and len(output_schema) == 1:
                final_output_field = next(iter(output_schema))

    normalized = {key: value for key, value in config.items() if key != "graph"}
    normalized.update(graph)
    normalized.update(
        {
            "graph_id": graph.get("graph_id"),
            "description": graph.get("description"),
            "entry_point": graph.get("entry_node"),
            "end_nodes": graph.get("end_nodes", []),
            "final_output_field": final_output_field,
        }
    )

    for field_name in ("persistence", "execution", "loops"):
        if field_name in graph:
            normalized[field_name] = graph[field_name]

    nodes: dict[str, dict[str, Any]] = {}
    for node in graph.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("node_id") or "").strip()
        if not node_id:
            continue
        if node_id in nodes:
            raise ValueError(f"Duplicate graph node_id: {node_id}")
        node_type = str(node.get("node_type") or "").strip().lower()
        if node_type == "agent":
            runtime_node: dict[str, Any] = {
                "type": "agent",
                "agent": node.get("agent_id") or node.get("agent_name"),
                "description": node.get("description"),
                "reads_state": node.get("reads_state", []),
                "writes_state": node.get("writes_state", []),
            }
        else:
            runtime_node = {
                "type": "deterministic",
                "reads_state": node.get("reads_state", []),
                "writes_state": node.get("writes_state", []),
                "operation": {
                    "type": node.get("operation_type"),
                    "description": node.get("description"),
                    "parameters": (
                        node.get("metadata", {}).get("parameters", {})
                        if isinstance(node.get("metadata"), dict)
                        else {}
                    ),
                },
            }
        runtime_node["execution_mode"] = node.get("execution_mode")
        metadata = node.get("metadata")
        interaction = metadata.get("user_interaction") if isinstance(metadata, dict) else None
        runtime_node["runtime_override"] = node.get("runtime_override") is True or (
            isinstance(interaction, dict)
            and bool(interaction.get("prompt_field"))
            and bool(interaction.get("response_field"))
        )
        runtime_node["metadata"] = (
            dict(node.get("metadata") or {})
            if isinstance(node.get("metadata"), dict)
            else {}
        )
        for field_name in ("command_destinations",):
            if field_name in node:
                runtime_node[field_name] = node[field_name]
        nodes[node_id] = runtime_node
    normalized["nodes"] = nodes

    normalized["edges"] = [
        {"from": edge.get("source"), "to": edge.get("target")}
        for edge in graph.get("edges", [])
        if isinstance(edge, dict)
    ]
    existing_edges = {
        (str(edge.get("from")), str(edge.get("to")))
        for edge in normalized["edges"]
    }
    for end_node in normalized["end_nodes"]:
        pair = (str(end_node), "END")
        if pair not in existing_edges:
            normalized["edges"].append({"from": pair[0], "to": pair[1]})
    conditional_edges: list[dict[str, Any]] = []
    for edge in graph.get("conditional_edges", []):
        if not isinstance(edge, dict):
            continue
        runtime_edge: dict[str, Any] = {"from": edge.get("source")}
        for field_name in (
            "decision_field",
            "router",
            "default_route",
            "send_destinations",
        ):
            if field_name in edge:
                runtime_edge[field_name] = edge[field_name]
        routes = edge.get("routes")
        if isinstance(routes, list):
            path_map: dict[str, Any] = {}
            for route in routes:
                if not isinstance(route, dict) or not route.get("route_key"):
                    continue
                route_key = str(route["route_key"])
                if route_key in path_map:
                    raise ValueError(
                        f"Duplicate route_key '{route_key}' for conditional source "
                        f"'{edge.get('source')}'"
                    )
                path_map[route_key] = route.get("target")
            runtime_edge["path_map"] = path_map
        conditional_edges.append(runtime_edge)
    normalized["conditional_edges"] = conditional_edges
    return normalized


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    ensure_runtime_dirs(settings)
    return settings


@lru_cache
def get_models_config() -> dict[str, Any]:
    return load_yaml_file(get_settings().models_config_path)


@lru_cache
def get_graph_config() -> dict[str, Any]:
    return normalize_graph_config(load_yaml_file(get_settings().graph_config_path))


@lru_cache
def get_agents_config() -> dict[str, Any]:
    return load_yaml_file(get_settings().agents_config_path)


@lru_cache
def get_tools_config() -> dict[str, Any]:
    return load_yaml_file(get_settings().tools_config_path)


@lru_cache
def get_mcp_config() -> dict[str, Any]:
    return load_yaml_file(get_settings().mcp_config_path)


@lru_cache
def get_connections_config() -> dict[str, Any]:
    return load_yaml_file(get_settings().connections_config_path)


def resolve_tool_config(tool_name: str, tool_config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve a named connection without changing the tool factory interface.

    Resolution does not open a connection or fetch credentials. Connector-specific
    secrets and lifecycle code are generated separately by the owning skill.
    """
    resolved = dict(tool_config)
    if "connection_name" not in resolved:
        return resolved
    name = resolved["connection_name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Tool '{tool_name}' connection_name must be a non-empty string")
    if "connector" in resolved or "configuration" in resolved:
        raise ValueError(
            f"Tool '{tool_name}' must move connector/configuration to connection.yml "
            "when using connection_name"
        )
    connections = get_connections_config().get("connections")
    connection = connections.get(name) if isinstance(connections, Mapping) else None
    if not isinstance(connection, Mapping):
        raise ValueError(f"Tool '{tool_name}' references undefined connection '{name}'")
    if not isinstance(connection.get("connector"), Mapping) or not isinstance(
        connection.get("configuration"), Mapping
    ):
        raise ValueError(f"Connection '{name}' requires connector and configuration mappings")
    resolved["connector"] = dict(connection["connector"])
    resolved["configuration"] = dict(connection["configuration"])
    return resolved
