from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from functools import reduce
from operator import or_
from pathlib import Path
from types import GenericAlias
from typing import (
    Annotated,
    Any,
    Literal,
    NotRequired,
    Required,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.types import Overwrite
from pydantic import BaseModel, ConfigDict, Field, create_model

from latest_ai_development.config.settings import (
    BASE_DIR,
    get_agents_config,
)
from latest_ai_development.graph.state import State
from latest_ai_development.llm.llm_factory import LLMFactory
from latest_ai_development.llm.model_attempt_logging import (
    LoggedModelFallbackMiddleware,
    ModelAttemptLoggingMiddleware,
    get_model_attempts,
)
from latest_ai_development.llm.model_settings import (
    ModelSettings,
    get_agent_model_settings,
)
from latest_ai_development.logging_utils import summarize_tools
from latest_ai_development.tools.tool_registry import get_tools_for_node

logger = logging.getLogger(__name__)


def _get_agent_config(agent_id: str) -> dict[str, Any]:
    agents = get_agents_config().get("agents", {})
    agent_config = agents.get(agent_id) if isinstance(agents, Mapping) else None
    if not isinstance(agent_config, dict):
        raise ValueError(f"Agent '{agent_id}' is not defined in agents.yaml")
    return agent_config


def _load_prompt(prompt_value: str) -> str:
    """Return an inline prompt, while retaining legacy prompt-file support."""
    value = prompt_value.strip()
    if not value:
        raise ValueError("Agent prompt must not be empty")

    looks_like_file = "\n" not in value and (
        value.startswith("prompts/") or value.endswith((".md", ".txt"))
    )
    if not looks_like_file:
        return value

    prompt_path = Path(value)
    if not prompt_path.is_absolute():
        prompt_path = BASE_DIR / "src" / "latest_ai_development" / prompt_path
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Agent prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def _build_user_message(state: Mapping[str, Any], fields: Sequence[str]) -> str:
    sections = []
    for field in fields:
        label = field.replace("_", " ").title()
        sections.append(f"{label}:\n{state.get(field, '')}")
    return "\n\n".join(sections)


def _extract_content(result: Any) -> Any:
    if isinstance(result, Mapping) and result.get("messages"):
        return getattr(result["messages"][-1], "content", result["messages"][-1])
    return getattr(result, "content", result)


def _optional_output_fields(agent_config: Mapping[str, Any]) -> set[str]:
    schema = agent_config.get("output_schema") or {}
    properties = schema.get("properties", schema)
    required = schema.get("required", [])
    return {
        str(output_field)
        for output_field, state_field in (agent_config.get("output_mapping") or {}).items()
        if isinstance(properties.get(state_field), Mapping)
        and (state_field not in required if "properties" in schema
             else properties[state_field].get("required", True) is False)
    }


def _output_type(spec: Mapping[str, Any], name: str, fallback: Any = Any) -> Any:
    """Preserve declared nested output shapes instead of reducing them to dict/Any."""
    alternatives = spec.get("anyOf") or spec.get("oneOf")
    if alternatives:
        return reduce(or_, (_output_type(item, f"{name}{index}", fallback)
                            for index, item in enumerate(alternatives)))
    kind = spec.get("type")
    if isinstance(kind, list):
        return reduce(or_, (_output_type({**spec, "type": item}, name, fallback)
                            for item in kind))
    if kind == "null" and not spec.get("enum"):
        return type(None)
    result: Any = {"string": str, "integer": int, "number": float,
                   "boolean": bool, "null": type(None)}.get(str(kind), fallback)
    if spec.get("enum"):
        result = Literal[tuple(spec["enum"])]
    elif kind == "array":
        result = GenericAlias(list, _output_type(spec.get("items") or {}, name + "Item"))
    elif kind == "object" and spec.get("properties"):
        required = spec.get("required") or []
        if not isinstance(required, list):
            required = []
        fields: dict[str, Any] = {}
        for key, child in spec["properties"].items():
            fields[key] = (_output_type(child, name + key.title()),
                           ... if key in required else child.get("default", None))
        result = create_model(name, __config__=ConfigDict(
            extra="forbid" if spec.get("additionalProperties") is False else "allow",
        ), **fields)
    elif kind == "object":
        values = spec.get("additionalProperties")
        value_type = _output_type(values, name + "Value") if isinstance(values, Mapping) else Any
        result = GenericAlias(dict, (str, value_type))
    constraints = {target: spec[source] for source, target in (
        ("minimum", "ge"), ("maximum", "le"), ("exclusiveMinimum", "gt"),
        ("exclusiveMaximum", "lt"), ("minLength", "min_length"),
        ("maxLength", "max_length"), ("minItems", "min_length"),
        ("maxItems", "max_length"), ("pattern", "pattern"),
    ) if source in spec}
    if spec.get("description"):
        constraints["description"] = spec["description"]
    if constraints:
        result = Annotated[result, Field(**constraints)]
    return result | None if spec.get("nullable") is True else result


def _build_response_format(
    agent_id: str, agent_config: Mapping[str, Any]
) -> type[BaseModel] | None:
    output_mapping = agent_config.get("output_mapping")
    if not output_mapping:
        return None
    if not isinstance(output_mapping, Mapping):
        raise ValueError("Agent output_mapping must be a mapping")
    # Required/NotRequired describe TypedDict key presence, not Pydantic value types.
    state_types = get_type_hints(State)
    optional = _optional_output_fields(agent_config)
    schema = agent_config.get("output_schema") or {}
    properties = schema.get("properties", schema)
    output_types = {
        output_field: _output_type(properties.get(state_field) or {},
                                   re.sub(r"\W", "", agent_id + output_field.title()),
                                   state_types.get(str(state_field), str))
        for output_field, state_field in output_mapping.items()
    }
    fields: dict[str, Any] = {
        str(output_field): (
            output_types[output_field] | None
            if output_field in optional else output_types[output_field],
            ...,
        )
        for output_field, state_field in output_mapping.items()
    }
    name_parts = re.findall(r"[A-Za-z0-9]+", agent_id)
    schema_name = "".join(part[:1].upper() + part[1:] for part in name_parts)
    if not schema_name or schema_name[0].isdigit():
        schema_name = f"Agent{schema_name}"
    model_name = f"{schema_name}Response"
    return create_model(model_name, **fields)


def _build_state_updates(
    result: Any, agent_config: Mapping[str, Any], state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output_mapping = agent_config.get("output_mapping")
    if not output_mapping:
        return {str(agent_config["output_field"]): _extract_content(result)}
    if not isinstance(output_mapping, Mapping):
        raise ValueError("Agent output_mapping must be a mapping")

    structured_content = result.get("structured_response") if isinstance(result, Mapping) else None
    if isinstance(structured_content, BaseModel):
        structured_content = structured_content.model_dump()
    if not isinstance(structured_content, Mapping):
        raise ValueError("Agent did not return its configured structured response")
    optional = _optional_output_fields(agent_config)
    missing_fields = [field for field in output_mapping
                      if field not in structured_content and field not in optional]
    if missing_fields:
        raise ValueError(f"Agent output is missing configured fields: {missing_fields}")
    updates = {}
    state_types = get_type_hints(State, include_extras=True) if optional else {}
    for output_field, state_field in output_mapping.items():
        value = structured_content.get(output_field)
        annotation = state_types.get(str(state_field))
        if get_origin(annotation) in (Required, NotRequired):
            annotation = get_args(annotation)[0]
        # Optional outputs describe this completed call, including absence.
        # Clear unavailable values without passing None through a reducer.
        # Resume ordinary accumulation once the cleared field has a new value.
        if (output_field in optional and get_origin(annotation) is Annotated
                and state is not None and str(state_field) in state
                and (value is None or state[str(state_field)] is None)):
            value = Overwrite(value)
        updates[str(state_field)] = value
    return updates


def _build_middleware(
    agent_id: str, primary_settings: ModelSettings, fallback_settings: list[ModelSettings]
) -> list[Any]:
    primary_attempt, fallback_attempts = get_model_attempts(
        agent_id,
        primary_settings,
        fallback_settings,
    )
    fallback_llms = [LLMFactory().create(settings) for settings in fallback_settings]
    if fallback_llms:
        return [
            LoggedModelFallbackMiddleware(
                primary_attempt,
                fallback_attempts,
                fallback_llms[0],
                *fallback_llms[1:],
            )
        ]
    return [ModelAttemptLoggingMiddleware(primary_attempt)]


class AgentNodeBuilder:
    """Build an agent node entirely from agents.yaml and graph_config.yaml."""

    @staticmethod
    def build(
        *,
        node_id: str,
        agent_id: str,
        execution_mode: str = "async",
    ) -> Callable[[State], Any]:
        agent_config = _get_agent_config(agent_id)
        primary_settings, fallback_settings = get_agent_model_settings(agent_config)
        llm = LLMFactory().create(primary_settings)
        middleware = _build_middleware(agent_id, primary_settings, fallback_settings)
        prompt = _load_prompt(str(agent_config["prompt"]))
        role = str(agent_config.get("role", "")).strip()
        system_prompt = f"Role: {role}\n\n{prompt}" if role else prompt
        reads_state = [str(field) for field in agent_config.get("reads_state", [])]
        tools = get_tools_for_node(node_id)
        response_format = _build_response_format(agent_id, agent_config)
        agent_kwargs: dict[str, Any] = {
            "model": llm,
            "tools": tools,
            "system_prompt": system_prompt,
            "middleware": middleware,
            "name": agent_id,
        }
        if response_format is not None:
            agent_kwargs["response_format"] = response_format
        agent = create_agent(**agent_kwargs)

        def agent_node_sync(state: State, /) -> dict[str, Any]:
            logger.info(
                "Invoking configured agent | node_id=%s | agent_id=%s | tools=%s",
                node_id,
                agent_id,
                summarize_tools(tools),
            )
            result = agent.invoke(
                cast(
                    Any,
                    {"messages": [HumanMessage(content=_build_user_message(state, reads_state))]},
                )
            )
            return _build_state_updates(result, agent_config, state)

        async def agent_node_async(state: State, /) -> dict[str, Any]:
            logger.info(
                "Invoking configured agent | node_id=%s | agent_id=%s | tools=%s",
                node_id,
                agent_id,
                summarize_tools(tools),
            )
            result = await agent.ainvoke(
                cast(
                    Any,
                    {"messages": [HumanMessage(content=_build_user_message(state, reads_state))]},
                )
            )
            return _build_state_updates(result, agent_config, state)

        if execution_mode == "sync":
            return agent_node_sync
        if execution_mode == "async":
            return agent_node_async
        raise ValueError(
            f"Agent node '{node_id}' has unsupported execution_mode '{execution_mode}'"
        )
