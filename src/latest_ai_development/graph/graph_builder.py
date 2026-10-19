from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from inspect import Parameter, isawaitable, signature
from typing import Any, cast

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from latest_ai_development.config.settings import get_graph_config
from latest_ai_development.graph.context import RuntimeContext
from latest_ai_development.graph.state import OutputState, State
from latest_ai_development.logging_utils import summarize_state_delta, summarize_state_keys
from latest_ai_development.nodes.agent_node_builder import AgentNodeBuilder
from latest_ai_development.nodes.workflow_nodes import NODE_REGISTRY, ROUTER_REGISTRY

logger = logging.getLogger(__name__)


class GraphBuilder:
    @staticmethod
    def _resolve_node_callable(
        node_id: str,
        node_config: Mapping[str, Any],
    ) -> Callable[..., Any]:
        override = NODE_REGISTRY.get(node_id)
        if override is not None:
            return override
        if node_config.get("type") == "agent":
            return AgentNodeBuilder.build(
                node_id=node_id,
                agent_id=str(node_config["agent"]),
                execution_mode=str(node_config.get("execution_mode") or "async"),
            )
        raise ValueError(
            f"Deterministic node '{node_id}' is absent from NODE_REGISTRY"
        )

    @staticmethod
    def _resolve_router_callable(router_id: str) -> Callable[..., Any]:
        try:
            return ROUTER_REGISTRY[router_id]
        except KeyError as exc:
            raise ValueError(f"Router '{router_id}' is absent from ROUTER_REGISTRY") from exc

    @staticmethod
    def _as_state_mapping(value: Any) -> Mapping[str, Any] | None:
        return value if isinstance(value, Mapping) else None

    @staticmethod
    def _invoke_with_runtime(
        target: Callable[..., Any],
        state: State,
        runtime: Runtime[RuntimeContext],
    ) -> Any:
        """Pass Runtime only to callables that explicitly support it."""
        try:
            parameters = signature(target).parameters
        except (TypeError, ValueError):
            return target(state)

        runtime_parameter = parameters.get("runtime")
        if runtime_parameter is not None:
            if runtime_parameter.kind == Parameter.KEYWORD_ONLY:
                return target(state, runtime=runtime)
            return target(state, runtime)
        if any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return target(state, runtime=runtime)
        return target(state)

    @classmethod
    def _wrap_node_callable(
        cls,
        node_id: str,
        node_callable: Callable[..., Any],
    ) -> Callable[..., Any]:
        async def wrapped_node(
            state: State,
            /,
            *,
            runtime: Runtime[RuntimeContext],
        ) -> Any:
            input_state = cls._as_state_mapping(state)
            logger.info("Node entered | node_id=%s", node_id)
            logger.debug(
                "Node input snapshot | node_id=%s | state_keys=%s",
                node_id,
                summarize_state_keys(input_state),
            )
            try:
                result = cls._invoke_with_runtime(
                    node_callable,
                    state,
                    runtime,
                )
                if isawaitable(result):
                    result = await result
            except Exception:
                logger.exception("Node failed | node_id=%s", node_id)
                raise

            result_state = cls._as_state_mapping(result)
            logger.info("Node exited | node_id=%s", node_id)
            logger.debug(
                "Node output snapshot | node_id=%s | state_keys=%s | delta=%s",
                node_id,
                summarize_state_keys(result_state),
                summarize_state_delta(input_state, result_state),
            )
            return result

        return wrapped_node

    @classmethod
    def _wrap_router_callable(
        cls,
        source_node_id: str,
        router_callable: Callable[..., Any],
    ) -> Callable[..., Any]:
        def wrapped_router(
            state: State,
            /,
            *,
            runtime: Runtime[RuntimeContext],
        ) -> Any:
            logger.info("Router evaluated | source_node=%s", source_node_id)
            try:
                decision = cls._invoke_with_runtime(router_callable, state, runtime)
            except Exception:
                logger.exception("Router failed | source_node=%s", source_node_id)
                raise
            logger.info(
                "Router selected path | source_node=%s | decision=%s",
                source_node_id,
                decision,
            )
            return decision

        wrapped_router.__name__ = f"{source_node_id}_router"
        return wrapped_router

    @staticmethod
    def _build_state_router(
        source_node_id: str,
        decision_field: str,
        path_map: Mapping[str, Any],
        default_route: str | None,
    ) -> Callable[[State], str]:
        def route_from_state(state: State, /) -> str:
            decision = str(state.get(decision_field, "")).strip()
            if decision in path_map:
                return decision
            if default_route and default_route in path_map:
                logger.info(
                    "Router used default path | source_node=%s | decision=%s | default=%s",
                    source_node_id,
                    decision or "<empty>",
                    default_route,
                )
                return default_route
            raise ValueError(
                f"Node '{source_node_id}' produced unsupported routing value "
                f"'{decision}' for state field '{decision_field}'"
            )

        return route_from_state

    @staticmethod
    def _resolve_target(target: str) -> Any:
        return END if target == "END" else target

    @staticmethod
    def _build_checkpointer(graph_config: Mapping[str, Any]) -> InMemorySaver | None:
        persistence = graph_config.get("persistence", {})
        if not isinstance(persistence, Mapping) or not persistence.get("enabled", False):
            return None
        backend = str(persistence.get("backend", "memory")).strip().lower()
        if backend != "memory":
            raise ValueError(f"Unsupported persistence backend '{backend}'")
        return InMemorySaver()

    @classmethod
    def build(cls):
        graph_config = get_graph_config()
        nodes = graph_config["nodes"]
        logger.info(
            "Graph compilation started | entry_point=%s | final_output_field=%s | node_count=%s",
            graph_config.get("entry_point"),
            graph_config.get("final_output_field"),
            len(nodes),
        )

        workflow = StateGraph(
            State,
            context_schema=RuntimeContext,
            # Workflow validates public InputState before adding internal defaults.
            # Preserve that complete initial state when entering the graph.
            input_schema=State,
            output_schema=OutputState,
        )
        for node_id, node_config in nodes.items():
            node_callable = cls._resolve_node_callable(node_id, node_config)
            node_metadata = {"component_type": "node", "node_id": node_id}
            if node_config.get("type") == "agent":
                node_metadata["agent_id"] = str(node_config["agent"])
            destinations = tuple(
                cls._resolve_target(str(target))
                for target in node_config.get("command_destinations", [])
            )
            workflow.add_node(
                node_id,
                cast(Any, cls._wrap_node_callable(node_id, node_callable)),
                metadata=node_metadata,
                destinations=destinations or None,
            )

        workflow.add_edge(START, str(graph_config["entry_point"]))
        for edge_config in graph_config.get("edges", []):
            workflow.add_edge(
                str(edge_config["from"]),
                cls._resolve_target(str(edge_config["to"])),
            )

        for conditional_config in graph_config.get("conditional_edges", []):
            source = str(conditional_config["from"])
            send_destinations = conditional_config.get("send_destinations")
            if send_destinations is not None:
                router = cls._resolve_router_callable(str(conditional_config["router"]))
                workflow.add_conditional_edges(
                    source,
                    cast(Any, cls._wrap_router_callable(source, router)),
                )
                continue

            path_map = {
                str(label): cls._resolve_target(str(target))
                for label, target in conditional_config["path_map"].items()
            }
            router_reference = conditional_config.get("router")
            if router_reference:
                router = cls._resolve_router_callable(str(router_reference))
            else:
                router = cls._build_state_router(
                    source,
                    str(conditional_config["decision_field"]),
                    path_map,
                    conditional_config.get("default_route"),
                )
            workflow.add_conditional_edges(
                source,
                cast(Any, cls._wrap_router_callable(source, router)),
                cast(Any, path_map),
            )

        logger.info("Graph compilation completed")
        graph_name = str(graph_config.get("graph_id") or "langgraph_workflow").strip()
        return workflow.compile(
            checkpointer=cls._build_checkpointer(graph_config),
            name=graph_name,
        )
