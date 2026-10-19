# LangGraph Template Guide

## Purpose

This repository is a configuration-driven LangGraph starter. Prefer changing
workflow configuration and workflow-specific behavior over changing reusable
runtime infrastructure.

## Sources of Truth

- `src/latest_ai_development/config/agents.yaml` owns agent roles, prompts,
  concrete providers and model names, state reads/writes, output fields, and
  tool/MCP assignments.
- `src/latest_ai_development/config/graph_config.yaml` owns nodes, entry/end nodes,
  direct edges, conditional routes, tracing, and the final output field.
- `src/latest_ai_development/graph/state.py` explicitly declares `InputState`,
  complete `State`, `OutputState`, reducers, and defaults. Nodes return partial
  state updates only.
- `src/latest_ai_development/config/models.yaml` owns provider generation defaults
  and fallback configuration. Agent entries use `model.primary` and optional
  `model.fallback` sections with `provider`, required `modelId`, and optional
  `generationDefaults` (`temperature`, `topP`, `maxOutputTokens`, and OpenAI
  `presencePenalty`/`frequencyPenalty`). An omitted fallback uses provider
  configuration; explicit `model.fallback: null` disables fallback for that agent.
  An omitted primary provider resolves to `models.yaml.default_provider`; fallback
  inherits the primary provider. Legacy scalar provider/model entries remain valid.
- `src/latest_ai_development/config/tools.yaml` owns tool factories and behavior.
- `src/latest_ai_development/config/connection.yml` owns named connection settings.
- `src/latest_ai_development/config/settings.py` owns generic named-connection resolution.
- `src/latest_ai_development/tools/connection.py` is absent from the base template;
  the owning connector skill generates its secret, connection and lifecycle functions.
- `src/latest_ai_development/config/mcp.yaml` owns MCP server transport settings.
- `workflow.py` derives required request fields and public response fields from
  `InputState` and `OutputState`; `/ask` accepts those fields dynamically.

## Stable Runtime Files

Do not modify these files for an ordinary workflow or agent addition:

- `src/latest_ai_development/graph/graph_builder.py`
- `src/latest_ai_development/nodes/agent_node_builder.py`
- `src/latest_ai_development/tools/tool_registry.py`
- `src/latest_ai_development/workflow.py`
- `src/latest_ai_development/main.py`
- `src/latest_ai_development/llm/*`
- `src/latest_ai_development/tracking.py`
- `tests/test_template.py`

A change to these files requires a runtime capability that the current
configuration contract cannot express.

Do not generate per-workflow `AskRequest` or `AskResponse` models and do not
hard-code `user_input` or `final_response` in the stable runtime files. Update
the explicit typed classes and exported field constants in `graph/state.py`;
the reusable API/workflow boundary discovers them automatically.

## Adding An Agent

1. Add the agent to `config/agents.yaml`.
2. Add an agent node to `config/graph_config.yaml`.
3. Put the complete prompt inline in the agent entry; legacy prompt-file
   references remain supported when intentionally needed.
4. Add direct or conditional transitions in `graph_config.yaml`.
5. Declare agent input, internal, and output fields in the typed state classes
   in `graph/state.py`.

Agent nodes must use `node_type: agent` and reference an existing agent ID with
`agent_name` (or `agent_id` when its display name differs). Do not
create another node-builder class for standard LLM agents.

Use `output_field` for a single text result. Use `output_mapping` when a generated
Pydantic response schema must populate multiple state fields, including fields
consumed by conditional routing. Do not parse model-generated JSON manually.
Preserve the design's `output_schema` alongside the mapping. Required outputs,
especially routing status, must always be present. Optional branch-specific
outputs use nullable structured-response fields. Missing or null optional values
explicitly clear the corresponding state fields to `None`, so previous results
do not survive a later failure or out-of-scope invocation. The runtime bypasses
reducers when clearing and reinitializing those fields, while preserving normal
accumulation for available values. Consumers must check the current status and
data availability before using an output.
Use `output_mapping` even for a single optional text output.

## Deterministic Nodes

Use deterministic nodes for input validation, normalization, routing-decision
calculation, retry counters, policy gates, and final response shaping. Every
deterministic node declares visible `operation_type`, `description`, and
`metadata.parameters` plus exact `reads_state` and `writes_state` in
`graph_config.yaml`.

Implement every deterministic node as a direct function in `workflow_nodes.py`
and register it under the exact graph node ID in `NODE_REGISTRY`. Register named
routers in `ROUTER_REGISTRY`. The operation metadata describes intent and
parameters for validation and UI display; it is never interpreted as executable
Python. Do not put builder classes, factory methods, or Python import paths in
graph YAML. Put all simple or complex behavior in the registered functions.

Prompts describe outputs; graph configuration controls execution order.

## Execution And Persistence

Use `ainvoke()` in async application paths. Keep the compiled graph and configured
agents reusable across requests rather than rebuilding them during node execution.
Enable a supported checkpointer in `graph_config.yaml` and always pass a stable
`thread_id`; resume an interrupt with `Command(resume=...)` using that same ID.

## Routing Rules

- Use `edges` for unconditional transitions.
- Use `conditional_edges` for mutually exclusive runtime branches.
- Prefer `decision_field` plus `path_map`.
- Use a custom `router` only when one state field cannot express the decision.
- Use `command_destinations` when a node returns `Command(update=..., goto=...)`.
- Use `send_destinations` with a custom router when it returns dynamic `Send`
  fan-out instructions.
- Do not configure more than one outgoing mechanism on a node: direct edges,
  conditional edges, or `Command` destinations.
- Every configured node must be reachable from `entry_point`.
- Every reachable path must be able to terminate at `END`.
- Every `end_nodes` entry must route only to `END`.

Use `runtime.context["values"]` for per-invocation dependencies and settings that
must not be persisted in graph state. Add defaults under `runtime_context` in
`graph_config.yaml`; do not copy runtime context into `State`.

## Tools

Declare Python tool factories in `config/tools.yaml`, then assign tool names to
agents. Keep MCP server definitions in `config/mcp.yaml` and reference server
names through an agent's `mcp_servers` list. The generic runtime resolves Python
tools by graph node ID. Do not patch individual agent-node builders or the
registry for each tool assignment.

Tool assignment is many-to-many. An agent's `tools` list may contain multiple
tool names, and the same tool name may appear in multiple agents. Define each
unique tool factory once in `tools.yaml`; the registry caches one runtime
instance per tool name and reuses it wherever assigned. Preserve tool-list order
and remove duplicate names within an individual agent configuration.

Use `factory_kwargs` only for non-secret constructor values such as table scope
or feature flags. Keep credentials in environment variables or the secrets
manager. Put connector metadata and non-secret configuration in `connection.yml`;
reference it with `connection_name` in `tools.yaml`. Do not duplicate connection
settings in tools or override resolved `tool_config` through `factory_kwargs`.
Generic `resolve_tool_config` stays in `config/settings.py`; generated connection
helpers contain connector-specific code only. Keep tool input schemas, metadata
retrieval and query execution in the generated tool module.
Optional description, tags, and usage/non-usage guidelines
should describe the generated tool without changing runtime wiring.

MCP configuration is validated by this POC, but MCP client execution requires an
adapter implementation and dependency. Until then, startup validation rejects
non-empty `mcp_servers` assignments instead of silently ignoring them.

Add direct and transitive runtime requirements to `pyproject.toml` when a
generated tool imports them, then update `uv.lock` with `uv lock`.

## Generated Change Budget

An initial four-agent, two-tool workflow should normally touch 6-11 files. An
ordinary additional agent should normally touch 3-4 files. Broad changes to API,
workflow bootstrap, graph compilation, model fallback, or tracking require an
explicit justification.

The test suite is intentionally configuration-driven and contained in
`tests/test_template.py`. Generated workflows must pass it without changing it.

## Required Checks

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check src tests
uv run mypy src
uv run python -m compileall -q src tests
git diff --check
```

Do not report implementation completion while any required check fails. Do not
commit temporary `_work` directories, generated caches, or local output files.
