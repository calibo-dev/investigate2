import asyncio
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from latest_ai_development.config import validators


def test_template_configuration_is_valid() -> None:
    validators._validate_tools_config()
    validators._validate_mcp_config()
    validators._validate_agents_config()
    validators._validate_graph_config()


@pytest.fixture
def connection_tools(monkeypatch):
    from pydantic import BaseModel

    from latest_ai_development.config import settings
    from latest_ai_development.tools import tool_registry

    connections = {"primary": {
        "connector": {"category": "RDBMS", "type": "SNOWFLAKE"},
        "configuration": {"account": "account-one"},
    }, "secondary": {
        "connector": {"category": "RDBMS", "type": "SNOWFLAKE"},
        "configuration": {"account": "account-two"},
    }}
    tools = {name: {"factory": "example:create_tool", "connection_name": ref}
             for name, ref in (("One", "primary"), ("Two", "primary"), ("Three", "secondary"))}
    monkeypatch.setattr(settings, "get_connections_config", lambda: {"connections": connections})
    monkeypatch.setattr(validators, "get_connections_config", lambda: {"connections": connections})
    monkeypatch.setattr(validators, "get_tools_config", lambda: {"tools": tools})
    monkeypatch.setattr(tool_registry, "get_tools_config", lambda: {"tools": tools})
    created = []

    class Input(BaseModel):
        value: str = ""

    class Implementation:
        description = "Test configured tool"
        args_schema = Input

        def __init__(self, name, config):
            self.name = name
            self.config = config

        def execute(self, value: str = "") -> str:
            return value

    def factory(*, tool_name, tool_config):
        tool = Implementation(tool_name, tool_config)
        created.append(tool)
        return tool

    monkeypatch.setattr(validators, "_resolve_reference", lambda ref: factory)
    monkeypatch.setattr(
        tool_registry, "import_module", lambda module: SimpleNamespace(create_tool=factory)
    )
    monkeypatch.setattr(tool_registry, "REGISTERED_TOOLS", {})
    return tools, connections, created, tool_registry


def test_named_connections_resolve_validate_and_preserve_tool_cache(connection_tools):
    tools, _, created, registry = connection_tools
    assert validators._validate_tools_config() == set(tools)
    loaded = registry.get_tools_by_names(["One", "Two", "Three", "One"])
    assert len(loaded) == 3
    assert created[0].config["configuration"] == created[1].config["configuration"]
    assert created[2].config["configuration"]["account"] == "account-two"
    assert "configuration" not in tools["One"]
    assert registry.get_tools_by_names(["One"])[0] is loaded[0]
    assert len(created) == 3
    asyncio.run(registry.close_registered_tools())
    assert registry.REGISTERED_TOOLS == {}


@pytest.mark.parametrize("change, message", [
    ({"connection_name": "missing"}, "undefined connection 'missing'"),
    ({"connection_name": ""}, "non-empty string"),
    ({"configuration": {}}, "move connector/configuration"),
    ({"factory_kwargs": {"tool_config": {}}}, "must not override"),
])
def test_named_connection_errors_match_startup_and_runtime(connection_tools, change, message):
    tools, _, _, registry = connection_tools
    tools["One"].update(change)
    with pytest.raises(ValueError, match=message):
        validators._validate_tools_config()
    with pytest.raises(ValueError, match=message):
        registry.get_tools_by_names(["One"])


def test_invalid_connection_definition_rejected(connection_tools):
    _, connections, _, _ = connection_tools
    connections["primary"]["connector"] = {}
    with pytest.raises(ValueError, match="requires connector.category"):
        validators._validate_tools_config()


def test_tools_without_connection_reference_keep_existing_factory_contract(connection_tools):
    tools, _, created, registry = connection_tools
    tools["One"].pop("connection_name")
    tools["One"]["configuration"] = {"account": "legacy"}
    validators._validate_tools_config()
    registry.get_tools_by_names(["One"])
    assert created[0].config == tools["One"]


def test_tool_free_template_initializes_without_connector_modules():
    from latest_ai_development.config.settings import get_tools_config

    if get_tools_config().get("tools"):
        pytest.skip("This regression covers tool-free configurations.")
    root = Path(__file__).resolve().parents[1]
    assert not (root / "src/latest_ai_development/tools/connection.py").exists()
    assert not (root / "src/latest_ai_development/tools/snowflake_connector.py").exists()
    result = subprocess.run(
        [sys.executable, "-c", """
import builtins
original_import = builtins.__import__
def without_connectors(name, *args, **kwargs):
    if name.split('.')[0] in {'snowflake', 'pandas'}:
        raise ImportError('Connector dependency must not be imported: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_connectors
from latest_ai_development.config import validators
from latest_ai_development.config.settings import get_tools_config, get_connections_config
from latest_ai_development.main import _initialize_workflow
from latest_ai_development.graph.state import REQUIRED_INPUT_FIELDS, OUTPUT_FIELDS
assert get_tools_config() == {'tools': {}}
assert get_connections_config() == {'connections': {}}
# Credentials are outside this connector-free startup test.
validators.get_model_settings = lambda *args, **kwargs: None
workflow = _initialize_workflow()
assert workflow.required_inputs() == REQUIRED_INPUT_FIELDS
assert workflow.output_fields() == OUTPUT_FIELDS
"""],
        cwd=root, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_structured_output_uses_values_from_required_and_optional_state_keys(monkeypatch):
    from typing import NotRequired, Required, TypedDict

    from latest_ai_development.nodes import agent_node_builder

    class TestState(TypedDict):
        confidence: NotRequired[float]
        count: Required[int]

    monkeypatch.setattr(agent_node_builder, "State", TestState)
    response = agent_node_builder._build_response_format(
        "Classifier", {"output_mapping": {"confidence": "confidence", "count": "count"}}
    )
    assert response(confidence=0.5, count=2).model_dump() == {"confidence": 0.5, "count": 2}
    with pytest.raises(ValueError):
        response(confidence=0.5)


def test_graph_preserves_internal_defaults_and_public_input_filter(monkeypatch):
    from typing import TypedDict

    from latest_ai_development import workflow
    from latest_ai_development.graph import graph_builder

    class PublicInput(TypedDict):
        request: str

    class PublicOutput(TypedDict):
        result: str

    class CompleteState(PublicInput, PublicOutput, total=False):
        attempts: int

    def finish(state):
        return {"result": str(state["attempts"])}

    monkeypatch.setattr(graph_builder, "State", CompleteState)
    monkeypatch.setattr(graph_builder, "OutputState", PublicOutput)
    monkeypatch.setattr(graph_builder, "NODE_REGISTRY", {"Finish": finish})
    monkeypatch.setattr(graph_builder, "get_graph_config", lambda: {
        "graph_id": "configured_workflow",
        "entry_point": "Finish", "final_output_field": "result", "end_nodes": ["Finish"],
        "nodes": {"Finish": {"type": "deterministic"}}, "edges": [{"from": "Finish", "to": "END"}],
    })
    monkeypatch.setattr(workflow, "INPUT_FIELDS", ("request",))
    monkeypatch.setattr(workflow, "REQUIRED_INPUT_FIELDS", ("request",))
    monkeypatch.setattr(workflow, "state_defaults", lambda: {"attempts": 0})
    controller = workflow.LatestAiDevelopmentWorkflow.__new__(workflow.LatestAiDevelopmentWorkflow)
    initial = controller.build_initial_state({"request": "x", "attempts": 99})
    assert initial == {"request": "x", "attempts": 0}
    graph = graph_builder.GraphBuilder.build()
    assert graph.name == "configured_workflow"
    assert asyncio.run(graph.ainvoke(initial)) == {"result": "0"}


def test_graph_and_agent_trace_names_use_their_configured_ids(monkeypatch):
    from typing import TypedDict

    from latest_ai_development.graph import graph_builder

    class TestState(TypedDict, total=False):
        result: str

    monkeypatch.setattr(graph_builder, "State", TestState)
    monkeypatch.setattr(graph_builder, "OutputState", TestState)
    monkeypatch.setattr(
        graph_builder.AgentNodeBuilder,
        "build",
        lambda **_: lambda state: {"result": "ok"},
    )
    monkeypatch.setattr(graph_builder, "get_graph_config", lambda: {
        "graph_id": "sales_workflow",
        "entry_point": "AnalyzeSales",
        "end_nodes": ["AnalyzeSales"],
        "nodes": {"AnalyzeSales": {"type": "agent", "agent": "sales_analyst"}},
        "edges": [{"from": "AnalyzeSales", "to": "END"}],
    })

    graph = graph_builder.GraphBuilder.build()

    assert graph.name == "sales_workflow"
    assert graph.nodes["AnalyzeSales"].metadata == {
        "component_type": "node",
        "node_id": "AnalyzeSales",
        "agent_id": "sales_analyst",
    }


def test_router_trace_name_uses_its_source_node():
    from latest_ai_development.graph.graph_builder import GraphBuilder

    router = GraphBuilder._wrap_router_callable(
        "evaluate_feedback",
        lambda state: state["route"],
    )

    assert router.__name__ == "evaluate_feedback_router"


def test_interaction_wrapper_owns_only_the_declared_human_response(monkeypatch):
    from typing import TypedDict

    class PublicOutput(TypedDict):
        question: str

    class CompleteState(PublicOutput, total=False):
        request: str
        answer: str
        unexpected: str

    node = {"type": "agent", "agent": "Ask", "reads_state": ["request"],
            "writes_state": ["question", "answer"], "metadata": {"user_interaction": {
                "kind": "clarification", "prompt_field": "question", "response_field": "answer"}}}
    monkeypatch.setattr(validators, "State", CompleteState)
    monkeypatch.setattr(validators, "OutputState", PublicOutput)
    monkeypatch.setattr(validators, "NODE_REGISTRY", {"Ask": lambda state: {}})
    monkeypatch.setattr(validators, "ROUTER_REGISTRY", {})
    monkeypatch.setattr(validators, "get_agents_config", lambda: {"agents": {
        "Ask": {"reads_state": ["request"], "writes_state": ["question"]}}})
    monkeypatch.setattr(validators, "get_graph_config", lambda: {
        "entry_point": "Ask", "final_output_field": "question", "end_nodes": ["Ask"],
        "nodes": {"Ask": node}, "edges": [{"from": "Ask", "to": "END"}],
    })
    validators._validate_graph_config()
    node["writes_state"].append("unexpected")
    with pytest.raises(ValueError, match="writes_state must match"):
        validators._validate_graph_config()


def test_design_graph_preserves_checkpoint_and_resume_configuration():
    from langgraph.checkpoint.memory import InMemorySaver

    from latest_ai_development.config.settings import normalize_graph_config
    from latest_ai_development.graph.graph_builder import GraphBuilder

    graph = {"persistence": {"enabled": True, "backend": "memory"},
             "execution": {"resume_enabled": True}, "loops": [{"counter_field": "attempts"}]}
    normalized = normalize_graph_config({"graph": graph})
    assert normalized["execution"] == graph["execution"]
    assert normalized["loops"] == graph["loops"]
    assert isinstance(GraphBuilder._build_checkpointer(normalized), InMemorySaver)
    graph["persistence"]["enabled"] = False
    assert GraphBuilder._build_checkpointer(normalize_graph_config({"graph": graph})) is None


@pytest.fixture
def workflow_trace_capture(monkeypatch):
    from langsmith import Client, RunTree, traceable

    from latest_ai_development import tracking

    # Exercise the installed tracing SDK without sending requests to LangSmith.
    monkeypatch.setattr(Client, "create_run", lambda *args, **kwargs: None)
    client = Client(api_key="test-only", auto_batch_tracing=False)
    runs = []
    monkeypatch.setattr(RunTree, "patch", lambda run: runs.append(run))
    monkeypatch.setattr(tracking, "traceable", lambda **kwargs: traceable(
        client=client, **kwargs,
    ))
    monkeypatch.setattr(tracking, "get_tracing_backend", lambda: "LANGSMITH")
    monkeypatch.setattr(tracking, "get_graph_config", lambda: {"graph_id": "reference_lookup"})
    monkeypatch.setattr(tracking, "INPUT_FIELDS", ("user_question",))
    monkeypatch.setattr(tracking, "OUTPUT_FIELDS", ("answer",))
    yield tracking, runs
    client.close()


def test_workflow_trace_uses_public_contract_and_preserves_child_spans(workflow_trace_capture):
    from langsmith import tracing_context

    tracking, runs = workflow_trace_capture

    @tracking.traceable(name="reference_tool", run_type="tool")
    def reference_tool():
        return "Example answer"

    async def invoke(state):
        return {"answer": reference_tool(), "internal_value": "not a public output"}

    async def run():
        with tracing_context(enabled=True):
            return await tracking.atrace_workflow_execution(
                invoke, flow_run_id="test-flow",
                inputs={"user_question": "Example question", "credential": "not public"},
                initial_state={"user_question": "Example question"},
            )

    result = asyncio.run(run())
    workflow = next(run for run in runs if run.name == "reference_lookup")
    child = next(run for run in runs if run.name == "reference_tool")
    assert result["internal_value"] == "not a public output"
    assert workflow.inputs == {
        "flow_run_id": "test-flow", "user_question": "Example question",
        "initial_state_keys": ["user_question"],
    }
    assert workflow.outputs == {"answer": "Example answer"}
    assert workflow.metadata["execution_status"] == "success"
    assert child.parent_run_id == workflow.id


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("error,category", [
    (RuntimeError("metadata_discovery_failed"), "execution"),
    (TimeoutError("request timed out"), "timeout"),
])
def test_failed_trace_has_no_fabricated_outputs_and_preserves_exception(
    workflow_trace_capture, asynchronous, error, category,
):
    from langsmith import tracing_context

    tracking, runs = workflow_trace_capture

    def fail(**kwargs):
        raise error

    async def async_fail(**kwargs):
        raise error

    traced = tracking.trace_workflow()(async_fail if asynchronous else fail)
    with tracing_context(enabled=True), pytest.raises(type(error)) as captured:
        if asynchronous:
            asyncio.run(traced(inputs={"user_question": "Example question"}))
        else:
            traced(inputs={"user_question": "Example question"})

    assert captured.value is error
    workflow = runs[-1]
    assert workflow.error
    assert workflow.outputs == {}
    assert workflow.metadata["execution_status"] == "failed"
    assert workflow.metadata["error_category"] == category
    assert workflow.metadata["error_type"] == type(error).__name__
    assert str(error) not in str(workflow.metadata)


def test_workflow_tracing_keeps_explicit_field_overrides(workflow_trace_capture, monkeypatch):
    from langsmith import tracing_context

    tracking, runs = workflow_trace_capture
    monkeypatch.setattr(tracking, "get_graph_config", lambda: {"tracing": {"workflow": {
        "name": "Configured lookup",
        "input_fields": {"question_length": {"path": "inputs.user_question", "transform": "len"}},
        "output_fields": {"answer_length": {"path": "answer", "transform": "len"}},
    }}})

    @tracking.trace_workflow()
    def invoke(**kwargs):
        return {"answer": "hello"}

    with tracing_context(enabled=True):
        invoke(inputs={"user_question": "example"})

    workflow = runs[-1]
    assert workflow.name == "Configured lookup"
    assert workflow.inputs == {"question_length": 7}
    assert workflow.outputs == {"answer_length": 5}


def test_disabled_workflow_trace_keeps_execution_unchanged(monkeypatch):
    from latest_ai_development import tracking

    monkeypatch.setattr(tracking, "get_tracing_backend", lambda: "NONE")
    monkeypatch.setattr(tracking, "traceable", lambda **kwargs: pytest.fail("Tracing disabled"))

    @tracking.trace_workflow()
    def invoke():
        return {"answer": "hello"}

    assert invoke() == {"answer": "hello"}
