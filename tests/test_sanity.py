import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest

from latest_ai_development.config import validators
from latest_ai_development.config.settings import get_graph_config, normalize_graph_config
from latest_ai_development.graph.state import OUTPUT_FIELDS
from latest_ai_development.workflow import LatestAiDevelopmentWorkflow


def test_tool_registry_injects_shared_connection_manager_and_adapts_tool(monkeypatch):
    from pydantic import BaseModel

    from latest_ai_development.tools import tool_registry

    class Input(BaseModel):
        value: str

    class Implementation:
        name = "Country reference"
        description = "Read country data"
        args_schema = Input
        tags = ["country"]
        metadata = {"connector": "snowflake"}

        def execute(self, value: str) -> str:
            return value.upper()

    class Manager:
        configurations = {"reference": {"connector": {}, "configuration": {}}}

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    manager = Manager()
    connection_module = ModuleType("latest_ai_development.tools.connection")
    connection_module.resolve_tool_config = lambda name, config, connections: {
        **config,
        **connections[config["connection_name"]],
    }
    monkeypatch.setitem(sys.modules, connection_module.__name__, connection_module)
    monkeypatch.setattr(tool_registry, "_get_connection_manager", lambda: manager)
    monkeypatch.setattr(tool_registry, "get_tools_config", lambda: {"tools": {
        "Country reference": {
            "factory": "test_connector:create_tool",
            "connection_name": "reference",
            "description": "Read country data",
        }
    }})
    factory_module = ModuleType("test_connector")

    def create_tool(*, tool_name, tool_config, connection_manager):
        assert tool_name == "Country reference"
        assert tool_config["connector"] == {}
        assert connection_manager is manager
        return Implementation()

    factory_module.create_tool = create_tool
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)
    monkeypatch.setattr(tool_registry, "_CONNECTION_MANAGER", manager)
    tool_registry.REGISTERED_TOOLS.clear()

    tool = tool_registry.get_tools_by_names(["Country reference"])[0]
    assert tool.invoke({"value": "india"}) == "INDIA"
    asyncio.run(tool_registry.close_registered_tools())
    assert manager.closed


@pytest.fixture
def isolated_state(monkeypatch):
    """Runtime unit cases must not depend on the workflow currently generated."""
    from typing_extensions import TypedDict

    from latest_ai_development.graph import graph_builder

    class TestInput(TypedDict):
        user_input: str

    class TestState(TestInput, total=False):
        normalized_feedback: str
        final_response: str
        human_decision: str
        urgency_score: int

    class TestOutput(TypedDict):
        final_response: str

    monkeypatch.setattr(validators, "InputState", TestInput)
    monkeypatch.setattr(validators, "State", TestState)
    monkeypatch.setattr(validators, "OutputState", TestOutput)
    monkeypatch.setattr(graph_builder, "State", TestState)
    monkeypatch.setattr(graph_builder, "OutputState", TestOutput)


@pytest.mark.parametrize('value,label', [(True, 'True'), (False, 'False')])
def test_boolean_state_router_preserves_declared_labels(value, label):
    from latest_ai_development.graph.graph_builder import GraphBuilder
    router = GraphBuilder._build_state_router(
        'Classify', 'clarification_needed', {'True': 'Ask', 'False': 'Respond'}, None,
    )
    assert router({'clarification_needed': value}) == label
    with pytest.raises(ValueError, match='unsupported routing value'):
        router({'clarification_needed': 'clarification_needed_' + label.lower()})


def test_state_router_does_not_guess_boolean_aliases():
    from latest_ai_development.graph.graph_builder import GraphBuilder
    router = GraphBuilder._build_state_router(
        'Classify', 'clarification_needed',
        {'clarification_needed_true': 'Ask', 'clarification_needed_false': 'Respond'}, None,
    )
    for value in (True, False):
        with pytest.raises(ValueError, match='unsupported routing value'):
            router({'clarification_needed': value})


@pytest.mark.parametrize("json_schema", [False, True])
@pytest.mark.parametrize("execution_mode", ["sync", "async"])
@pytest.mark.parametrize("status,sql,error", [
    ("valid_query", "SELECT NAME FROM ak_country", None),
    ("out_of_scope", None, "Requested data is unavailable."),
    ("failed", None, "Metadata retrieval failed."),
])
def test_agent_optional_outputs_preserve_actual_outcome(
    monkeypatch, json_schema, execution_mode, status, sql, error,
):
    from pydantic import ValidationError

    from latest_ai_development.nodes import agent_node_builder as builder

    properties = {"status": {"type": "string", "required": True},
                  "sql": {"type": "string", "required": False},
                  "error": {"type": "string", "required": False}}
    schema = ({"type": "object", "properties": {
        name: {"type": spec["type"]} for name, spec in properties.items()
    }, "required": ["status"]} if json_schema else properties)
    config = {"role": "SQL generator", "prompt": "Return actual outcomes only.",
              "reads_state": [], "output_mapping": {
                  "status": "status", "sql": "sql", "error": "error"},
              "output_schema": schema}
    monkeypatch.setattr(builder, "_get_agent_config", lambda _: config)
    monkeypatch.setattr(builder, "get_agent_model_settings", lambda _: (None, []))
    monkeypatch.setattr(builder.LLMFactory, "create", lambda *_: object())
    monkeypatch.setattr(builder, "_build_middleware", lambda *_: [])
    monkeypatch.setattr(builder, "get_tools_for_node", lambda _: [])
    monkeypatch.setattr(builder, "get_type_hints", lambda *_, **__: {
        "status": str, "sql": str, "error": str})

    def create_agent(**kwargs):
        assert kwargs["name"] == "generator"
        response = kwargs["response_format"]
        result = {"structured_response": response(status=status, sql=sql, error=error)}
        assert set(response.model_json_schema()["required"]) == {"status", "sql", "error"}
        with pytest.raises(ValidationError):
            response(status=None, sql=sql, error=error)

        async def ainvoke(_):
            return result

        return SimpleNamespace(invoke=lambda _: result, ainvoke=ainvoke)

    monkeypatch.setattr(builder, "create_agent", create_agent)
    node = builder.AgentNodeBuilder.build(
        node_id="GenerateSQL", agent_id="generator", execution_mode=execution_mode)
    updates = node({}) if execution_mode == "sync" else asyncio.run(node({}))
    expected = {"status": status, "sql": sql, "error": error}
    assert updates == expected
    assert builder._build_state_updates({"structured_response": expected}, config) == expected
    assert builder._build_state_updates({"structured_response": {"status": status}}, config) == {
        "status": status, "sql": None, "error": None}
    with pytest.raises(ValueError, match="missing configured fields"):
        builder._build_state_updates({"structured_response": {}}, config)


@pytest.mark.parametrize('execution_mode', ['sync', 'async'])
@pytest.mark.parametrize('reduced', [False, True])
def test_optional_outputs_do_not_leak_between_real_graph_retries(
    monkeypatch, execution_mode, reduced,
):
    from operator import add
    from typing import Annotated, TypedDict

    from langgraph.graph import END, START, StateGraph

    from latest_ai_development.nodes import agent_node_builder as builder

    class TrialState(TypedDict, total=False):
        status: str
        sql: str | None
        rows: Annotated[list[str] | None, add] if reduced else list[str] | None
        error: str | None
        attempt: int
        observations: list[dict]

    config = {'role': 'Query', 'prompt': 'Return the actual current outcome.', 'reads_state': [],
              'output_mapping': {name: name for name in ('status', 'sql', 'rows', 'error')},
              'output_schema': {'status': {'type': 'string', 'required': True},
                                'sql': {'type': 'string', 'required': False},
                                'rows': {'type': 'array', 'items': {'type': 'string'},
                                         'required': False},
                                'error': {'type': 'string', 'required': False}}}
    monkeypatch.setattr(builder, 'State', TrialState)
    monkeypatch.setattr(builder, '_get_agent_config', lambda _: config)
    monkeypatch.setattr(builder, 'get_agent_model_settings', lambda _: (None, []))
    monkeypatch.setattr(builder.LLMFactory, 'create', lambda *_: object())
    monkeypatch.setattr(builder, '_build_middleware', lambda *_: [])
    monkeypatch.setattr(builder, 'get_tools_for_node', lambda _: [])
    outcomes = [
        dict(status='success', sql='first query', rows=['first'], error=None),
        dict(status='failed', sql=None, rows=None, error='Current failure'),
        dict(status='success', sql='new query', rows=['new'], error=None),
    ]
    pending = iter(outcomes)

    def create_agent(**kwargs):
        def invoke(_):
            return {'structured_response': kwargs['response_format'](**next(pending))}

        async def ainvoke(state):
            return invoke(state)

        return SimpleNamespace(invoke=invoke, ainvoke=ainvoke)

    monkeypatch.setattr(builder, 'create_agent', create_agent)
    agent = builder.AgentNodeBuilder.build(
        node_id='Generate', agent_id='generator', execution_mode=execution_mode)

    def observe(state):
        snapshot = {name: state.get(name) for name in config['output_mapping']}
        return {'observations': state.get('observations', []) + [snapshot],
                'attempt': state.get('attempt', 0) + 1}

    graph = StateGraph(TrialState)
    graph.add_node('Generate', agent)
    graph.add_node('Observe', observe)
    graph.add_edge(START, 'Generate')
    graph.add_edge('Generate', 'Observe')
    graph.add_conditional_edges(
        'Observe', lambda state: 'Generate' if state['attempt'] < 3 else END)
    compiled = graph.compile()
    result = (compiled.invoke({}) if execution_mode == 'sync'
              else asyncio.run(compiled.ainvoke({})))
    assert result['observations'] == outcomes
    assert result['rows'] == ['new']
    assert result['error'] is None


def test_optional_output_clearing_preserves_available_value_reducers(monkeypatch):
    from operator import add
    from typing import Annotated, TypedDict

    from langgraph.types import Overwrite

    from latest_ai_development.nodes import agent_node_builder as builder

    class TrialState(TypedDict):
        rows: Annotated[list[str], add]

    monkeypatch.setattr(builder, 'State', TrialState)
    config = {'output_mapping': {'rows': 'rows'},
              'output_schema': {'rows': {'type': 'array', 'required': False}}}
    result = builder._build_state_updates(
        {'structured_response': {'rows': ['new']}}, config, {'rows': ['previous']})
    assert result == {'rows': ['new']}  # Ordinary declared accumulation is preserved.
    cleared = builder._build_state_updates({'structured_response': {}}, config, {'rows': ['old']})
    assert isinstance(cleared['rows'], Overwrite)
    assert cleared['rows'].value is None
    fresh = builder._build_state_updates(
        {'structured_response': {'rows': ['new']}}, config, {'rows': None})
    assert isinstance(fresh['rows'], Overwrite)
    assert fresh['rows'].value == ['new']


def test_optional_object_clears_independently_of_nested_required_properties():
    from latest_ai_development.nodes import agent_node_builder as builder

    config = {'output_mapping': {'result': 'result'}, 'output_schema': {
        'type': 'object', 'properties': {'result': {
            'type': 'object', 'properties': {'total': {'type': 'number'}}, 'required': ['total']}},
        'required': []}}
    assert builder._build_state_updates({'structured_response': {}}, config) == {'result': None}


def test_agent_outputs_without_schema_remain_required():
    from latest_ai_development.nodes import agent_node_builder as builder
    config = {"output_mapping": {"sql": "sql"}}
    with pytest.raises(ValueError, match="missing configured fields"):
        builder._build_state_updates({"structured_response": {}}, config)


def test_langgraph_template_exposes_runtime_contract():
    workflow = LatestAiDevelopmentWorkflow()
    assert set(workflow.required_inputs()).issubset(workflow.input_fields())
    assert workflow.final_output_field in workflow.output_fields()


def test_langgraph_template_has_expected_graph_structure():
    graph_config = get_graph_config()

    nodes = set(graph_config["nodes"])
    assert graph_config["entry_point"] in nodes
    assert graph_config["final_output_field"] in OUTPUT_FIELDS

    for edge in graph_config.get("edges", []):
        assert edge["from"] in nodes
        assert edge["to"] == "END" or edge["to"] in nodes


def test_normalize_graph_config_preserves_explicit_final_output_field():
    graph_config = normalize_graph_config(
        {
            "graph": {
                "entry_node": "start",
                "final_output_field": "explicit_response",
                "nodes": [],
            }
        }
    )

    assert graph_config["final_output_field"] == "explicit_response"


def test_normalize_graph_config_infers_final_output_field_from_single_output_mapping():
    graph_config = normalize_graph_config(
        {
            "graph": {
                "entry_node": "start",
                "nodes": [],
                "output_mapping": {"response": "final_response"},
            }
        }
    )

    assert graph_config["final_output_field"] == "final_response"


def test_normalize_graph_config_infers_final_output_field_from_single_output_schema():
    graph_config = normalize_graph_config(
        {
            "graph": {
                "entry_node": "start",
                "nodes": [],
                "output_schema": {"final_response": {"type": "string"}},
            }
        }
    )

    assert graph_config["final_output_field"] == "final_response"


def test_agent_config_allows_autonomous_agent_without_reads_state(monkeypatch, isolated_state):
    monkeypatch.setattr(
        validators,
        "get_agents_config",
        lambda: {
            "agents": {
                "metadata_retrieval_agent": {
                    "role": "Metadata retrieval agent",
                    "provider": "openai",
                    "model": "gpt-4.1-mini",
                    "prompt": "Retrieve the latest metadata.",
                    "reads_state": [],
                    "writes_state": ["normalized_feedback"],
                    "output_field": "normalized_feedback",
                    "tools": [],
                    "mcp_servers": [],
                }
            }
        },
    )
    monkeypatch.setattr(
        validators,
        "get_models_config",
        lambda: {
            "default_provider": "openai",
            "default_model": "gpt-4.1-mini",
            "providers": {"openai": {"chat_model": "gpt-4.1-mini"}},
        },
    )
    monkeypatch.setattr(validators, "get_tools_config", lambda: {"tools": {}})
    monkeypatch.setattr(validators, "get_mcp_config", lambda: {"servers": {}})

    validators._validate_agents_config()


def test_agent_model_profile_overrides_and_fallback(monkeypatch):
    from types import SimpleNamespace

    from latest_ai_development.llm import llm_factory, model_settings

    monkeypatch.setattr(model_settings, "get_models_config", lambda: {
        "default_provider": "openai", "default_model": "configured-default",
        "providers": {"openai": {"chat_model": "configured-default",
                                  "fallback": {"chat_model": "configured-fallback"}}},
    })
    monkeypatch.setattr(model_settings, "get_settings", lambda: SimpleNamespace(
        openai_api_key="offline-placeholder", openai_api_key_secret=""))
    monkeypatch.setattr(model_settings, "SecretsManager", lambda: None)
    profile = {"primary": {"modelId": "gpt-4.1-mini", "generationDefaults": {
        "temperature": 0, "topP": 0.2, "maxOutputTokens": 5000,
        "presencePenalty": 0.4, "frequencyPenalty": -0.3,
    }}, "fallback": {"modelId": "gpt-4.1-nano", "generationDefaults": {"temperature": 0.1}}}
    primary, fallbacks = model_settings.get_agent_model_settings({"model": profile})
    assert primary.provider == "openai"
    assert primary.model == "gpt-4.1-mini"
    assert (primary.temperature, primary.top_p, primary.max_tokens) == (0, 0.2, 5000)
    assert fallbacks[0].model == "gpt-4.1-nano"
    assert fallbacks[0].provider == "openai"
    assert fallbacks[0].temperature == 0.1
    captured = {}
    monkeypatch.setattr(llm_factory, "ChatOpenAI", lambda **kwargs: captured.update(kwargs))
    llm_factory.LLMFactory().create(primary)
    assert captured["top_p"] == 0.2
    assert captured["presence_penalty"] == 0.4
    assert captured["frequency_penalty"] == -0.3
    assert captured["max_completion_tokens"] == 5000
    assert model_settings.get_agent_model_settings({"model": dict(profile, fallback=None)})[1] == []
    _, configured_fallbacks = model_settings.get_agent_model_settings({"model": "gpt-4.1-mini"})
    assert configured_fallbacks[0].model == "configured-fallback"


def test_nested_agent_model_contract_validation(monkeypatch, isolated_state):
    import pytest

    agent = {"role": "Responder", "prompt": "Respond.", "reads_state": [],
             "writes_state": ["final_response"], "output_field": "final_response",
             "model": {"primary": {"modelId": "gpt-4.1-mini", "generationDefaults": {}},
                       "fallback": {"modelId": "gpt-4.1-nano"}}}
    monkeypatch.setattr(validators, "get_agents_config", lambda: {"agents": {"responder": agent}})
    monkeypatch.setattr(validators, "get_models_config", lambda: {
        "default_provider": "openai", "default_model": "default",
        "providers": {"openai": {"chat_model": "default"}},
    })
    monkeypatch.setattr(validators, "get_tools_config", lambda: {"tools": {}})
    monkeypatch.setattr(validators, "get_mcp_config", lambda: {"servers": {}})
    validators._validate_agents_config()
    agent["model"]["primary"].pop("modelId")
    with pytest.raises(ValueError, match="primary.modelId is required"):
        validators._validate_agents_config()


@pytest.mark.parametrize("execution_mode", ["sync", "async"])
@pytest.mark.parametrize("outcome", [
    "primary_success", "fallback_success", "both_fail", "fallback_disabled",
])
def test_agent_node_executes_configured_fallback(monkeypatch, execution_mode, outcome):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from latest_ai_development.llm import model_settings
    from latest_ai_development.nodes import agent_node_builder as builder

    config = {
        "role": "Responder", "prompt": "Return an answer.", "reads_state": [],
        "writes_state": ["final_response"], "output_field": "final_response",
        "model": {
            "primary": {"modelId": "primary", "generationDefaults": {}},
            "fallback": {"modelId": "fallback", "generationDefaults": {"temperature": 0.1}},
        },
    }
    if outcome == "fallback_disabled":
        config["model"]["fallback"] = None
    monkeypatch.setattr(builder, "_get_agent_config", lambda _: config)
    monkeypatch.setattr(builder, "get_tools_for_node", lambda _: [])
    monkeypatch.setattr(model_settings, "get_models_config", lambda: {
        "default_provider": "openai", "default_model": "primary",
        "providers": {"openai": {"chat_model": "primary"}},
    })
    monkeypatch.setattr(model_settings, "get_settings", lambda: SimpleNamespace(
        openai_api_key="offline-placeholder", openai_api_key_secret=""))
    monkeypatch.setattr(model_settings, "SecretsManager", lambda: None)
    created = []

    def create_model(factory, settings):
        created.append(settings)
        return FakeListChatModel(responses=[settings.model])

    monkeypatch.setattr(builder.LLMFactory, "create", create_model)
    attempts = []
    original_call = FakeListChatModel._call

    def call(model, *args, **kwargs):
        name = model.responses[0]
        attempts.append(name)
        if outcome in {"both_fail", "fallback_disabled"} or (
            outcome == "fallback_success" and name == "primary"
        ):
            raise RuntimeError(f"{name} unavailable")
        return original_call(model, *args, **kwargs)

    monkeypatch.setattr(FakeListChatModel, "_call", call)
    node = builder.AgentNodeBuilder.build(
        node_id="Respond", agent_id="responder", execution_mode=execution_mode,
    )
    expected_models = [("openai", "primary")]
    if outcome != "fallback_disabled":
        expected_models.append(("openai", "fallback"))
        assert created[1].temperature == 0.1
    assert [(settings.provider, settings.model) for settings in created] == expected_models

    def invoke():
        return node({}) if execution_mode == "sync" else asyncio.run(node({}))

    if outcome in {"both_fail", "fallback_disabled"}:
        error = "fallback unavailable" if outcome == "both_fail" else "primary unavailable"
        with pytest.raises(RuntimeError, match=error):
            invoke()
    else:
        expected = "primary" if outcome == "primary_success" else "fallback"
        assert invoke() == {"final_response": expected}
    expected_attempts = ["primary"]
    if outcome in {"fallback_success", "both_fail"}:
        expected_attempts.append("fallback")
    assert attempts == expected_attempts


def test_normalization_preserves_checkpoint_settings_and_interaction_overrides():
    graph = {
        "entry_node": "Ask", "final_output_field": "final_response",
        "persistence": {"enabled": True, "backend": "memory"},
        "execution": {"resume_enabled": True},
        "runtime_context": {"defaults": {"region": "Europe"}},
        "nodes": [{"node_id": "Ask", "node_type": "agent", "agent_name": "asker",
                   "metadata": {"user_interaction": {
                       "kind": "approval", "prompt_field": "clarification_question",
                       "response_field": "human_decision"}}}],
    }
    normalized = normalize_graph_config({"graph": graph})
    for key in ("persistence", "execution", "runtime_context"):
        assert normalized[key] == graph[key]
    assert normalized["nodes"]["Ask"]["runtime_override"] is True


def test_interaction_wrapper_may_write_the_human_response(monkeypatch, isolated_state):
    def ask(state):
        return {"final_response": state["user_input"], "human_decision": "approve"}

    graph = normalize_graph_config({"graph": {
        "entry_node": "Ask", "end_nodes": ["Ask"], "final_output_field": "final_response",
        "nodes": [{"node_id": "Ask", "node_type": "agent", "agent_name": "asker",
                   "reads_state": ["user_input"],
                   "writes_state": ["final_response", "human_decision"],
                   "metadata": {"user_interaction": {
                       "kind": "approval", "prompt_field": "final_response",
                       "response_field": "human_decision"}}}],
        "edges": [], "conditional_edges": [],
    }})
    monkeypatch.setattr(validators, "get_graph_config", lambda: graph)
    monkeypatch.setattr(validators, "get_agents_config", lambda: {"agents": {
        "asker": {"reads_state": ["user_input"], "writes_state": ["final_response"]}}})
    monkeypatch.setattr(validators, "NODE_REGISTRY", {"Ask": ask})
    monkeypatch.setattr(validators, "ROUTER_REGISTRY", {})
    validators._validate_graph_config()


def test_graph_invocation_preserves_internal_initial_state(monkeypatch, isolated_state):
    import asyncio

    from latest_ai_development.graph import graph_builder

    def respond(state):
        return {"final_response": str(state["urgency_score"])}

    monkeypatch.setattr(graph_builder, "get_graph_config", lambda: {
        "entry_point": "Respond", "final_output_field": "final_response",
        "nodes": {"Respond": {"type": "deterministic"}},
        "edges": [{"from": "Respond", "to": "END"}],
        "conditional_edges": [],
    })
    monkeypatch.setattr(graph_builder, "NODE_REGISTRY", {"Respond": respond})
    graph = graph_builder.GraphBuilder.build()
    result = asyncio.run(graph.ainvoke({"user_input": "hello", "urgency_score": 0}))
    assert result["final_response"] == "0"


@pytest.mark.parametrize("legacy", [False, True])
def test_structured_outputs_preserve_nested_required_fields(legacy):
    from pydantic import ValidationError

    from latest_ai_development.nodes.agent_node_builder import _build_response_format

    properties = {
        "countries": {"type": "array", "items": {
            "type": "object", "properties": {"name": {"type": "string"}},
            "required": ["name"], "additionalProperties": False}},
        "sales": {"type": "object", "properties": {
            "rows": {"type": "array", "items": {"type": "object"}},
            "freshness": {"type": "string", "enum": ["current", "stale"]}},
            "required": ["rows", "freshness"]},
    }
    schema = ({key: {**value, "required": True} for key, value in properties.items()}
              if legacy else {"properties": properties, "required": list(properties)})
    # A legacy binding's required flag belongs to the binding, not the object shape.
    if legacy:
        schema["sales"]["required"] = properties["sales"]["required"]
    model = _build_response_format("Reference", {
        "output_mapping": {"geography": "countries", "analysis": "sales"},
        "output_schema": schema,
    })
    good = {"geography": [{"name": "India"}],
            "analysis": {"rows": [], "freshness": "current"}}
    assert model.model_validate(good).model_dump() == good
    for invalid in ({**good, "geography": [{}]},
                    {**good, "analysis": {"invented": "value"}},
                    {**good, "analysis": {"rows": [], "freshness": "invented"}}):
        with pytest.raises(ValidationError):
            model.model_validate(invalid)


@pytest.mark.parametrize("legacy", [False, True])
def test_structured_output_preserves_typed_dictionary_values(legacy):
    from pydantic import ValidationError

    from latest_ai_development.nodes.agent_node_builder import _build_response_format

    properties = {
        "revenue": {"type": "object", "additionalProperties": {"type": "number"},
                    "nullable": True},
        "regional_revenue": {"type": "object", "additionalProperties": {
            "type": "object", "additionalProperties": {"type": "number"}}},
    }
    schema = ({key: {**value, "required": True} for key, value in properties.items()}
              if legacy else {"properties": properties, "required": list(properties)})
    model = _build_response_format("Revenue", {
        "output_mapping": {key: key for key in properties}, "output_schema": schema,
    })
    good = {"revenue": {"A": 12.5}, "regional_revenue": {"Asia": {"A": 10.0}}}
    assert model.model_validate(good).model_dump() == good
    assert model.model_validate({**good, "revenue": None}).revenue is None
    assert model.model_validate({**good, "revenue": {}}).revenue == {}
    assert model.model_json_schema()["properties"]["revenue"]["anyOf"][0][
        "additionalProperties"] == {"type": "number"}
    for invalid in ({**good, "revenue": {"A": "invented"}},
                    {**good, "regional_revenue": {"Asia": {"A": "invented"}}},
                    {"regional_revenue": good["regional_revenue"]}):
        with pytest.raises(ValidationError):
            model.model_validate(invalid)


def test_structured_output_preserves_nullable_values_and_constraints():
    from pydantic import ValidationError

    from latest_ai_development.nodes.agent_node_builder import _build_response_format

    model = _build_response_format("Measured", {
        "output_mapping": {"result": "measurement"},
        "output_schema": {"properties": {"measurement": {
            "type": "object", "properties": {
                "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
                "status": {"type": "string", "enum": ["ok"], "nullable": True}},
            "required": ["confidence", "status"]}}, "required": ["measurement"]},
    })
    assert model.model_validate({"result": {"confidence": None, "status": None}}).model_dump() == {
        "result": {"confidence": None, "status": None}}
    with pytest.raises(ValidationError):
        model.model_validate({"result": {"confidence": 1.1, "status": "ok"}})


def test_agent_response_format_strips_typed_dict_presence_markers(monkeypatch):
    from typing import Any, NotRequired, Required

    from typing_extensions import TypedDict

    from latest_ai_development.nodes import agent_node_builder as builder

    class GeneratedState(TypedDict):
        geography: Required[list[dict[str, Any]]]
        analysis: NotRequired[dict[str, Any]]

    monkeypatch.setattr(builder, "State", GeneratedState)
    model = builder._build_response_format("Reference", {
        "output_mapping": {"countries": "geography", "sales": "analysis"},
    })
    assert model.model_validate({"countries": [{"name": "India"}], "sales": {}}).model_dump() == {
        "countries": [{"name": "India"}], "sales": {}}
