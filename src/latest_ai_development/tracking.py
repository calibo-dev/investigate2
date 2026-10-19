from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from functools import wraps
from inspect import isawaitable, iscoroutinefunction
from typing import Any

from dotenv import load_dotenv

from latest_ai_development.config.settings import get_graph_config, get_settings
from latest_ai_development.graph.state import INPUT_FIELDS, OUTPUT_FIELDS
from latest_ai_development.secrets_manager import SecretsManager

load_dotenv()

if (
    not os.getenv("LANGSMITH_API_KEY", "").strip()
    and not os.getenv("LANGCHAIN_API_KEY", "").strip()
):
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

traceable: Any
get_current_run_tree: Any
try:
    from langsmith import get_current_run_tree
    from langsmith import traceable as langsmith_traceable

    traceable = langsmith_traceable
except ImportError:  # pragma: no cover
    traceable = None
    get_current_run_tree = None

logger = logging.getLogger("uvicorn.error")

_LANGSMITH_INITIALIZED = False


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def get_tracing_backend() -> str:
    configured_backend = os.getenv("TRACING_BACKEND")
    if configured_backend is None and _truthy(os.getenv("LANGSMITH_TRACING")):
        value = "LANGSMITH"
    else:
        value = (configured_backend or "NONE").strip().upper() or "NONE"

    if value in {"OFF", "DISABLED"}:
        value = "NONE"

    logger.info("Tracing backend: %s", value)
    return value


def _resolve_langsmith_api_key() -> str | None:
    settings = get_settings()

    api_key = settings.langsmith_api_key.strip()
    if api_key:
        return api_key

    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if api_key:
        return api_key

    langchain_api_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
    if langchain_api_key:
        return langchain_api_key

    secret_name = settings.langsmith_api_key_secret.strip()
    if not secret_name:
        secret_name = (
            os.getenv("LANGSMITH_API_KEY_SECRET", "").strip()
            or os.getenv("LANGCHAIN_API_KEY_SECRET", "").strip()
        )
        if not secret_name:
            return None

    secret_value = SecretsManager().get_secret(secret_name)
    return secret_value.strip() if secret_value else None


def initialize_langsmith_tracing() -> None:
    global _LANGSMITH_INITIALIZED

    tracing_backend = get_tracing_backend()
    if tracing_backend != "LANGSMITH":
        if tracing_backend == "NONE":
            os.environ["LANGSMITH_TRACING"] = "false"
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled")
        return

    if _LANGSMITH_INITIALIZED:
        return

    try:
        settings = get_settings()
        api_key = _resolve_langsmith_api_key()
        if not api_key:
            raise ValueError(
                "TRACING_BACKEND is LANGSMITH but LANGSMITH_API_KEY, "
                "LANGCHAIN_API_KEY, LANGSMITH_API_KEY_SECRET, or "
                "LANGCHAIN_API_KEY_SECRET is not configured."
            )

        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

        project = settings.langsmith_project.strip() or os.getenv("LANGSMITH_PROJECT", "").strip()
        if project:
            os.environ["LANGSMITH_PROJECT"] = project

        endpoint = (
            settings.langsmith_endpoint.strip() or os.getenv("LANGSMITH_ENDPOINT", "").strip()
        )
        if endpoint:
            os.environ["LANGSMITH_ENDPOINT"] = endpoint

        logger.info("LangSmith tracing enabled | project=%s", project or "not set")
        _LANGSMITH_INITIALIZED = True
    except Exception as exc:
        os.environ["TRACING_BACKEND"] = "NONE"
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.warning(
            "LangSmith tracing initialization failed; falling back to no tracing | error=%s",
            exc,
        )


def trace_workflow() -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _lazy_traceable(lambda: _workflow_trace_spec())


async def atrace_workflow_execution(
    invoke: Callable[[Any], Any],
    *,
    flow_run_id: str,
    inputs: dict[str, Any],
    initial_state: Any,
) -> Any:
    @trace_workflow()
    async def _invoke(
        *,
        flow_run_id: str,
        inputs: dict[str, Any],
        initial_state: Any,
    ) -> Any:
        result = invoke(initial_state)
        return await result if isawaitable(result) else result

    return await _invoke(
        flow_run_id=flow_run_id,
        inputs=inputs,
        initial_state=initial_state,
    )


def _lazy_traceable(
    spec_factory: Callable[[], dict[str, Any]],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                traced_func = _traceable_from_spec(spec_factory())(func)
                return await traced_func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            traced_func = _traceable_from_spec(spec_factory())(func)
            return traced_func(*args, **kwargs)

        return wrapper

    return decorator


def _traceable_from_spec(
    spec: dict[str, Any],
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if traceable is None:
        return lambda func: func

    if get_tracing_backend() != "LANGSMITH":
        return lambda func: func

    if not spec.get("enabled", True):
        return lambda func: func

    trace_decorator = traceable(
        name=spec.get("name") or "Trace",
        run_type=spec.get("run_type", "chain"),
        tags=list(spec.get("tags", [])),
        metadata=dict(spec.get("metadata", {})),
        process_inputs=_build_field_processor(spec.get("input_fields")),
        process_outputs=_build_field_processor(spec.get("output_fields")),
    )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        if iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    result = await func(*args, **kwargs)
                except Exception as exc:
                    _record_trace_outcome(exc)
                    raise
                _record_trace_outcome()
                return result

            return trace_decorator(async_wrapper)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                _record_trace_outcome(exc)
                raise
            _record_trace_outcome()
            return result

        return trace_decorator(wrapper)

    return decorator


def _record_trace_outcome(error: Exception | None = None) -> None:
    run = get_current_run_tree() if get_current_run_tree is not None else None
    if run is None:
        return
    metadata = {"execution_status": "success" if error is None else "failed"}
    if error is not None:
        category = next((label for kind, label in (
            (TimeoutError, "timeout"), (PermissionError, "permission"),
            (ConnectionError, "connection"), (ValueError, "validation"),
        ) if isinstance(error, kind)), "execution")
        metadata.update(error_type=type(error).__name__, error_category=category)
    run.add_metadata(metadata)


def _get_tracing_config() -> dict[str, Any]:
    graph_cfg = get_graph_config()
    tracing_cfg = graph_cfg.get("tracing", {})
    return tracing_cfg if isinstance(tracing_cfg, dict) else {}


def _get_named_tracing_spec(section_name: str, group_name: str) -> dict[str, Any]:
    tracing_cfg = _get_tracing_config()

    direct_section = tracing_cfg.get(section_name)
    if isinstance(direct_section, dict):
        return direct_section

    group = tracing_cfg.get(group_name, {})
    if isinstance(group, dict):
        section = group.get(section_name, {})
        return section if isinstance(section, dict) else {}

    return {}


def _workflow_trace_spec() -> dict[str, Any]:
    config_spec = _get_named_tracing_spec("workflow", "workflows")
    return _normalize_trace_spec(
        _default_workflow_trace_spec(),
        config_spec,
        fallback_name="LangGraph Workflow",
    )


def _default_workflow_trace_spec() -> dict[str, Any]:
    graph_config = get_graph_config()
    workflow_name = graph_config.get("graph_id") or "LangGraph Workflow"
    return {
        "enabled": True,
        "name": workflow_name,
        "run_type": "chain",
        "tags": ["workflow", "langgraph"],
        "metadata": {
            "workflow_name": workflow_name,
            "component_type": "workflow",
        },
        "input_fields": {
            "flow_run_id": "flow_run_id",
            **{field: f"inputs.{field}" for field in INPUT_FIELDS},
            "initial_state_keys": {"path": "initial_state", "transform": "keys"},
        },
        "output_fields": {field: field for field in OUTPUT_FIELDS},
    }


def _normalize_trace_spec(
    base_spec: dict[str, Any],
    override_spec: dict[str, Any],
    *,
    fallback_name: str,
) -> dict[str, Any]:
    normalized = dict(base_spec)
    normalized.setdefault("name", fallback_name)

    if "name" in override_spec:
        normalized["name"] = override_spec["name"]

    if "enabled" in override_spec:
        normalized["enabled"] = override_spec["enabled"]

    if "run_type" in override_spec:
        normalized["run_type"] = override_spec["run_type"]

    if "tags" in override_spec:
        normalized["tags"] = list(override_spec["tags"])

    if "metadata" in override_spec:
        base_metadata = normalized.get("metadata", {})
        override_metadata = override_spec["metadata"]
        if isinstance(base_metadata, Mapping) and isinstance(override_metadata, Mapping):
            normalized["metadata"] = _merge_dicts(base_metadata, override_metadata)
        else:
            normalized["metadata"] = override_metadata

    if "input_fields" in override_spec:
        normalized["input_fields"] = override_spec["input_fields"]

    if "output_fields" in override_spec:
        normalized["output_fields"] = override_spec["output_fields"]

    for key, value in override_spec.items():
        if key in {
            "name",
            "enabled",
            "run_type",
            "tags",
            "metadata",
            "input_fields",
            "output_fields",
        }:
            continue
        normalized[key] = value

    return normalized


def _merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def _resolve_path(payload: Any, path: str | None) -> Any:
    if path is None or path == "":
        return payload

    current: Any = payload
    for segment in path.split("."):
        if segment in {"", "."}:
            continue

        if isinstance(current, Mapping):
            if segment not in current:
                return None
            current = current[segment]
            continue

        if hasattr(current, segment):
            current = getattr(current, segment)
            continue

        return None

    return current


def _apply_transform(value: Any, transform: str | None) -> Any:
    if transform in (None, "", "identity"):
        return value

    if transform in {"bool", "exists"}:
        return bool(value)

    if transform == "len":
        if value is None:
            return 0
        try:
            return len(value)  # type: ignore[arg-type]
        except TypeError:
            return len(str(value))

    if transform == "str":
        return "" if value is None else str(value)

    if transform == "type":
        return type(value).__name__

    if transform == "keys":
        if isinstance(value, Mapping):
            return sorted(value.keys())
        keys = getattr(value, "keys", None)
        if callable(keys):
            return sorted(list(keys()))
        return []

    raise ValueError(f"Unsupported trace transform: {transform}")


def _normalize_field_spec(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"path": value}

    if isinstance(value, Mapping):
        return dict(value)

    raise TypeError(f"Trace field spec must be a string or mapping, got {type(value).__name__}")


def _build_field_processor(field_specs: Any) -> Callable[[Any], dict[str, Any]]:
    specs = field_specs if isinstance(field_specs, Mapping) else {}
    normalized_specs = {name: _normalize_field_spec(spec) for name, spec in specs.items()}

    def processor(payload: Any) -> dict[str, Any]:
        # An exception has no returned state. Do not fabricate output values or counts.
        if payload is None:
            return {}
        if not normalized_specs:
            if isinstance(payload, Mapping):
                return {"keys": sorted(payload.keys())}
            return {"output_type": type(payload).__name__}

        processed: dict[str, Any] = {}
        for field_name, spec in normalized_specs.items():
            raw_value = _resolve_path(payload, spec.get("path"))
            default_value = spec.get("default")
            value = raw_value if raw_value is not None else default_value
            processed[field_name] = _apply_transform(value, spec.get("transform"))
        return processed

    return processor


__all__ = [
    "atrace_workflow_execution",
    "get_tracing_backend",
    "initialize_langsmith_tracing",
    "trace_workflow",
]
