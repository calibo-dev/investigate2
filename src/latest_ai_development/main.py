from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, NotRequired, Required, get_args, get_origin, get_type_hints
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from latest_ai_development.config.settings import get_graph_config, get_settings
from latest_ai_development.config.validators import validate_configuration
from latest_ai_development.graph.state import InputState, OutputState, State
from latest_ai_development.logging_utils import configure_logging
from latest_ai_development.tracking import initialize_langsmith_tracing
from latest_ai_development.workflow import LatestAiDevelopmentWorkflow


def _initialize_settings():
    runtime_settings = get_settings()
    configure_logging(getattr(runtime_settings, "log_level", "INFO"))
    return runtime_settings


settings = _initialize_settings()
logger = logging.getLogger(__name__)


def _validate_startup_configuration() -> None:
    try:
        validate_configuration()
    except Exception:
        logger.exception("Startup configuration validation failed")
        raise


def _initialize_workflow() -> LatestAiDevelopmentWorkflow:
    try:
        _validate_startup_configuration()
        return LatestAiDevelopmentWorkflow()
    except Exception:
        logger.exception("Workflow initialization failed")
        raise


API_ROOT_PATH = settings.normalized_api_root_path


def _is_resume_enabled(graph_config: Mapping[str, Any]) -> bool:
    execution = graph_config.get("execution", {})
    return isinstance(execution, Mapping) and execution.get("resume_enabled") is True


RESUME_ENABLED = _is_resume_enabled(get_graph_config())


@asynccontextmanager
async def lifespan(application: FastAPI):
    initialize_langsmith_tracing()
    runtime_workflow = _initialize_workflow()
    try:
        await runtime_workflow.astart()
        application.state.workflow = runtime_workflow
        yield
    finally:
        await runtime_workflow.aclose()
        if hasattr(application.state, "workflow"):
            del application.state.workflow


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Configuration-driven LangGraph workflow",
    root_path=API_ROOT_PATH,
    lifespan=lifespan,
)


class ResumeRequest(BaseModel):
    thread_id: str
    value: Any
    context: dict[str, Any] = Field(default_factory=dict)


def _model_annotation(annotation: Any) -> Any:
    """Remove TypedDict presence markers before creating a Pydantic field."""
    if get_origin(annotation) in {Required, NotRequired}:
        return get_args(annotation)[0]
    return annotation


def _build_api_model(
    name: str,
    state_type: type,
    *,
    required_fields: set[str],
    envelope_fields: dict[str, tuple[Any, Any]],
) -> type[BaseModel]:
    """Build an OpenAPI-visible model from the generated state contract."""
    # Pydantic accepts ``(annotation, default)`` field definitions dynamically.
    # Keep the mapping value typed as Any so static analyzers do not interpret
    # Pydantic's ``tuple[str, Any]`` overload as requiring string annotations.
    fields: dict[str, Any] = {
        field_name: (
            _model_annotation(annotation),
            ... if field_name in required_fields else None,
        )
        for field_name, annotation in get_type_hints(
            state_type,
            include_extras=True,
        ).items()
    }
    fields.update(envelope_fields)
    return create_model(
        name,
        __config__=ConfigDict(extra="ignore"),
        **fields,
    )


AskRequest = _build_api_model(
    "AskRequest",
    InputState,
    required_fields=set(InputState.__required_keys__),
    envelope_fields={
        "thread_id": (str | None, None),
        "context": (dict[str, Any], Field(default_factory=dict)),
    },
)

AskResponse = _build_api_model(
    "AskResponse",
    OutputState,
    required_fields=set(),
    envelope_fields={
        "thread_id": (str, ...),
        "status": (str, ...),
        "interrupts": (list[Any], Field(default_factory=list)),
    },
)


def _get_workflow(request: Request) -> LatestAiDevelopmentWorkflow:
    workflow = getattr(request.app.state, "workflow", None)
    if not isinstance(workflow, LatestAiDevelopmentWorkflow):
        raise RuntimeError("Workflow runtime is not initialized")
    return workflow


def _build_response(
    result: State,
    *,
    workflow: LatestAiDevelopmentWorkflow,
    thread_id: str,
) -> dict[str, Any]:
    raw_interrupts = result.get("__interrupt__", [])  # type: ignore[typeddict-item]
    interrupts = [getattr(item, "value", item) for item in raw_interrupts]
    response = {
        field: result[field]
        for field in workflow.output_fields()
        if field in result
    }
    response.update(
        thread_id=thread_id,
        status="interrupted" if interrupts else "completed",
        interrupts=interrupts,
    )
    return response


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
    }


async def ask(payload: BaseModel, http_request: Request) -> Any:
    payload_values = payload.model_dump(exclude_unset=True)
    flow_run_id = uuid4().hex[:12]
    thread_id = str(payload_values.get("thread_id") or flow_run_id)
    context = payload_values.get("context") or {}
    inputs = {
        key: value
        for key, value in payload_values.items()
        if key not in {"thread_id", "context"}
    }
    workflow = _get_workflow(http_request)
    missing = [field for field in workflow.required_inputs() if field not in inputs]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Request body is missing required workflow inputs",
                "required_inputs": list(workflow.required_inputs()),
                "missing_inputs": missing,
            },
        )
    logger.info(
        "Request started | flow_run_id=%s | thread_id=%s",
        flow_run_id,
        thread_id,
    )

    try:
        result = await workflow.akickoff(
            inputs,
            flow_run_id=flow_run_id,
            thread_id=thread_id,
            context=context or {},
        )
    except Exception:
        logger.exception(
            "Request failed | flow_run_id=%s | thread_id=%s",
            flow_run_id,
            thread_id,
        )
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "response": "Something went wrong. Please contact your administrator.",
            },
        )

    logger.info(
        "Request completed | flow_run_id=%s | thread_id=%s",
        flow_run_id,
        thread_id,
    )
    return _build_response(result, workflow=workflow, thread_id=thread_id)


ask.__annotations__["payload"] = AskRequest
app.post("/ask", response_model=AskResponse)(ask)


async def resume(request: ResumeRequest, http_request: Request) -> dict[str, Any]:
    workflow = _get_workflow(http_request)
    try:
        result = await workflow.aresume(
            thread_id=request.thread_id,
            value=request.value,
            context=request.context,
        )
    except Exception as exc:
        logger.exception("Workflow resume failed | thread_id=%s", request.thread_id)
        raise HTTPException(status_code=500, detail="Workflow resume failed") from exc
    return _build_response(result, workflow=workflow, thread_id=request.thread_id)


if RESUME_ENABLED:
    app.post("/resume", response_model=AskResponse)(resume)


async def _run_cli_workflow(inputs: dict[str, Any]) -> State:
    runtime_workflow = _initialize_workflow()
    await runtime_workflow.astart()
    try:
        return await runtime_workflow.akickoff(inputs)
    finally:
        await runtime_workflow.aclose()


def test() -> None:
    result = asyncio.run(_run_cli_workflow({"user_input": "Latest AI Agent Frameworks"}))
    print(json.dumps(result, ensure_ascii=False, default=str))


def run_with_trigger() -> None:
    if len(sys.argv) < 2:
        raise ValueError("Provide JSON payload")

    payload = json.loads(sys.argv[1])
    result = asyncio.run(_run_cli_workflow(payload))
    print(result)


def run() -> None:
    import uvicorn

    uvicorn.run(
        "latest_ai_development.main:app",
        host=settings.api_host,
        port=settings.port,
        reload=settings.debug,
        log_level=str(settings.log_level).lower(),
    )


if __name__ == "__main__":
    run()


if API_ROOT_PATH:
    app.add_api_route(
        f"{API_ROOT_PATH}/health",
        health,
        methods=["GET"],
        include_in_schema=False,
    )
    app.add_api_route(
        f"{API_ROOT_PATH}/ask",
        ask,
        methods=["POST"],
        include_in_schema=False,
    )
    if RESUME_ENABLED:
        app.add_api_route(
            f"{API_ROOT_PATH}/resume",
            resume,
            methods=["POST"],
            include_in_schema=False,
        )
