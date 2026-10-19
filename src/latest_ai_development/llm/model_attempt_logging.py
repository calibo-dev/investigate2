from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelFallbackMiddleware

from latest_ai_development.llm.model_settings import ModelSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelAttempt:
    stage: str
    provider: str
    model_id: str


def model_attempt_from_settings(stage: str, settings: ModelSettings) -> ModelAttempt:
    return ModelAttempt(
        stage=stage,
        provider=settings.provider,
        model_id=settings.model,
    )


def get_model_attempts(
    stage: str,
    primary_settings: ModelSettings,
    fallback_settings: list[ModelSettings],
) -> tuple[ModelAttempt, list[ModelAttempt]]:
    primary_attempt = model_attempt_from_settings(stage, primary_settings)
    fallback_attempts = [
        model_attempt_from_settings(stage, settings) for settings in fallback_settings
    ]
    return primary_attempt, fallback_attempts


class ModelAttemptLoggingMiddleware(AgentMiddleware):
    def __init__(self, primary_attempt: ModelAttempt) -> None:
        super().__init__()
        self.primary_attempt = primary_attempt

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        log_model_try(self.primary_attempt)
        try:
            response = handler(request)
        except Exception as exc:
            log_model_failure(self.primary_attempt, exc)
            raise

        log_model_success(self.primary_attempt)
        return response

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        log_model_try(self.primary_attempt)
        try:
            response = await handler(request)
        except Exception as exc:
            log_model_failure(self.primary_attempt, exc)
            raise

        log_model_success(self.primary_attempt)
        return response


class LoggedModelFallbackMiddleware(ModelFallbackMiddleware):
    def __init__(
        self,
        primary_attempt: ModelAttempt,
        fallback_attempts: list[ModelAttempt],
        first_model: Any,
        *additional_models: Any,
    ) -> None:
        super().__init__(first_model, *additional_models)
        self.primary_attempt = primary_attempt
        self.fallback_attempts = fallback_attempts

    def wrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        last_exception: Exception

        log_model_try(self.primary_attempt)
        try:
            response = handler(request)
        except Exception as exc:
            log_model_failure(self.primary_attempt, exc)
            last_exception = exc
        else:
            log_model_success(self.primary_attempt)
            return response

        for index, fallback_model in enumerate(self.models):
            attempt = self.fallback_attempts[index]
            log_model_retry(attempt)
            try:
                response = handler(request.override(model=fallback_model))
            except Exception as exc:
                log_model_failure(attempt, exc)
                last_exception = exc
                continue

            log_model_success(attempt)
            return response

        raise last_exception

    async def awrap_model_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        last_exception: Exception

        log_model_try(self.primary_attempt)
        try:
            response = await handler(request)
        except Exception as exc:
            log_model_failure(self.primary_attempt, exc)
            last_exception = exc
        else:
            log_model_success(self.primary_attempt)
            return response

        for index, fallback_model in enumerate(self.models):
            attempt = self.fallback_attempts[index]
            log_model_retry(attempt)
            try:
                response = await handler(request.override(model=fallback_model))
            except Exception as exc:
                log_model_failure(attempt, exc)
                last_exception = exc
                continue

            log_model_success(attempt)
            return response

        raise last_exception


def log_model_try(attempt: ModelAttempt) -> None:
    logger.info(
        "Trying model | stage=%s | provider=%s | modelId=%s",
        attempt.stage,
        attempt.provider,
        attempt.model_id,
    )


def log_model_retry(attempt: ModelAttempt) -> None:
    logger.info(
        "Retrying with model | stage=%s | provider=%s | modelId=%s",
        attempt.stage,
        attempt.provider,
        attempt.model_id,
    )


def log_model_success(attempt: ModelAttempt) -> None:
    logger.info(
        "Model succeeded | stage=%s | provider=%s | modelId=%s",
        attempt.stage,
        attempt.provider,
        attempt.model_id,
    )


def log_model_failure(attempt: ModelAttempt, error: Any) -> None:
    logger.warning(
        "Model failed | stage=%s | provider=%s | modelId=%s | error=%s",
        attempt.stage,
        attempt.provider,
        attempt.model_id,
        _format_error(error),
    )


def _format_error(error: Any) -> str:
    if error is None:
        return "unknown"

    if isinstance(error, BaseException):
        return f"{type(error).__name__}: {error}"

    return str(error)
