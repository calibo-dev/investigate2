from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from latest_ai_development.llm.model_settings import (
    ModelSettings,
    get_fallback_model_settings,
    get_model_settings,
)

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    Create provider-specific LangChain chat models.
    Provider and model config are driven by centralized settings + models.yaml.

    IMPORTANT FOR AI / FUTURE EDITS:
    - Add new providers in src/latest_ai_development/config/models.yaml under `providers:`.
    - Do not create LLMs directly inside steps.
    - Steps should call LLMFactory().create_llm().
    """

    def create_llm(
        self,
        provider: str,
        *,
        model: str,
    ) -> BaseChatModel:
        """
        Resolve settings from centralized config and create the model.

        Supports:
        - OpenAI
        - Anthropic
        - Gemini
        - Bedrock
        - Ollama
        """
        settings = get_model_settings(
            provider_name=provider,
            model_name=model,
        )
        return self.create(settings)

    def create(self, model_settings: ModelSettings) -> BaseChatModel:
        """Build a chat model from an already-resolved ModelSettings instance."""
        provider = model_settings.provider
        logger.info(
            "Creating chat model | provider=%s | model=%s",
            provider,
            model_settings.model,
        )
        logger.debug(
            "Chat model settings | provider=%s | temperature=%s | max_tokens=%s | extra_keys=%s",
            provider,
            model_settings.temperature,
            model_settings.max_tokens,
            sorted(model_settings.extra.keys()),
        )

        if provider == "openai":
            if model_settings.model.startswith("gpt-5"):
                return ChatOpenAI(
                    model=model_settings.model,
                    api_key=model_settings.extra.get("api_key"),
                    max_completion_tokens=model_settings.max_tokens,
                )

            return ChatOpenAI(
                model=model_settings.model,
                temperature=model_settings.temperature,
                top_p=model_settings.top_p,
                api_key=model_settings.extra.get("api_key"),
                max_completion_tokens=model_settings.max_tokens,
                **{
                    key: model_settings.extra[key]
                    for key in ("presence_penalty", "frequency_penalty")
                    if key in model_settings.extra
                },
            )

        if provider == "anthropicai":
            try:
                from langchain_anthropic import ChatAnthropic
            except ImportError as exc:
                raise ImportError(
                    "Anthropic provider requires the 'langchain-anthropic' package. "
                    "Run `uv sync` to install template dependencies."
                ) from exc

            return ChatAnthropic(
                model_name=model_settings.model,
                temperature=model_settings.temperature,
                top_p=model_settings.top_p,
                api_key=model_settings.extra.get("api_key"),
                max_tokens_to_sample=model_settings.max_tokens,
                timeout=None,
                stop=None,
            )

        if provider == "geminiai":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                raise ImportError(
                    "Gemini provider requires the 'langchain-google-genai' package. "
                    "Run `uv sync` to install template dependencies."
                ) from exc

            return ChatGoogleGenerativeAI(
                model=model_settings.model,
                temperature=model_settings.temperature,
                top_p=model_settings.top_p,
                api_key=model_settings.extra.get("api_key"),
                max_tokens=model_settings.max_tokens,
            )

        if provider == "bedrockai":
            try:
                from langchain_aws import ChatBedrockConverse
            except ImportError as exc:
                raise ImportError(
                    "Bedrock provider requires the 'langchain-aws' package. "
                    "Run `uv sync` to install template dependencies."
                ) from exc

            return ChatBedrockConverse(
                model=model_settings.model,
                temperature=model_settings.temperature,
                top_p=model_settings.top_p,
                api_key=model_settings.extra.get("api_key"),
                max_tokens=model_settings.max_tokens,
                region_name=model_settings.extra.get("region_name"),
            )

        if provider == "microsoftfoundryai":
            api_key: str = model_settings.extra["api_key"]
            if model_settings.extra.get("api_family") == "anthropic_messages":
                try:
                    from langchain_anthropic import ChatAnthropic
                except ImportError as exc:
                    raise ImportError(
                        "Microsoft Foundry Anthropic models require the "
                        "'langchain-anthropic' package. Run `uv sync` to install "
                        "template dependencies."
                    ) from exc

                return ChatAnthropic(
                    **_foundry_kwargs(model_settings, api_key),
                    default_headers={"api-key": api_key},
                )

            return ChatOpenAI(
                **_foundry_kwargs(model_settings, api_key),
                use_responses_api=True,
                output_version="responses/v1",
            )

        if provider == "ollama":
            return ChatOllama(
                model=model_settings.model,
                temperature=model_settings.temperature,
                top_p=model_settings.top_p,
                num_predict=model_settings.max_tokens,
                base_url=model_settings.extra.get("base_url"),
            )

        raise ValueError(f"Unsupported provider: {provider}")


def _foundry_kwargs(model_settings: ModelSettings, api_key: str) -> dict[str, Any]:
    return {
        "model": model_settings.model,
        "api_key": api_key,
        "base_url": model_settings.extra.get("base_url"),
        "max_tokens": model_settings.max_tokens,
    }


def get_fallback_llms(provider: str) -> list[BaseChatModel]:
    factory = LLMFactory()
    return [factory.create(settings) for settings in get_fallback_model_settings(provider)]
