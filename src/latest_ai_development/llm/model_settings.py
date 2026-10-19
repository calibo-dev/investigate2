from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from latest_ai_development.config.settings import (
    get_models_config,
    get_settings,
    normalize_provider_name,
)
from latest_ai_development.secrets_manager import SecretsManager


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    extra: dict[str, Any] = field(default_factory=dict)


def _resolve_api_key(
    *,
    direct_value: str,
    secret_name: str,
    secrets_manager: SecretsManager,
) -> str | None:
    direct_value = direct_value.strip()
    if direct_value:
        return direct_value

    secret_name = secret_name.strip()
    if secret_name:
        return secrets_manager.get_secret(secret_name)

    return None


def _resolve_required_api_key(
    *,
    direct_value: str,
    secret_name: str,
    secrets_manager: SecretsManager,
) -> str:
    api_key = _resolve_api_key(
        direct_value=direct_value,
        secret_name=secret_name,
        secrets_manager=secrets_manager,
    )
    if not api_key:
        raise ValueError("API key not configured for the selected provider.")
    return api_key


def _resolve_microsoft_foundry_ai_endpoint(endpoint: str) -> dict[str, str]:
    if endpoint.endswith("/anthropic"):
        return {"api_family": "anthropic_messages", "base_url": endpoint}
    if endpoint.endswith("/openai"):
        return {"api_family": "openai_responses","base_url": f"{endpoint}/v1"}
    raise ValueError("MICROSOFT_FOUNDRY_AI_ENDPOINT must end with /anthropic or /openai.")


def _get_provider_section_config(
    providers: dict[str, Any],
    *,
    provider: str,
    section: str,
) -> tuple[str, dict[str, Any]]:
    provider_cfg = providers.get(provider)
    if not isinstance(provider_cfg, dict):
        raise ValueError(f"Provider '{provider}' not configured in models.yaml")

    if section == "primary":
        return provider, provider_cfg

    if section != "fallback":
        raise ValueError(f"Unsupported model settings section '{section}'.")

    fallback_cfg = provider_cfg.get("fallback")
    if not fallback_cfg:
        raise ValueError(f"Provider '{provider}' does not define a fallback model in models.yaml")
    if not isinstance(fallback_cfg, dict):
        raise ValueError(f"Provider '{provider}' fallback must be a mapping in models.yaml")

    fallback_provider = normalize_provider_name(fallback_cfg.get("provider") or provider)
    merged_cfg = dict(fallback_cfg)
    merged_cfg.pop("provider", None)
    return fallback_provider, merged_cfg


def get_model_settings(
    provider_name: str | None = None,
    *,
    section: str = "primary",
    model_name: str | None = None,
    generation_defaults: dict[str, Any] | None = None,
) -> ModelSettings:
    settings = get_settings()
    models_cfg = get_models_config()
    secrets_manager = SecretsManager()

    provider = normalize_provider_name(
        provider_name or models_cfg.get("default_provider")
    )
    providers = models_cfg.get("providers", {})
    resolved_provider, provider_cfg = _get_provider_section_config(
        providers,
        provider=provider,
        section=section,
    )

    resolved_model_name = str(
        (
            model_name
            or provider_cfg.get("chat_model")
            or models_cfg.get("default_model")
        )
        if section == "primary"
        else provider_cfg.get("chat_model", "")
    ).strip()
    if not resolved_model_name:
        source = "models.yaml defaults" if section == "primary" else "models.yaml fallback"
        raise ValueError(f"Model name for provider '{resolved_provider}' is missing from {source}")

    if generation_defaults is not None and not isinstance(generation_defaults, dict):
        raise ValueError("generationDefaults must be a mapping when provided")
    defaults = generation_defaults or {}
    extra: dict[str, Any] = {}
    for source, target in (("presencePenalty", "presence_penalty"),
                           ("frequencyPenalty", "frequency_penalty")):
        if source in defaults:
            extra[target] = defaults[source]

    if resolved_provider == "openai":
        api_key = _resolve_required_api_key(
            direct_value=settings.openai_api_key,
            secret_name=settings.openai_api_key_secret,
            secrets_manager=secrets_manager,
        )
        extra["api_key"] = api_key

    elif resolved_provider == "anthropicai":
        api_key = _resolve_required_api_key(
            direct_value=settings.anthropicai_api_key,
            secret_name=settings.anthropicai_api_key_secret,
            secrets_manager=secrets_manager,
        )
        extra["api_key"] = api_key

    elif resolved_provider == "geminiai":
        api_key = _resolve_required_api_key(
            direct_value=settings.geminiai_api_key,
            secret_name=settings.geminiai_api_key_secret,
            secrets_manager=secrets_manager,
        )
        extra["api_key"] = api_key

    elif resolved_provider == "bedrockai":
        bedrock_api_key = _resolve_api_key(
            direct_value=settings.bedrock_ai_api_key,
            secret_name=settings.bedrock_ai_api_key_secret,
            secrets_manager=secrets_manager,
        )
        region_name = os.getenv("AWS_REGION_NAME", "").strip()
        if not bedrock_api_key:
            raise ValueError(
                "BedrockAI API key not found. "
                "Set BEDROCK_AI_API_KEY or BEDROCK_AI_API_KEY_SECRET."
            )
        if not region_name:
            raise ValueError(
                "AWS_REGION_NAME is required for the BEDROCKAI provider. "
                "Configure a region where Amazon Bedrock is available."
            )
        extra["api_key"] = bedrock_api_key
        extra["region_name"] = region_name

    elif resolved_provider == "microsoftfoundryai":
        api_key = _resolve_required_api_key(
            direct_value=settings.microsoft_foundry_ai_api_key,
            secret_name=settings.microsoft_foundry_ai_api_key_secret,
            secrets_manager=secrets_manager,
        )
        endpoint = settings.microsoft_foundry_ai_endpoint.strip()
        if not endpoint:
            raise ValueError(
                "MICROSOFT_FOUNDRY_AI_ENDPOINT is required for Microsoft Foundry AI."
            )
        endpoint_settings = _resolve_microsoft_foundry_ai_endpoint(endpoint)
        extra["api_key"] = api_key
        extra["api_family"] = endpoint_settings["api_family"]
        extra["base_url"] = endpoint_settings["base_url"]

    elif resolved_provider == "ollama":
        extra["base_url"] = settings.ollama_base_url

    else:
        supported_providers = sorted(providers.keys())
        raise ValueError(
            f"Unsupported provider '{resolved_provider}'. "
            f"Supported providers: {supported_providers}"
        )

    return ModelSettings(
        provider=resolved_provider,
        model=resolved_model_name,
        temperature=float(defaults.get("temperature", provider_cfg.get("temperature", 0.3))),
        top_p=float(defaults.get("topP", provider_cfg.get("top_p", 1.0))),
        max_tokens=int(defaults.get("maxOutputTokens", provider_cfg.get("max_tokens", 2000))),
        extra=extra,
    )


def get_agent_model_settings(
    agent_config: dict[str, Any],
) -> tuple[ModelSettings, list[ModelSettings]]:
    """Apply approved per-agent overrides over the configured provider defaults."""
    model_config = agent_config.get("model")
    if isinstance(model_config, dict):
        primary_config = model_config.get("primary")
        if not isinstance(primary_config, dict):
            raise ValueError("Agent model.primary must be a mapping")
        if (not isinstance(primary_config.get("modelId"), str)
                or not primary_config["modelId"].strip()):
            raise ValueError("Agent model.primary.modelId is required")
    else:
        # Existing starter agents use scalar provider/model settings.
        primary_config = agent_config
        model_config = agent_config
    primary = get_model_settings(
        primary_config.get("provider"),
        model_name=primary_config.get("modelId", primary_config.get("model")),
        generation_defaults=primary_config.get("generationDefaults"),
    )
    if "fallback" not in model_config:
        return primary, get_fallback_model_settings(primary.provider)
    fallback = model_config["fallback"]
    if fallback is None:
        return primary, []
    if not isinstance(fallback, dict):
        raise ValueError("Agent fallback must be a mapping or null")
    fallback_model = fallback.get("modelId", fallback.get("model"))
    if not isinstance(fallback_model, str) or not fallback_model.strip():
        raise ValueError("Agent model.fallback.modelId is required")
    return primary, [get_model_settings(
        fallback.get("provider") or primary.provider,
        model_name=fallback_model,
        generation_defaults=fallback.get("generationDefaults"),
    )]


def get_fallback_model_settings(provider_name: str) -> list[ModelSettings]:
    models_cfg = get_models_config()
    provider = normalize_provider_name(provider_name)
    providers = models_cfg.get("providers", {})
    provider_cfg = providers.get(provider)

    if not isinstance(provider_cfg, dict):
        raise ValueError(f"Provider '{provider}' not configured in models.yaml")

    if not provider_cfg.get("fallback"):
        return []

    return [get_model_settings(provider, section="fallback")]
