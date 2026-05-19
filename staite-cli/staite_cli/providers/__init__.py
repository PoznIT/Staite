"""Provider factory and public exports."""

import logging

from staite_cli.config.run_config import AnthropicConfig, AzureConfig, OllamaConfig
from staite_cli.providers.anthropic_provider import AnthropicProvider
from staite_cli.providers.azure_provider import AzureProvider
from staite_cli.providers.base import LLMProvider
from staite_cli.providers.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


def create_provider(
    provider_type: str,
    anthropic: AnthropicConfig = None,
    azure: AzureConfig = None,
    ollama: OllamaConfig = None
) -> LLMProvider:

    if provider_type == "anthropic":
        if not anthropic:
            raise ValueError("anthropic config is required when provider is anthropic")

        logger.debug("Creating AnthropicProvider (model=%s)", anthropic.model)
        return AnthropicProvider(model=anthropic.model, api_key=anthropic.api_key)

    if provider_type == "azure":
        if not azure:
            raise ValueError("azure config is required when provider is azure")
        logger.debug("Creating AzureProvider (model=%s, endpoint=%s)", azure.model, azure.endpoint)
        return AzureProvider(endpoint=azure.endpoint, model=azure.model, api_key=azure.api_key)

    if provider_type == "ollama":
        if not ollama:
            raise ValueError("ollama config is required when provider is 'ollama'")
        url = ollama.url or "http://localhost:11434"
        resolved_model = ollama.model
        logger.debug("Creating OllamaProvider (model=%s, url=%s)", resolved_model, url)
        return OllamaProvider(model=resolved_model, url=url, timeout=600)

    raise ValueError(f"Unknown provider: {provider_type!r}. Choose 'anthropic', 'azure', or 'ollama'.")


__all__ = ["LLMProvider", "AnthropicProvider", "AzureProvider", "OllamaProvider", "create_provider"]