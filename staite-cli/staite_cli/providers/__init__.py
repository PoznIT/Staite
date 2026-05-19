"""Provider factory and public exports."""

import logging

from staite_cli.providers.anthropic_provider import AnthropicProvider
from staite_cli.providers.azure_provider import AzureProvider
from staite_cli.providers.base import LLMProvider
from staite_cli.providers.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


def create_provider(
    provider_type: str,
    model: str,
    anthropic_api_key: str | None = None,
    azure_endpoint: str | None = None,
    azure_api_key: str | None = None,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
) -> LLMProvider:

    if provider_type == "anthropic":
        logger.debug("Creating AnthropicProvider (model=%s)", model)
        return AnthropicProvider(model=model, api_key=anthropic_api_key)

    if provider_type == "azure":
        if not azure_endpoint:
            raise ValueError("azure.endpoint is required when provider is 'azure'")
        logger.debug("Creating AzureProvider (model=%s, endpoint=%s)", model, azure_endpoint)
        return AzureProvider(endpoint=azure_endpoint, model=model, api_key=azure_api_key)

    if provider_type == "ollama":
        url = ollama_url or "http://localhost:11434"
        resolved_model = ollama_model or model
        logger.debug("Creating OllamaProvider (model=%s, url=%s)", resolved_model, url)
        return OllamaProvider(model=resolved_model, url=url, timeout=600)

    raise ValueError(f"Unknown provider: {provider_type!r}. Choose 'anthropic', 'azure', or 'ollama'.")


__all__ = ["LLMProvider", "AnthropicProvider", "AzureProvider", "OllamaProvider", "create_provider"]