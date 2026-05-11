"""Provider factory and public exports."""

import logging

from staite.providers.anthropic_provider import AnthropicProvider
from staite.providers.azure_provider import AzureProvider
from staite.providers.base import LLMProvider

logger = logging.getLogger(__name__)


def create_provider(
    provider_type: str,
    model: str,
    azure_endpoint: str | None = None,
    azure_api_key: str | None = None,
) -> LLMProvider:
    """Instantiate the correct LLMProvider from config values.

    Args:
        provider_type: ``"anthropic"`` or ``"azure"``.
        model: Model name / ID.
        azure_endpoint: Required when provider_type is ``"azure"``.
        azure_api_key: Optional API key for Azure (falls back to env / DefaultAzureCredential).

    Returns:
        An uninitialised LLMProvider (use as async context manager before calling complete).

    Raises:
        ValueError: If provider_type is unknown or azure_endpoint is missing for Azure.
    """
    if provider_type == "anthropic":
        logger.debug("Creating AnthropicProvider (model=%s)", model)
        return AnthropicProvider(model=model)

    if provider_type == "azure":
        if not azure_endpoint:
            raise ValueError("azure.endpoint is required when provider is 'azure'")
        logger.debug("Creating AzureProvider (model=%s, endpoint=%s)", model, azure_endpoint)
        return AzureProvider(endpoint=azure_endpoint, model=model, api_key=azure_api_key)

    raise ValueError(f"Unknown provider: {provider_type!r}. Choose 'anthropic' or 'azure'.")


__all__ = ["LLMProvider", "AnthropicProvider", "AzureProvider", "create_provider"]
