"""Azure AI Foundry provider.

Requires the optional [azure] extra:
    pip install staite[azure]

Authentication is resolved in this order:
    1. ``api_key`` passed explicitly (or AZURE_API_KEY env var)
    2. ``DefaultAzureCredential`` — covers managed identity, Azure CLI,
       VS Code credentials, service principal env vars, etc.
       This is the recommended path for corporate environments.
"""

import logging
import os

from staite.providers.base import TokenUsage

logger = logging.getLogger(__name__)

_AZURE_MISSING = (
    "Azure dependencies are not installed. "
    "Run: pip install staite[azure]"
)


class AzureProvider:
    """LLMProvider backed by Azure AI Foundry (azure-ai-inference).

    Args:
        endpoint: Azure AI Foundry endpoint URL.
            e.g. ``https://<resource>.services.ai.azure.com/models``
        model: Model name as deployed in Azure AI Foundry.
        api_key: Optional API key. Falls back to AZURE_API_KEY env var,
            then DefaultAzureCredential.
    """

    def __init__(self, endpoint: str, model: str, api_key: str | None = None) -> None:
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key or os.environ.get("AZURE_API_KEY")
        self._client: object | None = None
        self.usage = TokenUsage()

    def _build_client(self) -> object:
        try:
            from azure.ai.inference.aio import ChatCompletionsClient  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(_AZURE_MISSING) from exc

        if self._api_key:
            from azure.core.credentials import AzureKeyCredential  # type: ignore[import-untyped]
            credential = AzureKeyCredential(self._api_key)
            logger.debug("Azure provider: using API key authentication")
        else:
            try:
                from azure.identity import DefaultAzureCredential  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(_AZURE_MISSING) from exc
            credential = DefaultAzureCredential()
            logger.debug("Azure provider: using DefaultAzureCredential")

        return ChatCompletionsClient(endpoint=self._endpoint, credential=credential)

    async def complete(self, prompt: str, max_tokens: int) -> str:
        if self._client is None:
            raise RuntimeError("AzureProvider must be used as an async context manager")

        response = await self._client.complete(  # type: ignore[union-attr]
            messages=[{"role": "user", "content": prompt}],
            model=self._model,
            max_tokens=max_tokens,
        )
        if response.usage:
            self.usage += TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        return response.choices[0].message.content.strip()

    async def __aenter__(self) -> "AzureProvider":
        if self._client is None:
            self._client = self._build_client()
        await self._client.__aenter__()  # type: ignore[union-attr]
        logger.debug("Azure AI Foundry client opened (endpoint=%s)", self._endpoint)
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._client is not None:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)  # type: ignore[union-attr]
            logger.debug("Azure AI Foundry client closed")
