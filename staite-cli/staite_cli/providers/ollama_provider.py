"""Ollama local LLM provider."""

import logging

import httpx

from staite_cli.providers.base import TokenUsage

logger = logging.getLogger(__name__)


class OllamaProvider:
    """LLMProvider backed by a local Ollama instance.

    Uses httpx async client under the hood; the session is opened/closed
    via the async context manager.
    """

    def __init__(self, model: str, url: str, timeout: int= 300) -> None:
        self._model = model
        self._url = url.rstrip("/")
        self._client: httpx.AsyncClient | None = None
        self.__timeout = timeout
        self.usage = TokenUsage()

    async def complete(self, prompt: str, max_tokens: int) -> str:
        assert self._client is not None, "OllamaProvider must be used as an async context manager"
        response = await self._client.post(
            f"{self._url}/api/chat",
            json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
        )
        response.raise_for_status()
        data = response.json()

        # Ollama reports eval_count (output) and prompt_eval_count (input)
        self.usage += TokenUsage(
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )
        return data["message"]["content"].strip()

    async def __aenter__(self) -> "OllamaProvider":
        self._client = httpx.AsyncClient(timeout=self.__timeout)
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
            logger.debug("Ollama client closed")