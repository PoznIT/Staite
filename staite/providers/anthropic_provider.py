"""Anthropic API provider."""

import logging

import anthropic

from staite.providers.base import TokenUsage

logger = logging.getLogger(__name__)


class AnthropicProvider:
    """LLMProvider backed by the Anthropic async SDK.

    The Anthropic client manages its own connection pool; we close it on exit.
    """

    def __init__(self, model: str, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._model = model
        self._client = client or anthropic.AsyncAnthropic()
        self.usage = TokenUsage()

    async def complete(self, prompt: str, max_tokens: int) -> str:
        message = await self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        self.usage += TokenUsage(
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
        return message.content[0].text.strip()

    async def __aenter__(self) -> "AnthropicProvider":
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self._client.close()
        logger.debug("Anthropic client closed")
