"""LLMProvider protocol — the only interface describer and synthesizer depend on."""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class TokenUsage:
    """Accumulated token consumption across all calls made through a provider."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __iadd__(self, other: "TokenUsage") -> "TokenUsage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        return self

    def __str__(self) -> str:
        return (
            f"{self.total_tokens:,} tokens total "
            f"({self.input_tokens:,} in / {self.output_tokens:,} out)"
        )


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal async LLM interface.

    Providers are async context managers so resources (HTTP sessions, credentials)
    are cleaned up correctly regardless of which backend is used.

    Usage::

        async with provider:
            text = await provider.complete("describe this file", max_tokens=512)
            print(provider.usage)
    """

    usage: TokenUsage
    """Accumulated token usage since the provider was instantiated."""

    async def complete(self, prompt: str, max_tokens: int) -> str:
        """Send a single user prompt and return the response text.

        Args:
            prompt: Full user-turn prompt string.
            max_tokens: Maximum tokens to generate.

        Returns:
            Response text, stripped of leading/trailing whitespace.
        """
        ...

    async def __aenter__(self) -> "LLMProvider":
        ...

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        ...
