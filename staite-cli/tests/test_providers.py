"""Tests for staite.providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from staite_cli.providers import AnthropicProvider, AzureProvider, create_provider
from staite_cli.providers.base import LLMProvider, TokenUsage


class TestTokenUsage:
    def test_str_format(self):
        u = TokenUsage(input_tokens=1000, output_tokens=200)
        s = str(u)
        assert "1,200" in s
        assert "1,000" in s
        assert "200" in s

    def test_iadd(self):
        u = TokenUsage(100, 50)
        u += TokenUsage(200, 75)
        assert u.input_tokens == 300
        assert u.output_tokens == 125

    def test_total_tokens(self):
        u = TokenUsage(300, 100)
        assert u.total_tokens == 400

    def test_zero_by_default(self):
        u = TokenUsage()
        assert u.total_tokens == 0


class TestLLMProviderProtocol:
    def test_anthropic_provider_satisfies_protocol(self):
        provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
        assert isinstance(provider, LLMProvider)

    def test_azure_provider_satisfies_protocol(self):
        provider = AzureProvider(endpoint="https://example.com", model="claude-3-5-haiku")
        assert isinstance(provider, LLMProvider)


class TestAnthropicProvider:
    def _mock_client(self, text: str = "Response.", input_tokens: int = 100, output_tokens: int = 20) -> MagicMock:
        client = MagicMock()
        message = MagicMock()
        message.content = [MagicMock(text=text)]
        message.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
        client.messages = MagicMock()
        client.messages.create = AsyncMock(return_value=message)
        client.close = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_complete_returns_text(self):
        client = self._mock_client("Hello world.")
        provider = AnthropicProvider(model="claude-haiku-4-5-20251001", client=client)
        async with provider:
            result = await provider.complete("Say hello.", max_tokens=50)
        assert result == "Hello world."

    @pytest.mark.asyncio
    async def test_complete_passes_prompt_and_max_tokens(self):
        client = self._mock_client()
        provider = AnthropicProvider(model="claude-haiku-4-5-20251001", client=client)
        async with provider:
            await provider.complete("My prompt.", max_tokens=128)
        call_kwargs = client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 128
        assert call_kwargs["messages"][0]["content"] == "My prompt."

    @pytest.mark.asyncio
    async def test_tracks_token_usage(self):
        client = self._mock_client(input_tokens=150, output_tokens=30)
        provider = AnthropicProvider(model="claude-haiku-4-5-20251001", client=client)
        async with provider:
            await provider.complete("prompt", max_tokens=50)
        assert provider.usage.input_tokens == 150
        assert provider.usage.output_tokens == 30
        assert provider.usage.total_tokens == 180

    @pytest.mark.asyncio
    async def test_accumulates_usage_across_calls(self):
        client = self._mock_client(input_tokens=100, output_tokens=20)
        provider = AnthropicProvider(model="claude-haiku-4-5-20251001", client=client)
        async with provider:
            await provider.complete("first", max_tokens=50)
            await provider.complete("second", max_tokens=50)
        assert provider.usage.input_tokens == 200
        assert provider.usage.output_tokens == 40

    @pytest.mark.asyncio
    async def test_client_closed_on_exit(self):
        client = self._mock_client()
        provider = AnthropicProvider(model="claude-haiku-4-5-20251001", client=client)
        async with provider:
            pass
        client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_strips_whitespace(self):
        client = self._mock_client("  padded  ")
        provider = AnthropicProvider(model="claude-haiku-4-5-20251001", client=client)
        async with provider:
            result = await provider.complete(".", max_tokens=10)
        assert result == "padded"


class TestAzureProvider:
    @pytest.mark.asyncio
    async def test_complete_without_entering_context_raises(self):
        provider = AzureProvider(endpoint="https://example.com", model="m")
        with pytest.raises(RuntimeError, match="context manager"):
            await provider.complete("prompt", max_tokens=10)

    def _mock_azure_client(self, content: str = "Result.", prompt_tokens: int = 80, completion_tokens: int = 15) -> MagicMock:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=content))]
        mock_response.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        mock_client.complete = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        return mock_client

    @pytest.mark.asyncio
    async def test_complete_returns_text(self):
        """complete() calls the underlying client and returns stripped text."""
        provider = AzureProvider(endpoint="https://example.com/models", model="claude-3-5-haiku", api_key="k")
        provider._client = self._mock_azure_client()

        async with provider:
            result = await provider.complete("Hello.", max_tokens=100)

        assert result == "Result."

    @pytest.mark.asyncio
    async def test_tracks_token_usage(self):
        provider = AzureProvider(endpoint="https://example.com", model="m")
        provider._client = self._mock_azure_client(prompt_tokens=80, completion_tokens=15)

        async with provider:
            await provider.complete("p", max_tokens=50)

        assert provider.usage.input_tokens == 80
        assert provider.usage.output_tokens == 15
        assert provider.usage.total_tokens == 95

    @pytest.mark.asyncio
    async def test_complete_strips_whitespace(self):
        provider = AzureProvider(endpoint="https://example.com", model="m")
        provider._client = self._mock_azure_client(content="  padded  ")

        async with provider:
            result = await provider.complete("p", max_tokens=10)
        assert result == "padded"

    def test_missing_azure_deps_raises_import_error(self):
        """_build_client raises ImportError with helpful message if azure-ai-inference not installed."""
        import sys
        provider = AzureProvider(endpoint="https://example.com", model="m", api_key="key")
        # Simulate missing azure package by temporarily hiding it
        azure_modules = {k: v for k, v in sys.modules.items() if k.startswith("azure")}
        for mod in list(azure_modules):
            sys.modules.pop(mod, None)
        sys.modules["azure"] = None  # type: ignore[assignment]
        sys.modules["azure.ai"] = None  # type: ignore[assignment]
        sys.modules["azure.ai.inference"] = None  # type: ignore[assignment]
        sys.modules["azure.ai.inference.aio"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((ImportError, TypeError)):
                provider._build_client()
        finally:
            # Restore
            for mod in ["azure", "azure.ai", "azure.ai.inference", "azure.ai.inference.aio"]:
                sys.modules.pop(mod, None)
            sys.modules.update(azure_modules)


class TestCreateProvider:
    def test_creates_anthropic_provider(self):
        p = create_provider("anthropic", "claude-haiku-4-5-20251001")
        assert isinstance(p, AnthropicProvider)

    def test_creates_azure_provider(self):
        p = create_provider("azure", "claude-3-5-haiku", azure_endpoint="https://example.com")
        assert isinstance(p, AzureProvider)

    def test_azure_missing_endpoint_raises(self):
        with pytest.raises(ValueError, match="endpoint"):
            create_provider("azure", "claude-3-5-haiku")

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("openai", "gpt-4")
