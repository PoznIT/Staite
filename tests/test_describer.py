"""Tests for staite.describer."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from staite.cache import DescriptionCache
from staite.describer import _sentence_count, describe_files


class TestSentenceCount:
    def test_small_file(self):
        assert _sentence_count(5) == 2

    def test_boundary_below_30(self):
        assert _sentence_count(29) == 2

    def test_boundary_at_30(self):
        assert _sentence_count(30) == 4

    def test_medium_file(self):
        assert _sentence_count(100) == 4

    def test_boundary_below_150(self):
        assert _sentence_count(149) == 4

    def test_boundary_at_150(self):
        assert _sentence_count(150) == 6

    def test_large_file(self):
        assert _sentence_count(500) == 6


class TestDescribeFiles:
    @pytest.fixture()
    def mock_provider(self):
        provider = MagicMock()
        provider.complete = AsyncMock(return_value="This file does something important.")
        provider.__aenter__ = AsyncMock(return_value=provider)
        provider.__aexit__ = AsyncMock(return_value=None)
        return provider

    @pytest.fixture()
    def cache(self, tmp_path: Path) -> DescriptionCache:
        return DescriptionCache.load(tmp_path / "cache.json")

    @pytest.mark.asyncio
    async def test_describes_new_file(
        self, tmp_path: Path, mock_provider: MagicMock, cache: DescriptionCache
    ):
        f = tmp_path / "src" / "foo.py"
        f.parent.mkdir()
        f.write_text("def foo(): pass\n")

        result = await describe_files(
            provider=mock_provider,
            root=tmp_path,
            rel_paths=[Path("src/foo.py")],
            cache=cache,
        )

        assert "src/foo.py" in result.descriptions
        assert result.descriptions["src/foo.py"] == "This file does something important."
        assert result.cache_miss_count == 1
        mock_provider.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_uses_cache_hit(
        self, tmp_path: Path, mock_provider: MagicMock, cache: DescriptionCache
    ):
        f = tmp_path / "src" / "bar.py"
        f.parent.mkdir()
        f.write_text("def bar(): pass\n")

        from staite.cache import hash_file
        h = hash_file(f)
        cache.set("src/bar.py", h, "Cached description.")

        result = await describe_files(
            provider=mock_provider,
            root=tmp_path,
            rel_paths=[Path("src/bar.py")],
            cache=cache,
        )

        assert result.descriptions["src/bar.py"] == "Cached description."
        assert result.cache_miss_count == 0
        mock_provider.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_cache_miss_on_changed_file(
        self, tmp_path: Path, mock_provider: MagicMock, cache: DescriptionCache
    ):
        f = tmp_path / "changed.py"
        f.write_text("new content\n")
        cache.set("changed.py", "old_hash", "Stale description.")

        result = await describe_files(
            provider=mock_provider,
            root=tmp_path,
            rel_paths=[Path("changed.py")],
            cache=cache,
        )

        mock_provider.complete.assert_awaited_once()
        assert result.descriptions["changed.py"] == "This file does something important."
        assert result.cache_miss_count == 1

    @pytest.mark.asyncio
    async def test_empty_file_list(
        self, tmp_path: Path, mock_provider: MagicMock, cache: DescriptionCache
    ):
        result = await describe_files(
            provider=mock_provider,
            root=tmp_path,
            rel_paths=[],
            cache=cache,
        )
        assert result.descriptions == {}
        assert result.cache_miss_count == 0
        mock_provider.complete.assert_not_awaited()
