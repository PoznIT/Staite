"""Tests for staite.synthesizer."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from staite_cli.synthesizer import (
    SynthesisCache,
    SynthesisRecord,
    should_regenerate,
    synthesize,
)


class TestShouldRegenerate:
    def test_no_cache_always_regenerates(self):
        assert should_regenerate(0, 10, 0.2, has_cached=False) is True

    def test_below_threshold_uses_cache(self):
        assert should_regenerate(1, 10, 0.2, has_cached=True) is False  # 10% < 20%

    def test_at_threshold_regenerates(self):
        assert should_regenerate(2, 10, 0.2, has_cached=True) is True  # 20% == 20%

    def test_above_threshold_regenerates(self):
        assert should_regenerate(5, 10, 0.2, has_cached=True) is True  # 50% > 20%

    def test_zero_total_files_no_regen(self):
        assert should_regenerate(0, 0, 0.2, has_cached=True) is False

    def test_all_files_changed_regenerates(self):
        assert should_regenerate(10, 10, 0.2, has_cached=True) is True


class TestSynthesisCache:
    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        cache = SynthesisCache.load(tmp_path / "synthesis.json")
        assert cache.record is None

    def test_load_existing_file(self, tmp_path: Path):
        path = tmp_path / "synthesis.json"
        record = SynthesisRecord(use_cases="UC1", conventions_ai="C1")
        path.write_text(record.model_dump_json())
        cache = SynthesisCache.load(path)
        assert cache.record is not None
        assert cache.record.use_cases_diagram == "UC1"

    def test_update_and_save(self, tmp_path: Path):
        path = tmp_path / "synthesis.json"
        cache = SynthesisCache.load(path)
        cache.update(SynthesisRecord(use_cases="UC", conventions_ai="C"))
        cache.save()
        assert path.exists()

    def test_save_skipped_when_not_dirty(self, tmp_path: Path):
        path = tmp_path / "synthesis.json"
        cache = SynthesisCache.load(path)
        cache.save()
        assert not path.exists()

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        path = tmp_path / "deep" / "dir" / "synthesis.json"
        cache = SynthesisCache.load(path)
        cache.update(SynthesisRecord(use_cases="x", conventions_ai="y"))
        cache.save()
        assert path.exists()

    def test_reload_after_save(self, tmp_path: Path):
        path = tmp_path / "s.json"
        cache = SynthesisCache.load(path)
        cache.update(SynthesisRecord(use_cases="scenarios", conventions_ai="conventions"))
        cache.save()

        reloaded = SynthesisCache.load(path)
        assert reloaded.record is not None
        assert reloaded.record.use_cases_diagram == "scenarios"
        assert reloaded.record.conventions_ai == "conventions"


class TestSynthesize:
    def _make_provider(self, use_cases_text: str = "UC", conventions_text: str = "C") -> MagicMock:
        provider = MagicMock()
        provider.complete = AsyncMock(side_effect=[use_cases_text, conventions_text])
        provider.__aenter__ = AsyncMock(return_value=provider)
        provider.__aexit__ = AsyncMock(return_value=None)
        return provider

    @pytest.mark.asyncio
    async def test_regenerates_when_no_cache(self, tmp_path: Path):
        provider = self._make_provider("Use cases here.", "Conventions here.")
        cache = SynthesisCache.load(tmp_path / "s.json")

        result = await synthesize(
            provider=provider,
            project_name="P",
            instructions="",
            user_conventions="",
            tree_lines=["root/"],
            descriptions={"a.py": "Does A."},
            cache=cache,
            miss_count=1,
            regen_threshold=0.2,
        )

        assert result.regenerated is True
        assert result.use_cases_diagram == "Use cases here."
        assert result.conventions_ai == "Conventions here."
        assert provider.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_uses_cache_below_threshold(self, tmp_path: Path):
        provider = self._make_provider()
        path = tmp_path / "s.json"
        existing = SynthesisRecord(use_cases="Cached UC", conventions_ai="Cached C")
        cache = SynthesisCache.load(path)
        cache.update(existing)

        result = await synthesize(
            provider=provider,
            project_name="P",
            instructions="",
            user_conventions="",
            tree_lines=["root/"],
            descriptions={f"file{i}.py": "desc" for i in range(10)},
            cache=cache,
            miss_count=1,  # 10% < 20% threshold
            regen_threshold=0.2,
        )

        assert result.regenerated is False
        assert result.use_cases_diagram == "Cached UC"
        provider.complete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_regenerates_above_threshold(self, tmp_path: Path):
        provider = self._make_provider("New UC", "New C")
        path = tmp_path / "s.json"
        cache = SynthesisCache.load(path)
        cache.update(SynthesisRecord(use_cases="Old UC", conventions_ai="Old C"))

        result = await synthesize(
            provider=provider,
            project_name="P",
            instructions="",
            user_conventions="",
            tree_lines=["root/"],
            descriptions={f"file{i}.py": "desc" for i in range(10)},
            cache=cache,
            miss_count=5,  # 50% > 20% threshold
            regen_threshold=0.2,
        )

        assert result.regenerated is True
        assert result.use_cases_diagram == "New UC"
