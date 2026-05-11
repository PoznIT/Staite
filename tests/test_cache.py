"""Tests for staite.cache."""

import json
from pathlib import Path

import pytest

from staite.cache import CacheEntry, DescriptionCache, hash_file


class TestDescriptionCache:
    def test_empty_cache_on_missing_file(self, tmp_cache_path: Path):
        cache = DescriptionCache.load(tmp_cache_path)
        assert cache.size == 0

    def test_get_returns_none_on_empty(self, tmp_cache_path: Path):
        cache = DescriptionCache.load(tmp_cache_path)
        assert cache.get("src/foo.py", "abc123") is None

    def test_set_and_get_hit(self, tmp_cache_path: Path):
        cache = DescriptionCache.load(tmp_cache_path)
        cache.set("src/foo.py", "abc123", "Does something useful.")
        assert cache.get("src/foo.py", "abc123") == "Does something useful."

    def test_get_miss_on_wrong_hash(self, tmp_cache_path: Path):
        cache = DescriptionCache.load(tmp_cache_path)
        cache.set("src/foo.py", "abc123", "Old description.")
        assert cache.get("src/foo.py", "different_hash") is None

    def test_save_creates_file(self, tmp_cache_path: Path):
        cache = DescriptionCache.load(tmp_cache_path)
        cache.set("src/foo.py", "abc123", "A file.")
        cache.save()
        assert tmp_cache_path.exists()

    def test_save_and_reload(self, tmp_cache_path: Path):
        cache = DescriptionCache.load(tmp_cache_path)
        cache.set("src/bar.py", "xyz", "Bar module.")
        cache.save()

        reloaded = DescriptionCache.load(tmp_cache_path)
        assert reloaded.get("src/bar.py", "xyz") == "Bar module."

    def test_save_skipped_when_not_dirty(self, tmp_cache_path: Path):
        cache = DescriptionCache.load(tmp_cache_path)
        cache.save()  # no-op — should not create the file
        assert not tmp_cache_path.exists()

    def test_overwrite_existing_entry(self, tmp_cache_path: Path):
        cache = DescriptionCache.load(tmp_cache_path)
        cache.set("a.py", "h1", "First.")
        cache.set("a.py", "h2", "Second.")
        assert cache.get("a.py", "h2") == "Second."
        assert cache.size == 1

    def test_load_existing_json(self, tmp_cache_path: Path):
        tmp_cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"src/foo.py": {"sha256": "deadbeef", "description": "Loaded from disk."}}
        tmp_cache_path.write_text(json.dumps(payload))
        cache = DescriptionCache.load(tmp_cache_path)
        assert cache.get("src/foo.py", "deadbeef") == "Loaded from disk."


class TestHashFile:
    def test_same_content_same_hash(self, tmp_path: Path):
        f = tmp_path / "a.txt"
        f.write_bytes(b"hello world")
        h1 = hash_file(f)
        h2 = hash_file(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path: Path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"hello")
        f2.write_bytes(b"world")
        assert hash_file(f1) != hash_file(f2)

    def test_returns_hex_string(self, tmp_path: Path):
        f = tmp_path / "f.txt"
        f.write_bytes(b"data")
        h = hash_file(f)
        assert len(h) == 64  # SHA-256 hex = 64 chars
        assert all(c in "0123456789abcdef" for c in h)
