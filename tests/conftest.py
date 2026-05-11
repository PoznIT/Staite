"""Shared pytest fixtures."""

import json
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """Create a minimal fake project tree for testing."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("import os\nprint('hello')\n")
    (tmp_path / "src" / "utils.py").write_text("def helper(): pass\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("from src import main\n")
    (tmp_path / "README.md").write_text("# Project\n")
    return tmp_path


@pytest.fixture()
def tmp_cache_path(tmp_path: Path) -> Path:
    return tmp_path / ".staite" / "cache.json"
