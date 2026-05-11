"""Tests for staite.scanner."""

from pathlib import Path

import pytest

from staite.scanner import FileTree, scan


class TestScan:
    def test_include_all(self, tmp_project: Path):
        result = scan(tmp_project, include_patterns=["**/*"], exclude_patterns=[])
        posix_files = {f.as_posix() for f in result.files}
        assert "src/main.py" in posix_files
        assert "src/utils.py" in posix_files
        assert "README.md" in posix_files

    def test_include_src_only(self, tmp_project: Path):
        result = scan(tmp_project, include_patterns=["src/**"], exclude_patterns=[])
        posix_files = {f.as_posix() for f in result.files}
        assert all(f.startswith("src/") for f in posix_files)
        assert "README.md" not in posix_files

    def test_exclude_tests(self, tmp_project: Path):
        result = scan(
            tmp_project,
            include_patterns=["**/*"],
            exclude_patterns=["tests/**"],
        )
        posix_files = {f.as_posix() for f in result.files}
        assert not any(f.startswith("tests/") for f in posix_files)

    def test_exclude_extension(self, tmp_project: Path):
        result = scan(
            tmp_project,
            include_patterns=["**/*"],
            exclude_patterns=["*.md"],
        )
        posix_files = {f.as_posix() for f in result.files}
        assert "README.md" not in posix_files

    def test_returns_only_files_not_dirs(self, tmp_project: Path):
        result = scan(tmp_project, include_patterns=["**/*"], exclude_patterns=[])
        for f in result.files:
            assert (tmp_project / f).is_file()

    def test_invalid_root_raises(self, tmp_path: Path):
        with pytest.raises(NotADirectoryError):
            scan(tmp_path / "nonexistent", ["**/*"], [])

    def test_tree_lines_populated(self, tmp_project: Path):
        result = scan(tmp_project, include_patterns=["**/*"], exclude_patterns=[])
        assert len(result.tree_lines) > 0
        assert result.tree_lines[0].endswith("/")  # root line ends with /

    def test_empty_project(self, tmp_path: Path):
        result = scan(tmp_path, include_patterns=["**/*"], exclude_patterns=[])
        assert result.files == []
        assert result.tree_lines[0].endswith("/")
