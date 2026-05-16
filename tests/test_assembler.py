"""Tests for staite.assembler."""

import json
from pathlib import Path

import pytest

from staite.assembler import assemble, write


SAMPLE_DESCRIPTIONS = {
    "src/main.py": "Entry point of the application.",
    "src/utils.py": "Utility helpers shared across modules.",
}

SAMPLE_TREE = ["myproject/", "├── src/", "│   ├── main.py", "│   └── utils.py"]

SAMPLE_DIAGRAM = "graph TD\n    src_main_py[main.py]\n    src_utils_py[utils.py]"

SAMPLE_USE_CASES = "1. Developers scan their project to generate context."

SAMPLE_AI_CONVENTIONS = "- All modules use stdlib logging via getLogger(__name__)."


def _assemble(**kwargs):
    """Call assemble with sensible defaults for unspecified args."""
    defaults = dict(
        project_name="MyApp",
        instructions="",
        user_conventions="",
        use_cases=SAMPLE_USE_CASES,
        ai_conventions=SAMPLE_AI_CONVENTIONS,
        tree_lines=SAMPLE_TREE,
        diagram=SAMPLE_DIAGRAM,
        descriptions=SAMPLE_DESCRIPTIONS,
    )
    defaults.update(kwargs)
    return assemble(**defaults)


def _parse(result: str) -> dict:
    return json.loads(result)


class TestAssemble:
    def test_returns_string(self):
        assert isinstance(_assemble(), str)

    def test_valid_json(self):
        _parse(_assemble())  # raises if invalid

    def test_contains_project_name(self):
        assert _parse(_assemble())["metadata"]["name"] == "MyApp"

    def test_contains_instructions(self):
        doc = _parse(_assemble(instructions="Custom instructions here."))
        assert doc["instructions"] == "Custom instructions here."

    def test_contains_use_cases(self):
        doc = _parse(_assemble(use_cases="Scenario one."))
        assert "Scenario one." in doc["use_cases"]

    def test_contains_conventions_user(self):
        doc = _parse(_assemble(user_conventions="Never use print()."))
        assert "Never use print()." in doc["conventions"]["user"]

    def test_contains_conventions_ai(self):
        doc = _parse(_assemble(ai_conventions="All errors are AppError."))
        assert "All errors are AppError." in doc["conventions"]["ai"]

    def test_conventions_labels_absent(self):
        # Labels are no longer embedded in the value; keys carry that meaning.
        doc = _parse(_assemble(user_conventions="Rule A.", ai_conventions="Rule B."))
        assert "Rule A." in doc["conventions"]["user"]
        assert "Rule B." in doc["conventions"]["ai"]

    def test_contains_file_tree(self):
        doc = _parse(_assemble())
        assert "main.py" in doc["file_tree"]

    def test_contains_architecture(self):
        doc = _parse(_assemble())
        assert "mermaid" in doc["architecture"].lower() or "graph" in doc["architecture"]

    def test_contains_all_file_paths(self):
        doc = _parse(_assemble())
        assert "src/main.py" in doc["files"]
        assert "src/utils.py" in doc["files"]

    def test_contains_file_descriptions(self):
        doc = _parse(_assemble())
        assert doc["files"]["src/main.py"] == "Entry point of the application."
        assert doc["files"]["src/utils.py"] == "Utility helpers shared across modules."

    def test_contains_file_count(self):
        assert _parse(_assemble())["metadata"]["file_count"] == 2

    def test_special_chars_roundtrip(self):
        # JSON natively handles & < > and quotes — no escaping needed.
        result = _assemble(
            instructions='x & y < z "quoted"',
            descriptions={"a.py": "Uses <b> & 'quotes'."},
        )
        doc = _parse(result)
        assert doc["instructions"] == 'x & y < z "quoted"'
        assert doc["files"]["a.py"] == "Uses <b> & 'quotes'."

    def test_empty_descriptions(self):
        assert _parse(_assemble(descriptions={}))["metadata"]["file_count"] == 0

    def test_files_sorted(self):
        doc = _parse(_assemble())
        keys = list(doc["files"].keys())
        assert keys == sorted(keys)


class TestWrite:
    def test_creates_output_file(self, tmp_path: Path):
        out = tmp_path / "out" / "STATE.json"
        write('{"ok": true}', out)
        assert out.exists()
        assert out.read_text() == '{"ok": true}'

    def test_creates_parent_dirs(self, tmp_path: Path):
        out = tmp_path / "deep" / "nested" / "STATE.json"
        write("content", out)
        assert out.exists()

    def test_overwrites_existing(self, tmp_path: Path):
        out = tmp_path / "STATE.json"
        out.write_text("old")
        write("new", out)
        assert out.read_text() == "new"
