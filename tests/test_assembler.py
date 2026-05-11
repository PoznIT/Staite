"""Tests for staite.assembler."""

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


class TestAssemble:
    def test_returns_string(self):
        assert isinstance(_assemble(), str)

    def test_contains_project_state_root(self):
        result = _assemble()
        assert "<project_state>" in result
        assert "</project_state>" in result

    def test_contains_project_name(self):
        assert "MyApp" in _assemble()

    def test_contains_instructions(self):
        result = _assemble(instructions="Custom instructions here.")
        assert "Custom instructions here." in result

    def test_contains_use_cases(self):
        result = _assemble(use_cases="Scenario one.")
        assert "Scenario one." in result
        assert "<use_cases>" in result

    def test_contains_conventions_user(self):
        result = _assemble(user_conventions="Never use print().")
        assert "Never use print()." in result
        assert "<conventions>" in result

    def test_contains_conventions_ai(self):
        result = _assemble(ai_conventions="All errors are AppError.")
        assert "All errors are AppError." in result

    def test_conventions_labels(self):
        result = _assemble(user_conventions="Rule A.", ai_conventions="Rule B.")
        assert "[User-defined]" in result
        assert "[AI-inferred]" in result

    def test_conventions_only_ai_no_user_label(self):
        result = _assemble(user_conventions="", ai_conventions="Rule B.")
        assert "[User-defined]" not in result
        assert "[AI-inferred]" in result

    def test_contains_file_tree(self):
        result = _assemble()
        assert "main.py" in result
        assert "<file_tree>" in result

    def test_contains_architecture(self):
        result = _assemble()
        assert "mermaid" in result
        assert "<architecture>" in result

    def test_contains_all_file_paths(self):
        result = _assemble()
        assert 'path="src/main.py"' in result
        assert 'path="src/utils.py"' in result

    def test_contains_file_descriptions(self):
        result = _assemble()
        assert "Entry point of the application." in result
        assert "Utility helpers shared across modules." in result

    def test_contains_file_count(self):
        assert "<file_count>2</file_count>" in _assemble()

    def test_escapes_special_xml_chars(self):
        result = _assemble(
            instructions="x & y < z",
            descriptions={"a.py": "Uses <b> & 'quotes'."},
        )
        assert "&amp;" in result or "&lt;" in result

    def test_empty_descriptions(self):
        result = _assemble(descriptions={})
        assert "<file_count>0</file_count>" in result


class TestWrite:
    def test_creates_output_file(self, tmp_path: Path):
        out = tmp_path / "out" / "STATE.xml"
        write("<project_state/>", out)
        assert out.exists()
        assert out.read_text() == "<project_state/>"

    def test_creates_parent_dirs(self, tmp_path: Path):
        out = tmp_path / "deep" / "nested" / "STATE.xml"
        write("content", out)
        assert out.exists()

    def test_overwrites_existing(self, tmp_path: Path):
        out = tmp_path / "STATE.xml"
        out.write_text("old")
        write("new", out)
        assert out.read_text() == "new"
