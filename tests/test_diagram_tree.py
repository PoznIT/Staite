"""Tests for staite.diagram.tree."""

from pathlib import Path

from staite.diagram.tree import build_tree_diagram


class TestBuildTreeDiagram:
    def test_empty_files(self):
        result = build_tree_diagram([])
        assert result == "graph TD\n"

    def test_single_file(self):
        result = build_tree_diagram([Path("main.py")])
        assert "graph TD" in result
        assert "main_py" in result

    def test_nested_file(self):
        result = build_tree_diagram([Path("src/utils.py")])
        assert "graph TD" in result
        # directory node
        assert "src" in result
        # file node
        assert "utils_py" in result
        # edge between them
        assert "-->" in result

    def test_multiple_files_same_dir(self):
        files = [Path("src/a.py"), Path("src/b.py")]
        result = build_tree_diagram(files)
        assert "a_py" in result
        assert "b_py" in result

    def test_deep_nesting(self):
        files = [Path("a/b/c/deep.py")]
        result = build_tree_diagram(files)
        assert "a" in result
        assert "b" in result
        assert "c" in result
        assert "deep_py" in result

    def test_no_duplicate_edges(self):
        files = [Path("src/a.py"), Path("src/b.py")]
        result = build_tree_diagram(files)
        lines = result.splitlines()
        edge_lines = [l for l in lines if "-->" in l]
        # src → a.py and src → b.py only, no duplicates
        assert len(edge_lines) == len(set(edge_lines))

    def test_output_starts_with_graph_td(self):
        result = build_tree_diagram([Path("foo.py")])
        assert result.startswith("graph TD")
