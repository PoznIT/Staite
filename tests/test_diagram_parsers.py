"""Tests for language parsers."""

from pathlib import Path

from staite.diagram.parsers.javascript import JavaScriptParser
from staite.diagram.parsers.python import PythonParser


class TestPythonParser:
    def setup_method(self):
        self.parser = PythonParser()

    def test_simple_import(self):
        deps = self.parser.extract_dependencies(Path("main.py"), "import os\n")
        assert "os" in deps

    def test_from_import(self):
        deps = self.parser.extract_dependencies(Path("main.py"), "from pathlib import Path\n")
        assert "pathlib" in deps

    def test_relative_import(self):
        deps = self.parser.extract_dependencies(Path("src/main.py"), "from .utils import helper\n")
        assert ".utils" in deps

    def test_deeply_relative_import(self):
        deps = self.parser.extract_dependencies(Path("a/b/c.py"), "from ...top import x\n")
        assert "...top" in deps

    def test_multiple_imports(self):
        code = "import os\nimport sys\nfrom pathlib import Path\n"
        deps = self.parser.extract_dependencies(Path("f.py"), code)
        assert "os" in deps
        assert "sys" in deps
        assert "pathlib" in deps

    def test_syntax_error_returns_empty(self):
        deps = self.parser.extract_dependencies(Path("bad.py"), "def (broken syntax{{{")
        assert deps == []

    def test_empty_file(self):
        deps = self.parser.extract_dependencies(Path("empty.py"), "")
        assert deps == []

    def test_extensions(self):
        assert ".py" in PythonParser.extensions


class TestJavaScriptParser:
    def setup_method(self):
        self.parser = JavaScriptParser()

    def test_es_module_import(self):
        deps = self.parser.extract_dependencies(
            Path("app.js"), "import foo from './foo'\n"
        )
        assert "./foo" in deps

    def test_es_module_double_quote(self):
        deps = self.parser.extract_dependencies(
            Path("app.js"), 'import bar from "./bar"\n'
        )
        assert "./bar" in deps

    def test_require(self):
        deps = self.parser.extract_dependencies(
            Path("app.js"), "const x = require('./utils')\n"
        )
        assert "./utils" in deps

    def test_side_effect_import(self):
        deps = self.parser.extract_dependencies(
            Path("app.js"), "import './styles.css'\n"
        )
        assert "./styles.css" in deps

    def test_ignores_third_party(self):
        deps = self.parser.extract_dependencies(
            Path("app.js"), "import React from 'react'\n"
        )
        assert deps == []

    def test_export_from(self):
        deps = self.parser.extract_dependencies(
            Path("index.js"), "export { foo } from './foo'\n"
        )
        assert "./foo" in deps

    def test_empty_file(self):
        deps = self.parser.extract_dependencies(Path("empty.ts"), "")
        assert deps == []

    def test_extensions_cover_ts_tsx(self):
        exts = JavaScriptParser.extensions
        assert ".ts" in exts
        assert ".tsx" in exts
        assert ".js" in exts
        assert ".jsx" in exts
