"""Python dependency parser.

Extracts ``import x`` and ``from x import y`` statements using the stdlib
``ast`` module — no regex fragility.
"""

import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PythonParser:
    extensions: list[str] = [".py"]

    def extract_dependencies(self, filepath: Path, content: str) -> list[str]:
        """Return all module names imported in *content*.

        Uses ``ast.parse`` for accuracy. Falls back to an empty list on
        syntax errors (e.g. Python 2 files, partial snippets).
        """
        try:
            tree = ast.parse(content, filename=str(filepath))
        except SyntaxError:
            logger.debug("Syntax error parsing %s — skipping dependency extraction", filepath)
            return []

        deps: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    deps.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    prefix = "." * (node.level or 0)
                    deps.append(f"{prefix}{node.module}")
        return deps
