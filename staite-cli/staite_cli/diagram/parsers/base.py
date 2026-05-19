"""Base protocol for language-specific dependency parsers."""

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class LanguageParser(Protocol):
    """Extract intra-project dependency edges from a source file.

    Implementors inspect *content* and return a list of import targets as
    they appear in the source (e.g. ``"./utils"``, ``"services.auth"``).
    The diagram generator resolves these against the scanned file list.
    """

    extensions: list[str]
    """File extensions handled by this parser, e.g. ``[".py"]``."""

    def extract_dependencies(self, filepath: Path, content: str) -> list[str]:
        """Return raw import/dependency strings found in *content*.

        Args:
            filepath: The file being parsed (relative to project root).
            content: Full text content of the file.

        Returns:
            List of raw dependency strings. May be empty.
        """
        ...
