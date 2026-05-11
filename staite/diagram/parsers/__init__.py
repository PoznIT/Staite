"""Language parser registry and built-in parsers."""

import logging
from pathlib import Path

from staite.diagram.parsers.base import LanguageParser
from staite.diagram.parsers.javascript import JavaScriptParser
from staite.diagram.parsers.python import PythonParser

logger = logging.getLogger(__name__)

# Default registry: extension (lowercase, without dot) → parser instance
_DEFAULT_PARSERS: list[LanguageParser] = [
    PythonParser(),
    JavaScriptParser(),
]


def build_registry(extra: list[LanguageParser] | None = None) -> dict[str, LanguageParser]:
    """Build an extension → parser mapping from the default parsers plus any extras.

    Later entries win on extension conflicts, so callers can override defaults.
    """
    parsers = list(_DEFAULT_PARSERS)
    if extra:
        parsers.extend(extra)

    registry: dict[str, LanguageParser] = {}
    for parser in parsers:
        for ext in parser.extensions:
            registry[ext.lower().lstrip(".")] = parser
    return registry


__all__ = ["LanguageParser", "build_registry"]
