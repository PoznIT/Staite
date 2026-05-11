"""Filesystem scanner — builds a list of files and a tree structure.

No AI involved. Pure filesystem traversal filtered by pathspec patterns.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pathspec

logger = logging.getLogger(__name__)


@dataclass
class FileTree:
    """Represents the scanned project filesystem."""

    root: Path
    files: list[Path]
    """All matched files as paths relative to root."""

    tree_lines: list[str] = field(default_factory=list)
    """Human-readable indented tree, suitable for embedding in the state file."""


def _build_spec(patterns: list[str]) -> pathspec.PathSpec:
    return pathspec.PathSpec.from_lines("gitignore", patterns)


def _render_tree(root: Path, files: list[Path]) -> list[str]:
    """Render a sorted, indented directory tree from a flat file list.

    Args:
        root: The root directory all paths are relative to.
        files: Relative paths to include in the tree.

    Returns:
        Lines of the rendered tree (without trailing newlines).
    """
    # Build a nested dict representing the tree
    tree: dict = {}
    for f in sorted(files):
        parts = f.parts
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    lines: list[str] = [root.name + "/"]

    def _walk(node: dict, prefix: str) -> None:
        entries = sorted(node.keys())
        for i, name in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")
            lines.append(f"{prefix}{connector}{name}")
            if node[name]:  # has children → it's a directory
                _walk(node[name], child_prefix)

    _walk(tree, "")
    return lines


def scan(
    root: Path,
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> FileTree:
    """Walk *root* and return every file matching include/exclude patterns.

    Patterns follow gitignore (gitwildmatch) syntax via pathspec.

    Args:
        root: Directory to scan.
        include_patterns: Files/dirs to include.
        exclude_patterns: Files/dirs to exclude (applied after include).

    Returns:
        A FileTree with matched files and a rendered tree.

    Raises:
        NotADirectoryError: If root does not exist or is not a directory.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"Scan root is not a directory: {root}")

    include_spec = _build_spec(include_patterns)
    exclude_spec = _build_spec(exclude_patterns) if exclude_patterns else None

    matched: list[Path] = []

    for abs_path in sorted(root.rglob("*")):
        if not abs_path.is_file():
            continue

        rel = abs_path.relative_to(root)
        rel_str = rel.as_posix()

        if not include_spec.match_file(rel_str):
            continue
        if exclude_spec and exclude_spec.match_file(rel_str):
            logger.debug("Excluded: %s", rel_str)
            continue

        matched.append(rel)

    logger.info("Scan complete: %d file(s) matched under %s", len(matched), root)

    tree_lines = _render_tree(root, matched)
    return FileTree(root=root, files=matched, tree_lines=tree_lines)
