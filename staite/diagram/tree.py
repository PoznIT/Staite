"""Directory tree → Mermaid node/edge builder.

Produces a ``graph TD`` diagram where directories are rounded boxes and
files are plain boxes, reflecting the project hierarchy.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _node_id(path: str) -> str:
    """Stable Mermaid node ID from a path string (no special chars)."""
    return path.replace("/", "_").replace(".", "_").replace("-", "_")


def build_tree_diagram(files: list[Path]) -> str:
    """Generate a Mermaid ``graph TD`` of the directory structure.

    Args:
        files: Relative file paths to include.

    Returns:
        Mermaid diagram string.
    """
    if not files:
        logger.warning("No files provided — generating empty diagram")
        return "graph TD\n"

    # Collect unique directories and files
    dirs: set[str] = set()
    edges: list[tuple[str, str]] = []  # (parent_id, child_id)

    for f in sorted(files):
        parts = f.parts
        # Register every ancestor directory
        for depth in range(len(parts) - 1):
            dirs.add("/".join(parts[: depth + 1]))
        # Edge: parent dir → file  (or root → file for top-level files)
        if len(parts) > 1:
            parent = "/".join(parts[:-1])
            edges.append((_node_id(parent), _node_id(f.as_posix())))
        # Edge: parent dir → child dir (intermediate levels)
        for depth in range(1, len(parts) - 1):
            parent = "/".join(parts[:depth])
            child = "/".join(parts[: depth + 1])
            edges.append((_node_id(parent), _node_id(child)))

    # Deduplicate edges while preserving order
    seen: set[tuple[str, str]] = set()
    unique_edges: list[tuple[str, str]] = []
    for edge in edges:
        if edge not in seen:
            seen.add(edge)
            unique_edges.append(edge)

    lines: list[str] = ["graph TD"]

    # Node definitions — directories get rounded boxes, files get plain boxes
    for d in sorted(dirs):
        nid = _node_id(d)
        label = d.split("/")[-1]
        lines.append(f'    {nid}("{label}/")' )

    for f in sorted(files):
        nid = _node_id(f.as_posix())
        label = f.name
        lines.append(f"    {nid}[{label}]")

    # Edges
    for parent_id, child_id in unique_edges:
        lines.append(f"    {parent_id} --> {child_id}")

    return "\n".join(lines)
