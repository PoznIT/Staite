"""Orchestrates tree diagram + language-specific dependency edges.

The final Mermaid output is a directory tree enriched with dependency
arrows between files where the active language parsers can resolve them.
"""

import logging
from pathlib import Path

from staite.diagram.parsers import LanguageParser, build_registry
from staite.diagram.tree import _node_id, build_tree_diagram

logger = logging.getLogger(__name__)


def _resolve_dep(
    dep: str,
    source_file: Path,
    file_set: set[str],
) -> str | None:
    """Try to resolve a raw dependency string to a relative path in file_set.

    Handles:
    - Relative JS paths: ``./foo`` → ``src/foo.ts`` (tries common extensions)
    - Python dotted paths: ``services.auth`` → ``services/auth.py``
    - Relative Python imports: ``.models`` (resolved from source_file's package)
    """
    # --- JavaScript-style relative path (starts with . or /) ---
    if dep.startswith(".") or dep.startswith("/"):
        base = source_file.parent / dep if not dep.startswith("/") else Path(dep.lstrip("/"))
        candidates = [base.as_posix()]
        # Try appending common extensions if the dep has none
        if not base.suffix:
            for ext in (".ts", ".tsx", ".js", ".jsx", ".py"):
                candidates.append(base.with_suffix(ext).as_posix())
            # Also try index files
            for ext in (".ts", ".tsx", ".js", ".jsx"):
                candidates.append((base / f"index{ext}").as_posix())
        for candidate in candidates:
            # Normalise away any ".." components
            try:
                resolved = Path(candidate).resolve().relative_to(Path(".").resolve())
            except ValueError:
                resolved = Path(candidate)
            if resolved.as_posix() in file_set:
                return resolved.as_posix()
        return None

    # --- Python relative import (leading dots) ---
    if dep.startswith("."):
        stripped = dep.lstrip(".")
        levels = len(dep) - len(stripped)
        pkg_parts = source_file.parent.parts
        base_parts = pkg_parts[: max(0, len(pkg_parts) - levels + 1)]
        candidate = Path(*base_parts, *stripped.split(".")).with_suffix(".py")
        if candidate.as_posix() in file_set:
            return candidate.as_posix()
        return None

    # --- Python absolute dotted import ---
    candidate = Path(*dep.split(".")).with_suffix(".py")
    if candidate.as_posix() in file_set:
        return candidate.as_posix()
    return None


def generate(
    files: list[Path],
    root: Path,
    extra_parsers: list[LanguageParser] | None = None,
) -> str:
    """Generate a Mermaid diagram combining tree structure and dep edges.

    Args:
        files: Relative file paths (from scanner).
        root: Project root (used to read file contents for parsing).
        extra_parsers: Additional language parsers to register.

    Returns:
        Mermaid diagram string.
    """
    base_diagram = build_tree_diagram(files)

    registry = build_registry(extra_parsers)
    if not registry:
        return base_diagram

    file_set = {f.as_posix() for f in files}
    dep_edges: list[tuple[str, str]] = []

    for f in files:
        ext = f.suffix.lower().lstrip(".")
        parser = registry.get(ext)
        if parser is None:
            continue

        abs_path = root / f
        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Could not read %s for dep parsing: %s", f, exc)
            continue

        raw_deps = parser.extract_dependencies(f, content)
        for dep in raw_deps:
            target = _resolve_dep(dep, f, file_set)
            if target and target != f.as_posix():
                dep_edges.append((_node_id(f.as_posix()), _node_id(target)))

    if not dep_edges:
        logger.debug("No resolvable dependency edges found")
        return base_diagram

    # Deduplicate
    seen: set[tuple[str, str]] = set()
    lines = [base_diagram]
    lines.append("\n    %% dependency edges")
    for src, tgt in dep_edges:
        if (src, tgt) not in seen:
            seen.add((src, tgt))
            lines.append(f"    {src} -.-> {tgt}")

    logger.info("Diagram: %d dependency edge(s) added", len(seen))
    return "\n".join(lines)
