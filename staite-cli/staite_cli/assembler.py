"""Assembles all computed data into a Claude-optimised JSON state file."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def assemble(
    project_name: str,
    instructions: str,
    user_conventions: str,
    use_cases: str,
    ai_conventions: str,
    tree_lines: list[str],
    diagram: str,
    descriptions: dict[str, str],
) -> str:
    """Build the full JSON state string.

    Args:
        project_name: Human name for the project.
        instructions: Raw user instructions from config.
        user_conventions: User-provided conventions from config.
        use_cases: AI-generated use-case scenarios.
        ai_conventions: AI-inferred coding conventions.
        tree_lines: Lines from scanner.FileTree.tree_lines.
        diagram: Mermaid diagram string from diagram.generator.
        descriptions: Mapping of posix rel_path → description.

    Returns:
        Complete JSON document as a string.
    """
    now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    doc = {
        "metadata": {
            "name": project_name,
            "generated_at": now,
            "file_count": len(descriptions),
        },
        "instructions": instructions.strip(),
        "use_cases": use_cases.strip(),
        "conventions": {
            "user": user_conventions.strip(),
            "ai": ai_conventions.strip(),
        },
        "file_tree": "\n".join(tree_lines),
        "architecture": diagram,
        "files": {path: desc for path, desc in sorted(descriptions.items())},
    }

    result = json.dumps(doc, ensure_ascii=False, indent=2)
    logger.info("State assembled: %d files, %d bytes", len(descriptions), len(result))
    return result


def write(content: str, path: Path) -> None:
    """Write *content* to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("State written to %s", path)
