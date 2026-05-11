"""Assembles all computed data into a Claude-optimised XML state file.

The output uses Anthropic-recommended XML tags so Claude can parse sections
reliably when the file is pasted into project instructions.

Output structure:
    <project_state>
      <metadata>          — name, generation timestamp
      <instructions>      — user-provided context from config
      <use_cases>         — AI-generated use-case scenarios
      <conventions>       — user-provided + AI-inferred coding conventions
      <file_tree>         — rendered directory tree (plain text)
      <architecture>      — Mermaid diagram
      <files>             — one <file> tag per scanned file with description
    </project_state>
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

logger = logging.getLogger(__name__)

_INDENT = "  "


def _tag(name: str, content: str, attrs: dict[str, str] | None = None, indent: int = 0) -> str:
    """Wrap *content* in an XML tag with optional attributes."""
    pad = _INDENT * indent
    attr_str = ""
    if attrs:
        attr_str = " " + " ".join(f'{k}="{escape(v)}"' for k, v in attrs.items())
    return f"{pad}<{name}{attr_str}>\n{content}\n{pad}</{name}>"


def _conventions_content(user_conventions: str, ai_conventions: str) -> str:
    """Merge user-provided and AI-inferred conventions into a single block."""
    parts: list[str] = []
    if user_conventions.strip():
        parts.append(f"{_INDENT * 2}[User-defined]\n{_INDENT * 2}{user_conventions.strip()}")
    if ai_conventions.strip():
        parts.append(f"{_INDENT * 2}[AI-inferred]\n{_INDENT * 2}{ai_conventions.strip()}")
    return "\n\n".join(parts)


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
    """Build the full XML state string.

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
        Complete XML document as a string.
    """
    now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

    # <metadata>
    meta_inner = (
        f"{_INDENT * 2}<name>{escape(project_name)}</name>\n"
        f"{_INDENT * 2}<generated_at>{now}</generated_at>\n"
        f"{_INDENT * 2}<file_count>{len(descriptions)}</file_count>"
    )
    metadata = _tag("metadata", meta_inner, indent=1)

    # <instructions>
    instr_block = _tag("instructions", escape(instructions.strip()), indent=1)

    # <use_cases>
    use_cases_block = _tag("use_cases", f"{_INDENT * 2}{escape(use_cases.strip())}", indent=1)

    # <conventions>
    conv_content = _conventions_content(user_conventions, ai_conventions)
    conventions_block = _tag("conventions", conv_content, indent=1)

    # <file_tree>
    tree_text = "\n".join(f"{_INDENT * 2}{line}" for line in tree_lines)
    file_tree = _tag("file_tree", tree_text, indent=1)

    # <architecture>
    diagram_block = _tag(
        "architecture",
        f"{_INDENT * 2}```mermaid\n"
        + "\n".join(f"{_INDENT * 2}{line}" for line in diagram.splitlines())
        + f"\n{_INDENT * 2}```",
        indent=1,
    )

    # <files>
    file_tags: list[str] = []
    for rel_path in sorted(descriptions):
        desc = escape(descriptions[rel_path])
        file_tags.append(
            _tag("file", f"{_INDENT * 2}{desc}", attrs={"path": rel_path}, indent=2)
        )
    files_inner = "\n".join(file_tags)
    files_block = _tag("files", files_inner, indent=1)

    # Root tag — ordered for Claude readability: summary first, detail last
    inner = "\n\n".join(
        [metadata, instr_block, use_cases_block, conventions_block, file_tree, diagram_block, files_block]
    )
    root = _tag("project_state", inner)

    logger.info(
        "State assembled: %d files, %d bytes",
        len(descriptions),
        len(root),
    )
    return root


def write(content: str, path: Path) -> None:
    """Write *content* to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("State written to %s", path)
