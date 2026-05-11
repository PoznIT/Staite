"""High-level synthesis: use-case scenarios and conventions.

Makes two AI calls (use-cases + conventions) using the full set of file
descriptions as input. Both are gated behind a change threshold — if fewer
than `regen_threshold` fraction of files changed since the last run, the
cached synthesis is reused as-is.

Persists to `.staite/synthesis.json`.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from staite.providers.base import LLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class SynthesisRecord(BaseModel):
    use_cases: str
    conventions_ai: str


class SynthesisCache:
    """Persist and retrieve the last generated synthesis."""

    def __init__(self, path: Path, record: SynthesisRecord | None) -> None:
        self._path = path
        self._record = record
        self._dirty = False

    @classmethod
    def load(cls, path: Path) -> "SynthesisCache":
        if path.exists():
            logger.debug("Loading synthesis cache from %s", path)
            record = SynthesisRecord.model_validate_json(path.read_text(encoding="utf-8"))
        else:
            logger.debug("No synthesis cache at %s — will regenerate", path)
            record = None
        return cls(path=path, record=record)

    @property
    def record(self) -> SynthesisRecord | None:
        return self._record

    def update(self, record: SynthesisRecord) -> None:
        self._record = record
        self._dirty = True

    def save(self) -> None:
        if not self._dirty or self._record is None:
            logger.debug("Synthesis cache unchanged — skipping write")
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(self._record.model_dump_json(indent=2), encoding="utf-8")
        self._dirty = False
        logger.info("Synthesis cache saved to %s", self._path)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class SynthesisResult:
    use_cases: str
    """AI-generated use-case scenario description."""

    conventions_ai: str
    """AI-inferred coding conventions (may be empty string)."""

    regenerated: bool
    """True if a new AI call was made; False if the cache was reused."""


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def _use_cases_prompt(
    project_name: str,
    tree: str,
    descriptions: str,
    instructions: str,
) -> str:
    return (
        f"You are analysing a software project called '{project_name}'.\n\n"
        f"<file_tree>\n{tree}\n</file_tree>\n\n"
        f"<file_descriptions>\n{descriptions}\n</file_descriptions>\n\n"
        f"<instructions>\n{instructions}\n</instructions>\n\n"
        "Write 3–5 concrete use-case scenarios this project supports. "
        "For each scenario: describe the user goal and name the key files/components involved. "
        "Format as a short numbered list. "
        "Be specific — mention actual file names where relevant. "
        "This will be read by an AI assistant helping a developer navigate the codebase."
    )


def _conventions_prompt(
    project_name: str,
    tree: str,
    descriptions: str,
    user_conventions: str,
) -> str:
    user_block = (
        f"<user_conventions>\n{user_conventions}\n</user_conventions>\n\n"
        if user_conventions.strip()
        else ""
    )
    return (
        f"You are analysing a software project called '{project_name}'.\n\n"
        f"<file_tree>\n{tree}\n</file_tree>\n\n"
        f"<file_descriptions>\n{descriptions}\n</file_descriptions>\n\n"
        f"{user_block}"
        "Infer the key coding conventions and architectural patterns used in this project. "
        "Be specific — mention actual file names, class names, or patterns you observe. "
        + (
            "Do NOT repeat anything already stated in <user_conventions>. "
            if user_conventions.strip()
            else ""
        )
        + "Output as a concise bulleted list (5–10 bullets max). "
        "This will be read by an AI assistant helping a developer navigate the codebase."
    )


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _should_regenerate(
    miss_count: int,
    total_count: int,
    threshold: float,
    has_cached: bool,
) -> bool:
    """Return True if synthesis should be regenerated."""
    if not has_cached:
        return True
    if total_count == 0:
        return False
    ratio = miss_count / total_count
    logger.debug(
        "Change ratio: %.1f%% (%d/%d) vs threshold %.1f%%",
        ratio * 100,
        miss_count,
        total_count,
        threshold * 100,
    )
    return ratio >= threshold


def _format_descriptions(descriptions: dict[str, str]) -> str:
    return "\n".join(f"{path}: {desc}" for path, desc in sorted(descriptions.items()))


async def synthesize(
    provider: LLMProvider,
    project_name: str,
    instructions: str,
    user_conventions: str,
    tree_lines: list[str],
    descriptions: dict[str, str],
    cache: SynthesisCache,
    miss_count: int,
    regen_threshold: float,
) -> SynthesisResult:
    """Generate or retrieve use-cases and conventions.

    Args:
        provider: LLM provider to use.
        project_name: Human name for the project.
        instructions: User-provided instructions from config.
        user_conventions: User-provided conventions from config.
        tree_lines: File tree lines from scanner.
        descriptions: Full file description mapping.
        cache: Synthesis cache (will be mutated if regenerated).
        miss_count: Number of files that were cache misses in the describe step.
        regen_threshold: Fraction of changed files that triggers regeneration.

    Returns:
        SynthesisResult with use_cases, conventions_ai, and regenerated flag.
    """
    total = len(descriptions)
    should_regen = _should_regenerate(
        miss_count=miss_count,
        total_count=total,
        threshold=regen_threshold,
        has_cached=cache.record is not None,
    )

    if not should_regen:
        assert cache.record is not None
        logger.info(
            "Synthesis cache reused (%.1f%% changed < %.1f%% threshold)",
            (miss_count / total * 100) if total else 0,
            regen_threshold * 100,
        )
        return SynthesisResult(
            use_cases=cache.record.use_cases,
            conventions_ai=cache.record.conventions_ai,
            regenerated=False,
        )

    logger.info("Regenerating synthesis (%d/%d files changed)", miss_count, total)

    tree_str = "\n".join(tree_lines)
    descs_str = _format_descriptions(descriptions)

    use_cases_msg, conventions_msg = await _call_both(
        provider=provider,
        project_name=project_name,
        tree_str=tree_str,
        descs_str=descs_str,
        instructions=instructions,
        user_conventions=user_conventions,
    )

    record = SynthesisRecord(use_cases=use_cases_msg, conventions_ai=conventions_msg)
    cache.update(record)

    return SynthesisResult(
        use_cases=use_cases_msg,
        conventions_ai=conventions_msg,
        regenerated=True,
    )


async def _call_both(
    provider: LLMProvider,
    project_name: str,
    tree_str: str,
    descs_str: str,
    instructions: str,
    user_conventions: str,
) -> tuple[str, str]:
    """Fire use-cases and conventions calls concurrently."""
    import asyncio

    use_cases, conventions = await asyncio.gather(
        provider.complete(_use_cases_prompt(project_name, tree_str, descs_str, instructions), 1024),
        provider.complete(_conventions_prompt(project_name, tree_str, descs_str, user_conventions), 1024),
    )
    return use_cases, conventions
