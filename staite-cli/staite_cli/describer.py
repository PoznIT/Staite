"""AI-powered file description generator.

Uses the Anthropic async SDK to produce concise file descriptions.
Description length scales with file size:

    < 30 lines   → 2 sentences
    30–150 lines → 4 sentences
    > 150 lines  → 6 sentences

Results are cached by file hash — only changed/new files hit the API.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import aiofiles

from staite_cli.cache import DescriptionCache, hash_file
from staite_cli.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# Concurrency caps per provider — Azure is more sensitive to parallel connections
CONCURRENCY_ANTHROPIC = 10
CONCURRENCY_AZURE = 5


@dataclass
class DescribeResult:
    descriptions: dict[str, str]
    cache_miss_count: int

_SENTENCE_THRESHOLDS: list[tuple[int, int]] = [
    (30, 2),   # < 30 lines  → 2 sentences
    (150, 4),  # < 150 lines → 4 sentences
]
_SENTENCE_DEFAULT = 6


def sentence_count(line_count: int) -> int:
    for threshold, count in _SENTENCE_THRESHOLDS:
        if line_count < threshold:
            return count
    return _SENTENCE_DEFAULT


def _build_prompt(rel_path: str, content: str, sentences: int) -> str:
    return (
        f"You are analysing a source file in a software project.\n"
        f"File: {rel_path}\n\n"
        f"<file_content>\n{content}\n</file_content>\n\n"
        f"Write exactly {sentences} sentence(s) describing this file. "
        f"Focus on: its purpose, the key classes/functions/exports it defines, "
        f"and how it fits into the broader project. "
        f"Be dense and precise — this description will be read by an AI assistant, not a human."
    )


async def _describe_one(
    provider: LLMProvider,
    rel_path: str,
    abs_path: Path,
    cache: DescriptionCache,
    semaphore: asyncio.Semaphore,
) -> tuple[str, str, bool]:
    """Describe a single file, using the cache when possible.

    Returns:
        (rel_path, description, was_cache_miss)
    """
    file_hash = hash_file(abs_path)
    cached = cache.get(rel_path, file_hash)
    if cached is not None:
        return rel_path, cached, False

    async with semaphore:
        async with aiofiles.open(abs_path, encoding="utf-8", errors="replace") as fh:
            content = await fh.read()

        line_count = content.count("\n") + 1
        sentences = sentence_count(line_count)

        logger.debug("Describing %s (%d lines → %d sentences)", rel_path, line_count, sentences)

        description = await provider.complete(
            prompt=_build_prompt(rel_path, content, sentences),
            max_tokens=512,
        )

    cache.set(rel_path, file_hash, description)
    logger.debug("Described: %s", rel_path)
    return rel_path, description, True


async def describe_files(
    provider: LLMProvider,
    root: Path,
    rel_paths: list[Path],
    cache: DescriptionCache,
    concurrency: int,
    on_file_done: Callable[[str, bool], None] | None = None,
) -> DescribeResult:
    """Describe all files, returning descriptions and the number of cache misses.

    Args:
        provider: LLM provider to use for descriptions.
        root: Project root directory.
        rel_paths: Files to describe, as paths relative to root.
        cache: Description cache (will be mutated with new entries).
        on_file_done: Optional callback invoked after each file with
            ``(rel_path, was_cache_miss)``.  Useful for progress reporting.

    Returns:
        DescribeResult with descriptions dict and cache_miss_count.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _task(p: Path):
        result = await _describe_one(
            provider=provider,
            rel_path=p.as_posix(),
            abs_path=root / p,
            cache=cache,
            semaphore=semaphore,
        )
        if on_file_done:
            on_file_done(result[0], result[2])  # rel_path, was_miss
        return result

    tasks = [_task(p) for p in rel_paths]

    descriptions: dict[str, str] = {}
    miss_count = 0
    total = len(tasks)
    logger.debug("Describing %d file(s) (cache may reduce API calls)", total)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    errors = []
    for result in results:
        if isinstance(result, BaseException):
            errors.append(result)
            continue
        rel_path, description, was_miss = result
        descriptions[rel_path] = description
        if was_miss:
            miss_count += 1

    if errors:
        raise errors[0]

    logger.debug(
        "Descriptions complete: %d file(s) processed, %d cache miss(es)",
        total,
        miss_count,
    )
    return DescribeResult(descriptions=descriptions, cache_miss_count=miss_count)
