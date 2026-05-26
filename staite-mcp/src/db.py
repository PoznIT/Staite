"""PostgreSQL persistence layer for StAIte MCP.

Stores structured project metadata (projects registry, overviews, conventions,
diagrams, file trees) — everything that isn't a raw file-description vector.
ChromaDB continues to own the file-chunk embeddings.
"""

from __future__ import annotations

import logging

import asyncpg

from .config import PostgresSettings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    name             TEXT        PRIMARY KEY,
    last_indexed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    file_chunk_count INTEGER     NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS project_overviews (
    project     TEXT        PRIMARY KEY REFERENCES projects(name) ON DELETE CASCADE,
    use_cases   TEXT,
    conventions TEXT,
    diagram     TEXT,
    file_tree   TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------

_pool: asyncpg.Pool | None = None


async def init_pool(settings: PostgresSettings) -> asyncpg.Pool:
    """Return the shared connection pool, creating and migrating on first call."""
    global _pool
    if _pool is None:
        logger.debug("Creating asyncpg pool → %s", settings.dsn)
        _pool = await asyncpg.create_pool(dsn=settings.dsn)
        async with _pool.acquire() as conn:
            await conn.execute(_SCHEMA)
        logger.info("PostgreSQL pool ready")
    return _pool


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------


async def upsert_project(pool: asyncpg.Pool, name: str, file_chunk_count: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO projects (name, last_indexed_at, file_chunk_count)
            VALUES ($1, now(), $2)
            ON CONFLICT (name) DO UPDATE
                SET last_indexed_at  = now(),
                    file_chunk_count = EXCLUDED.file_chunk_count
            """,
            name,
            file_chunk_count,
        )


async def upsert_overview(
    pool: asyncpg.Pool,
    project: str,
    *,
    use_cases: str | None,
    conventions: str | None,
    diagram: str | None,
    file_tree: str | None,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO project_overviews
                (project, use_cases, conventions, diagram, file_tree, updated_at)
            VALUES ($1, $2, $3, $4, $5, now())
            ON CONFLICT (project) DO UPDATE
                SET use_cases   = EXCLUDED.use_cases,
                    conventions = EXCLUDED.conventions,
                    diagram     = EXCLUDED.diagram,
                    file_tree   = EXCLUDED.file_tree,
                    updated_at  = now()
            """,
            project,
            use_cases,
            conventions,
            diagram,
            file_tree,
        )


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------


async def list_projects(pool: asyncpg.Pool) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT name, last_indexed_at, file_chunk_count FROM projects ORDER BY name"
        )
    return [
        {
            "project": r["name"],
            "chunks": r["file_chunk_count"],
            "last_indexed_at": r["last_indexed_at"].isoformat(),
        }
        for r in rows
    ]


async def get_overview(pool: asyncpg.Pool, project: str) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT use_cases, conventions, diagram, file_tree
            FROM project_overviews
            WHERE project = $1
            """,
            project,
        )
    if row is None:
        return None
    return {
        "project": project,
        "use_cases": row["use_cases"],
        "conventions": row["conventions"],
        "diagram": row["diagram"],
        "file_tree": row["file_tree"],
    }


async def project_names(pool: asyncpg.Pool) -> list[str]:
    """Return all known project names — used to enumerate ChromaDB collections."""
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT name FROM projects ORDER BY name")
    return [r["name"] for r in rows]
