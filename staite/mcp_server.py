"""MCP server exposing one or more StAIte vector indexes as queryable tools.

Four tools:
  search(query, k=5, project=None)  — semantic search; cross-project if project omitted
  get_file(path, project=None)       — direct lookup of a file's description
  get_overview(project)              — use_cases, conventions, and architecture for a project
  list_projects()                    — list all indexed projects

When running with --transport sse or --transport http, a POST /update endpoint is also
available on the same port for pushing a new STATE.json payload without restarting.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path.home() / ".staite" / "vector_db"
_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_NAME)


def _embed(text: str) -> list[float]:
    return _embedding_model().encode(text, show_progress_bar=False).tolist()


def _get_server(
    state_paths: list[Path],
    db_path: Path,
    host: str = "0.0.0.0",
    port: int = 8080,
) -> tuple:
    """Build the FastMCP server instance. Returns (mcp, collections, db_path)."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "MCP deps not installed. Run: pip install 'staite[vector]'"
        ) from exc

    from staite.vectorizer import (
        _collection_is_empty,
        _collection_name,
        build_index,
        get_project_name,
        load_collection,
    )

    # Build / verify index for each state file.
    collections: dict[str, object] = {}  # project_name → chroma collection
    for state_path in state_paths:
        if not state_path.exists():
            logger.warning("STATE.json not found at %s — skipping (push via 'staite update')", state_path)
            continue
        project_name = get_project_name(state_path)
        col_name = _collection_name(project_name)
        if _collection_is_empty(db_path, col_name):
            logger.info(
                "Vector index empty for %r — building from %s …", project_name, state_path
            )
            build_index(state_path, db_path)
        collections[project_name] = load_collection(db_path, col_name)
        logger.info("Loaded index for project %r (%d chunks)", project_name, collections[project_name].count())

    mcp = FastMCP("staite", host=host, port=port)

    @mcp.tool()
    def search(query: str, k: int = 5, project: str | None = None) -> str:
        """Semantic search over codebase file descriptions and summaries.

        Returns the top-k most relevant chunks across all indexed projects.
        Pass project= to restrict results to a single project.
        """
        embedding = _embed(query)
        targets = {project: collections[project]} if project and project in collections else collections

        if not targets:
            return json.dumps([])

        all_items: list[dict] = []
        for proj_name, col in targets.items():
            count = col.count()
            if count == 0:
                continue
            results = col.query(
                query_embeddings=[embedding],
                n_results=min(k, count),
                include=["documents", "metadatas", "distances"],
            )
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                all_items.append({
                    "project": proj_name,
                    "metadata": meta,
                    "score": round(1 - dist, 4),
                    "text": doc,
                })

        all_items.sort(key=lambda x: x["score"], reverse=True)
        return json.dumps(all_items[:k], indent=2)

    @mcp.tool()
    def get_file(path: str, project: str | None = None) -> str:
        """Return the AI-generated description for a specific file path.

        path should be the relative path as it appears in STATE.json,
        e.g. 'staite/config.py' or 'src/auth/handler.ts'.
        If project is omitted and multiple projects are loaded, all matches are returned.
        """
        targets = {project: collections[project]} if project and project in collections else collections
        items: list[dict] = []
        for proj_name, col in targets.items():
            results = col.get(ids=[f"file:{path}"], include=["documents", "metadatas"])
            if results["documents"]:
                items.append({
                    "project": proj_name,
                    "path": path,
                    "text": results["documents"][0],
                    "metadata": results["metadatas"][0],
                })
        if not items:
            return json.dumps({"error": f"File not found in index: {path}"})
        return json.dumps(items if len(items) > 1 else items[0], indent=2)

    @mcp.tool()
    def get_overview(project: str) -> str:
        """Return high-level summaries for a project: use_cases, conventions, and architecture diagram.

        project must be one of the loaded project names (see list_projects).
        """
        if project not in collections:
            available = list(collections.keys())
            return json.dumps({"error": f"Unknown project {project!r}. Available: {available}"})

        col = collections[project]
        overview_ids = ["overview", "conventions", "diagram"]
        results = col.get(ids=overview_ids, include=["documents", "metadatas"])
        items = [
            {"id": id_, "project": project, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(overview_ids, results["documents"], results["metadatas"])
            if doc
        ]
        return json.dumps(items, indent=2)

    @mcp.tool()
    def list_projects() -> str:
        """List all projects currently indexed in the vector store."""
        info = [
            {"project": name, "chunks": col.count()}
            for name, col in collections.items()
        ]
        return json.dumps(info, indent=2)

    from starlette.requests import Request
    from starlette.responses import JSONResponse

    from staite.vectorizer import _collection_name, build_index_from_dict, load_collection

    @mcp.custom_route("/update", methods=["POST"])
    async def update_handler(request: Request) -> JSONResponse:
        try:
            state = await request.json()
        except Exception:
            return JSONResponse({"status": "error", "message": "Invalid JSON body"}, status_code=400)

        project_name = state.get("metadata", {}).get("name")
        if not project_name:
            return JSONResponse(
                {"status": "error", "message": "Missing metadata.name"}, status_code=422
            )

        try:
            col_name, chunk_count = build_index_from_dict(state, db_path)
        except Exception as exc:
            logger.exception("Failed to re-index project %r", project_name)
            return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)

        collections[project_name] = load_collection(db_path, _collection_name(project_name))
        logger.info("Updated index for %r via /update (%d chunks)", project_name, chunk_count)
        return JSONResponse({"status": "ok", "project": project_name, "chunks": chunk_count})

    return mcp, collections, db_path


def serve(
    state_paths: list[Path],
    db_path: Path = _DEFAULT_DB_PATH,
    transport: str = "stdio",
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Start the MCP server (blocking). transport: 'stdio', 'sse', or 'http'."""
    mcp, _collections, _db_path = _get_server(state_paths, db_path, host=host, port=port)
    mcp.run(transport="streamable-http" if transport == "http" else transport)
