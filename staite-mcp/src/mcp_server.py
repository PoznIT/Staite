"""MCP server exposing one or more StAIte vector indexes as queryable tools.

Six tools:
  search(query, k=5, project=None)  — semantic search over file descriptions
  get_file(path, project=None)       — direct lookup of a file's description
  get_overview(project)              — use-cases / high-level purpose for a project
  get_conventions(project)           — coding conventions and architectural patterns
  get_diagram(project)               — architecture diagram (Mermaid)
  get_file_tree(project)             — file tree
  list_projects()                    — list all indexed projects

Storage split:
  ChromaDB  — file-description vectors (search / get_file)
  PostgreSQL — projects registry + overviews / conventions / diagrams / file trees

When running with --transport sse or --transport http, a POST /update endpoint is also
available on the same port for pushing a new STATE.json payload without restarting.
"""

from __future__ import annotations

import json
import logging
import httpx
from functools import lru_cache

from mcp.server.fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.config import AppSettings, OllamaSettings
from src.vectorizer import VectorClient, _collection_name
from src import db

logger = logging.getLogger(__name__)


def _embed(text: str, ollama: OllamaSettings) -> list[float]:
    with httpx.Client(timeout=30) as client:
        response = client.post(
            f"{ollama.url}/api/embed",
            json={"model": ollama.model, "input": [text]},
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]


def _get_server(config: AppSettings) -> FastMCP:
    """Build the FastMCP server instance."""

    vector_client = VectorClient(config.chroma, config.ollama)
    mcp = FastMCP("staite", host=config.server.host, port=config.server.port)

    # ------------------------------------------------------------------
    # MCP tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def search(query: str, k: int = 5, project: str | None = None) -> str:
        """Semantic search over codebase file descriptions and summaries.

        Returns the top-k most relevant chunks across all indexed projects.
        Pass project= to restrict results to a single project.
        """
        embedding = _embed(query, config.ollama)

        if project:
            project_names = [project]
        else:
            pool = await db.init_pool(config.postgres)
            project_names = await db.project_names(pool)

        if not project_names:
            return json.dumps([])

        all_items: list[dict] = []
        for proj_name in project_names:
            try:
                col = vector_client.load_collection(_collection_name(proj_name))
            except FileNotFoundError:
                continue
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
    async def get_file(path: str, project: str | None = None) -> str:
        """Return the AI-generated description for a specific file path.

        path should be the relative path as it appears in STATE.json,
        e.g. 'staite/config.py' or 'src/auth/handler.ts'.
        If project is omitted and multiple projects are loaded, all matches are returned.
        """
        if project:
            project_names = [project]
        else:
            pool = await db.init_pool(config.postgres)
            project_names = await db.project_names(pool)

        items: list[dict] = []
        for proj_name in project_names:
            try:
                col = vector_client.load_collection(_collection_name(proj_name))
            except FileNotFoundError:
                continue
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

    async def _fetch_overview_field(project: str, field: str) -> str:
        pool = await db.init_pool(config.postgres)
        overview = await db.get_overview(pool, project)
        if overview is None:
            available = await db.project_names(pool)
            return json.dumps({"error": f"Unknown project {project!r}. Available: {available}"})
        value = overview.get(field)
        if not value:
            return json.dumps({"error": f"No {field} found for project {project!r}"})
        return json.dumps({"project": project, field: value}, indent=2)

    @mcp.tool()
    async def get_overview(project: str) -> str:
        """Return the use-cases / high-level purpose description for a project.

        project must be one of the indexed project names (see list_projects).
        """
        return await _fetch_overview_field(project, "use_cases")

    @mcp.tool()
    async def get_conventions(project: str) -> str:
        """Return the coding conventions and architectural patterns for a project.

        project must be one of the indexed project names (see list_projects).
        """
        return await _fetch_overview_field(project, "conventions")

    @mcp.tool()
    async def get_diagram(project: str) -> str:
        """Return the architecture diagram (Mermaid) for a project.

        project must be one of the indexed project names (see list_projects).
        """
        return await _fetch_overview_field(project, "diagram")

    @mcp.tool()
    async def get_file_tree(project: str) -> str:
        """Return the file tree for a project.

        project must be one of the indexed project names (see list_projects).
        """
        return await _fetch_overview_field(project, "file_tree")

    @mcp.tool()
    async def list_projects() -> str:
        """List all projects currently indexed in the vector store."""
        pool = await db.init_pool(config.postgres)
        return json.dumps(await db.list_projects(pool), indent=2)

    # ------------------------------------------------------------------
    # CORS preflight routes
    # ------------------------------------------------------------------

    @mcp.custom_route("/sse", methods=["OPTIONS"])
    async def sse_cors_preflight(request: Request) -> Response:
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age": "86400",
            },
        )

    @mcp.custom_route("/mcp", methods=["OPTIONS"])
    async def mcp_cors_preflight(request: Request) -> Response:
        return Response(
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Max-Age": "86400",
            },
        )

    # ------------------------------------------------------------------
    # /update endpoint — writes to both ChromaDB (file vectors) and PG
    # ------------------------------------------------------------------

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

        # 1. Embed file descriptions → ChromaDB
        try:
            _col_name, file_chunk_count = vector_client.build_index_from_dict(state)
        except Exception as exc:
            logger.exception("Failed to re-index project %r in ChromaDB", project_name)
            return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)

        # 2. Persist structured metadata → PostgreSQL
        try:
            pool = await db.init_pool(config.postgres)
            overview = VectorClient.extract_overview_data(state)
            db_action, indexed_at = await db.upsert_project(pool, project_name, file_chunk_count)
            overview_fields = await db.upsert_overview(pool, project_name, **overview)
        except Exception as exc:
            logger.exception("Failed to persist project %r to PostgreSQL", project_name)
            return JSONResponse({"status": "error", "message": f"DB error: {exc}"}, status_code=500)

        logger.info(
            "%s index and DB for %r via /update (%d file chunks, fields: %s)",
            db_action.capitalize(),
            project_name,
            file_chunk_count,
            overview_fields,
        )
        return JSONResponse({
            "status": "ok",
            "project": project_name,
            "chunks": file_chunk_count,
            "db": {
                "action": db_action,
                "indexed_at": indexed_at,
                "overview_fields": overview_fields,
            },
        })

    return mcp


def mcp_server(config: AppSettings) -> None:
    """Start the MCP server (blocking). transport: 'sse' or 'http'."""
    mcp = _get_server(config)
    mcp.run(
        transport="streamable-http" if config.server.transport == "http" else config.server.transport
    )
