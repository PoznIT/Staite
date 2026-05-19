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
import httpx
from functools import lru_cache

from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from src.config import AppSettings, OllamaSettings
from src.vectorizer import VectorClient, _collection_name

logger = logging.getLogger(__name__)







def _embed(text: str, ollama: OllamaSettings) -> list[float]:
    with httpx.Client(timeout=30) as client:
        response = client.post(
            f"{ollama.url}/api/embed",
            json={"model": ollama.model, "input": [text]},
        )
        response.raise_for_status()
        return response.json()["embeddings"][0]  # single text → first element


def _get_server(config: AppSettings) -> FastMCP:
    """Build the FastMCP server instance."""

    vector_client = VectorClient(config.chroma, config.ollama)
    mcp = FastMCP("staite", host=config.server.host, port=config.server.port)

    @mcp.tool()
    def search(query: str, k: int = 5, project: str | None = None) -> str:
        """Semantic search over codebase file descriptions and summaries.

        Returns the top-k most relevant chunks across all indexed projects.
        Pass project= to restrict results to a single project.
        """
        embedding = _embed(query, config.ollama)
        project_names = (
            [project] if project else vector_client.list_indexed_projects()
        )

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
    def get_file(path: str, project: str | None = None) -> str:
        """Return the AI-generated description for a specific file path.

        path should be the relative path as it appears in STATE.json,
        e.g. 'staite/config.py' or 'src/auth/handler.ts'.
        If project is omitted and multiple projects are loaded, all matches are returned.
        """
        project_names = (
            [project] if project else vector_client.list_indexed_projects()
        )

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

    @mcp.tool()
    def get_overview(project: str) -> str:
        """Return high-level summaries for a project: use_cases, conventions, and architecture diagram.

        project must be one of the indexed project names (see list_projects).
        """
        try:
            col = vector_client.load_collection(_collection_name(project))
        except FileNotFoundError:
            available = vector_client.list_indexed_projects()
            return json.dumps({"error": f"Unknown project {project!r}. Available: {available}"})

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
        project_names = vector_client.list_indexed_projects()
        info = []
        for proj_name in project_names:
            try:
                col = vector_client.load_collection(_collection_name(proj_name))
                info.append({"project": proj_name, "chunks": col.count()})
            except FileNotFoundError:
                info.append({"project": proj_name, "chunks": 0})
        return json.dumps(info, indent=2)

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
            _col_name, chunk_count = vector_client.build_index_from_dict(state)
        except Exception as exc:
            logger.exception("Failed to re-index project %r", project_name)
            return JSONResponse({"status": "error", "message": str(exc)}, status_code=500)

        logger.info("Updated index for %r via /update (%d chunks)", project_name, chunk_count)
        return JSONResponse({"status": "ok", "project": project_name, "chunks": chunk_count})

    return mcp


def mcp_server(config: AppSettings) -> None:
    """Start the MCP server (blocking). transport: 'sse' or 'http'."""
    mcp = _get_server(config)
    mcp.run(
        transport="streamable-http" if config.server.transport == "http" else config.server.transport
    )