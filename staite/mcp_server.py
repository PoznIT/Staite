"""MCP server exposing the StAIte vector index as queryable tools.

Three tools:
  search(query, k=5)   — semantic search over file descriptions and summaries
  get_file(path)       — direct lookup of a specific file's description
  get_overview()       — returns use_cases, conventions, and architecture chunks
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(".staite/vector_db")
_DEFAULT_JSON_PATH = Path(".staite/STATE.json")
_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(_MODEL_NAME)


def _embed(text: str) -> list[float]:
    return _embedding_model().encode(text, show_progress_bar=False).tolist()


def _get_server(db_path: Path, json_file_path: Path, host: str = "0.0.0.0", port: int = 8080):
    """Build and return the FastMCP server instance."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise RuntimeError(
            "MCP deps not installed. Run: pip install 'staite[vector]'"
        ) from exc

    from staite.vectorizer import _collection_is_empty, build_index, load_collection

    # Build index on first start if needed.
    if _collection_is_empty(db_path):
        logger.info("Vector index empty — building now from %s …", json_file_path)
        build_index(json_file_path, db_path)

    collection = load_collection(db_path)
    mcp = FastMCP("staite", host=host, port=port)

    @mcp.tool()
    def search(query: str, k: int = 5) -> str:
        """Semantic search over codebase file descriptions and summaries.

        Returns the top-k most relevant chunks (file descriptions, use_cases,
        conventions, or architecture) for the given natural-language query.
        """
        embedding = _embed(query)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            items.append({"metadata": meta, "score": round(1 - dist, 4), "text": doc})
        return json.dumps(items, indent=2)

    @mcp.tool()
    def get_file(path: str) -> str:
        """Return the AI-generated description for a specific file path.

        path should be the relative path as it appears in STATE.json,
        e.g. 'staite/config.py' or 'src/auth/handler.ts'.
        """
        results = collection.get(
            ids=[f"file:{path}"],
            include=["documents", "metadatas"],
        )
        if not results["documents"]:
            return json.dumps({"error": f"File not found in index: {path}"})
        return json.dumps(
            {"path": path, "text": results["documents"][0], "metadata": results["metadatas"][0]},
            indent=2,
        )

    @mcp.tool()
    def get_overview() -> str:
        """Return high-level project summaries: use_cases, conventions, and architecture diagram."""
        overview_ids = ["overview", "conventions", "diagram"]
        results = collection.get(
            ids=overview_ids,
            include=["documents", "metadatas"],
        )
        items = [
            {"id": id_, "text": doc, "metadata": meta}
            for id_, doc, meta in zip(
                overview_ids, results["documents"], results["metadatas"]
            )
            if doc
        ]
        return json.dumps(items, indent=2)

    return mcp


def serve(
    db_path: Path = _DEFAULT_DB_PATH,
    json_file_path: Path = _DEFAULT_JSON_PATH,
    transport: str = "stdio",
    host: str = "0.0.0.0",
    port: int = 8080,
) -> None:
    """Start the MCP server (blocking). transport: 'stdio', 'sse', or 'http'."""
    mcp = _get_server(db_path, json_file_path, host=host, port=port)
    # FastMCP uses "streamable-http" internally; accept "http" as the user-facing alias.
    mcp.run(transport="streamable-http" if transport == "http" else transport)
