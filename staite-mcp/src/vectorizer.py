"""VectorClient: a config-aware wrapper around the ChromaDB indexing helpers."""

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import chromadb

if TYPE_CHECKING:
    from chromadb import Collection

from .config import ChromaDbSettings, OllamaSettings  # adjust import path as needed

logger = logging.getLogger(__name__)

_COLLECTION_PREFIX = "staite__"
_MODEL_NAME = "all-MiniLM-L6-v2"


def _collection_name(project_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", project_name.lower()).strip("_")
    return f"{_COLLECTION_PREFIX}{slug or 'default'}"


class VectorClient:
    """Manages embedding and retrieval against a ChromaDB instance.

    Connection details are taken from a :class:`ChromaDbSettings` object so
    they flow from ``config.dev.yml`` / environment variables rather than being
    hard-coded or read from ``os.getenv`` directly.

    Usage::

        from staite.config import AppSettings
        from staite.vector_client import VectorClient

        settings = AppSettings()
        client = VectorClient(settings.chroma)
        client.build_index_from_dict(state_dict)
    """

    def __init__(self, chroma_db_settings: ChromaDbSettings, ollamaSettings: OllamaSettings) -> None:
        self.__chroma_db_settings = chroma_db_settings
        self.__ollama_settings = ollamaSettings
        self.__chroma: chromadb.ClientAPI | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _chroma(self) -> chromadb.ClientAPI:
        """Lazy-initialised ChromaDB client."""
        if self.__chroma is None:
            host = self.__chroma_db_settings.host
            port = self.__chroma_db_settings.port
            logger.debug("Connecting to ChromaDB at %s:%d", host, port)
            self.__chroma = chromadb.HttpClient(host=host, port=port)
        return self.__chroma

    @staticmethod
    def extract_overview_data(doc: dict) -> dict:
        """Extract structured non-file metadata for PostgreSQL storage.

        Returns a dict with keys ``use_cases``, ``conventions``, ``diagram``,
        and ``file_tree`` (each a ``str | None``).
        """
        conv = doc.get("conventions", {})
        parts = []
        if conv.get("user", "").strip():
            parts.append(f"[User-defined]\n{conv['user'].strip()}")
        if conv.get("ai", "").strip():
            parts.append(f"[AI-inferred]\n{conv['ai'].strip()}")
        conventions_text = "\n\n".join(parts) if parts else None

        return {
            "use_cases": doc.get("use_cases", "").strip() or None,
            "conventions": conventions_text,
            "diagram": doc.get("architecture", "").strip() or None,
            "file_tree": doc.get("file_tree", "").strip() or None,
        }

    @staticmethod
    def _parse_chunks_from_dict(doc: dict, project_name: str) -> list[dict]:
        """Build ChromaDB-bound chunks — file descriptions only.

        Structured metadata (overview, conventions, diagram, file_tree) is
        intentionally excluded here; use :meth:`extract_overview_data` to
        retrieve that data and persist it to PostgreSQL instead.
        """
        chunks: list[dict] = []

        for path, desc in doc.get("files", {}).items():
            if desc and desc.strip():
                chunks.append({
                    "id": f"file:{path}",
                    "text": desc.strip(),
                    "metadata": {"type": "file", "path": path, "project": project_name},
                })

        logger.debug("Parsed %d file chunks for project %r", len(chunks), project_name)
        return chunks

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        with httpx.Client(timeout=60) as client:
            for text in texts:
                response = client.post(
                    f"{self.__ollama_settings.url}/api/embed",
                    json={
                        "model": self.__ollama_settings.model,
                        "input": text,  # single string, not a list
                        "options": {"num_ctx": self.__ollama_settings.num_ctx},
                    },
                )
                response.raise_for_status()
                embeddings.append(response.json()["embeddings"][0])
        return embeddings

    def _embed_and_upsert(self, chunks: list[dict], project_name: str) -> tuple[str, int]:
        col_name = _collection_name(project_name)
        if not chunks:
            logger.warning("No chunks for project %r — index will be empty", project_name)
            return col_name, 0

        logger.info("Embedding %d chunks via Ollama (%s) …", len(chunks), self.__ollama_settings.model)
        embeddings = self._embed_batch([c["text"] for c in chunks])  # single HTTP call

        collection = self._chroma.get_or_create_collection(col_name)
        collection.upsert(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        logger.info("Index built for %r: %d chunks in collection %r", project_name, len(chunks), col_name)
        return col_name, len(chunks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self, json_path: Path) -> str:
        """Embed a STATE.json file and upsert into the project's collection.

        Returns the collection name used.
        """
        if not json_path.exists():
            raise FileNotFoundError(
                f"STATE.json not found at {json_path}. Run 'staite run' first."
            )
        import json

        doc = json.loads(json_path.read_text(encoding="utf-8"))
        project_name: str = doc["metadata"]["name"]
        chunks = self._parse_chunks_from_dict(doc, project_name)
        col_name, _ = self._embed_and_upsert(chunks, project_name)
        return col_name

    def build_index_from_dict(self, state: dict) -> tuple[str, int]:
        """Like :meth:`build_index` but accepts a parsed state dict.

        Returns ``(collection_name, chunk_count)``.
        """
        project_name = state.get("metadata", {}).get("name")
        if not project_name:
            raise ValueError("state dict is missing metadata.name")
        chunks = self._parse_chunks_from_dict(state, project_name)
        return self._embed_and_upsert(chunks, project_name)

    def load_collection(self, collection_name: str) -> "Collection":
        """Open a ChromaDB collection by name (raises if not built yet)."""
        try:
            return self._chroma.get_collection(collection_name)
        except Exception as exc:
            raise FileNotFoundError(
                f"Vector index {collection_name!r} not found. "
                "Run 'staite serve' to build it."
            ) from exc

    def collection_is_empty(self, collection_name: str) -> bool:
        """Return ``True`` if the collection doesn't exist or has no documents."""
        try:
            return self._chroma.get_collection(collection_name).count() == 0
        except Exception:
            return True

    def list_indexed_projects(self) -> list[str]:
        """Return project names for all collections with the ``staite__`` prefix."""
        try:
            return [
                col.name[len(_COLLECTION_PREFIX):]
                for col in self._chroma.list_collections()
                if col.name.startswith(_COLLECTION_PREFIX)
            ]
        except Exception:
            return []