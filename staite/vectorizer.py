"""Parse STATE.json into chunks and upsert into a per-project ChromaDB collection.

Client mode is determined by environment:
  CHROMA_HOST set  → HttpClient (Docker Compose, remote)
  CHROMA_HOST unset → PersistentClient (local dev)

Collection naming: each project gets its own collection named
``staite__{slug}`` where slug is the project name lowercased with
non-alphanumeric runs replaced by underscores.
"""

import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_COLLECTION_PREFIX = "staite__"
_MODEL_NAME = "all-MiniLM-L6-v2"

_SECTION_KEYS: list[tuple[str, str]] = [
    ("use_cases", "overview"),
    ("file_tree", "file_tree"),
    ("architecture", "diagram"),
]


def _collection_name(project_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", project_name.lower()).strip("_")
    return f"{_COLLECTION_PREFIX}{slug or 'default'}"


def get_project_name(json_path: Path) -> str:
    """Read the project name from a STATE.json file."""
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    return doc["metadata"]["name"]


def _make_client(db_path: Path):
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Vector deps not installed. Run: pip install 'staite[vector]'") from exc

    host = os.getenv("CHROMA_HOST")
    if host:
        port = int(os.getenv("CHROMA_PORT", "8000"))
        logger.debug("Connecting to ChromaDB at %s:%d", host, port)
        return chromadb.HttpClient(host=host, port=port)

    db_path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def _collection_is_empty(db_path: Path, collection_name: str) -> bool:
    """Return True if the collection doesn't exist or has no documents."""
    try:
        client = _make_client(db_path)
        return client.get_collection(collection_name).count() == 0
    except Exception:
        return True


def _parse_chunks_from_dict(doc: dict, project_name: str) -> list[dict]:
    chunks: list[dict] = []

    conv = doc.get("conventions", {})
    parts = []
    if conv.get("user", "").strip():
        parts.append(f"[User-defined]\n{conv['user'].strip()}")
    if conv.get("ai", "").strip():
        parts.append(f"[AI-inferred]\n{conv['ai'].strip()}")
    if parts:
        chunks.append({
            "id": "conventions",
            "text": "\n\n".join(parts),
            "metadata": {"type": "conventions", "project": project_name},
        })

    for key, chunk_type in _SECTION_KEYS:
        text = doc.get(key, "")
        if text and text.strip():
            chunks.append({
                "id": chunk_type,
                "text": text.strip(),
                "metadata": {"type": chunk_type, "project": project_name},
            })

    for path, desc in doc.get("files", {}).items():
        if desc and desc.strip():
            chunks.append({
                "id": f"file:{path}",
                "text": desc.strip(),
                "metadata": {"type": "file", "path": path, "project": project_name},
            })

    logger.debug("Parsed %d chunks for project %r", len(chunks), project_name)
    return chunks


def _parse_chunks(json_path: Path, project_name: str) -> list[dict]:
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    return _parse_chunks_from_dict(doc, project_name)


def _embed_and_upsert(chunks: list[dict], project_name: str, db_path: Path) -> tuple[str, int]:
    """Embed chunks and upsert into ChromaDB. Returns (collection_name, chunk_count)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Vector deps not installed. Run: pip install 'staite[vector]'") from exc

    col_name = _collection_name(project_name)
    if not chunks:
        logger.warning("No chunks for project %r — index will be empty", project_name)
        return col_name, 0

    logger.info("Loading embedding model %s …", _MODEL_NAME)
    model = SentenceTransformer(_MODEL_NAME)

    logger.info("Embedding %d chunks for project %r …", len(chunks), project_name)
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=False).tolist()

    client = _make_client(db_path)
    collection = client.get_or_create_collection(col_name)
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    logger.info("Index built for %r: %d chunks in collection %r", project_name, len(chunks), col_name)
    return col_name, len(chunks)


def build_index(json_path: Path, db_path: Path) -> str:
    """Embed STATE.json chunks and upsert into the project's ChromaDB collection.

    Returns the collection name used.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"STATE.json not found at {json_path}. Run 'staite run' first.")

    project_name = get_project_name(json_path)
    chunks = _parse_chunks(json_path, project_name)
    col_name, _ = _embed_and_upsert(chunks, project_name, db_path)
    return col_name


def build_index_from_dict(state: dict, db_path: Path) -> tuple[str, int]:
    """Like build_index() but accepts a parsed state dict instead of a file path.

    Returns (collection_name, chunk_count).
    """
    project_name = state.get("metadata", {}).get("name")
    if not project_name:
        raise ValueError("state dict is missing metadata.name")
    chunks = _parse_chunks_from_dict(state, project_name)
    return _embed_and_upsert(chunks, project_name, db_path)


def load_collection(db_path: Path, collection_name: str):  # type: ignore[return]
    """Open a ChromaDB collection by name (raises if not built yet)."""
    client = _make_client(db_path)
    try:
        return client.get_collection(collection_name)
    except Exception as exc:
        raise FileNotFoundError(
            f"Vector index {collection_name!r} not found. Run 'staite serve' to build it."
        ) from exc


def list_indexed_projects(db_path: Path) -> list[str]:
    """Return project names for all collections with the staite__ prefix."""
    try:
        client = _make_client(db_path)
        return [
            col.name[len(_COLLECTION_PREFIX):]
            for col in client.list_collections()
            if col.name.startswith(_COLLECTION_PREFIX)
        ]
    except Exception:
        return []
