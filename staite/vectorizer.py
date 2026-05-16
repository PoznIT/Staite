"""Parse STATE.json into chunks and upsert into ChromaDB.

Client mode is determined by environment:
  CHROMA_HOST set  → HttpClient (Docker Compose, remote)
  CHROMA_HOST unset → PersistentClient (local dev)
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "staite"
_MODEL_NAME = "all-MiniLM-L6-v2"

_SECTION_KEYS: list[tuple[str, str]] = [
    ("use_cases", "overview"),
    ("file_tree", "file_tree"),
    ("architecture", "diagram"),
]


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


def _collection_is_empty(db_path: Path) -> bool:
    """Return True if the collection doesn't exist or has no documents."""
    try:
        client = _make_client(db_path)
        return client.get_collection(_COLLECTION_NAME).count() == 0
    except Exception:
        return True


def _parse_chunks(json_path: Path) -> list[dict]:
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    chunks: list[dict] = []

    conv = doc.get("conventions", {})
    parts = []
    if conv.get("user", "").strip():
        parts.append(f"[User-defined]\n{conv['user'].strip()}")
    if conv.get("ai", "").strip():
        parts.append(f"[AI-inferred]\n{conv['ai'].strip()}")
    if parts:
        chunks.append({"id": "conventions", "text": "\n\n".join(parts), "metadata": {"type": "conventions"}})

    for key, chunk_type in _SECTION_KEYS:
        text = doc.get(key, "")
        if text and text.strip():
            chunks.append({"id": chunk_type, "text": text.strip(), "metadata": {"type": chunk_type}})

    for path, desc in doc.get("files", {}).items():
        if desc and desc.strip():
            chunks.append({"id": f"file:{path}", "text": desc.strip(), "metadata": {"type": "file", "path": path}})

    logger.debug("Parsed %d chunks from %s", len(chunks), json_path)
    return chunks


def build_index(json_path: Path, db_path: Path) -> None:
    """Embed STATE.json chunks and upsert into ChromaDB."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Vector deps not installed. Run: pip install 'staite[vector]'") from exc

    if not json_path.exists():
        raise FileNotFoundError(f"STATE.json not found at {json_path}. Run 'staite run' first.")

    chunks = _parse_chunks(json_path)
    if not chunks:
        logger.warning("No chunks extracted from %s — index will be empty", json_path)
        return

    logger.info("Loading embedding model %s …", _MODEL_NAME)
    model = SentenceTransformer(_MODEL_NAME)

    logger.info("Embedding %d chunks …", len(chunks))
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=False).tolist()

    client = _make_client(db_path)
    collection = client.get_or_create_collection(_COLLECTION_NAME)
    collection.upsert(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    logger.info("Index built: %d chunks", len(chunks))


def load_collection(db_path: Path):  # type: ignore[return]
    """Open the ChromaDB collection (raises if not built yet)."""
    client = _make_client(db_path)
    try:
        return client.get_collection(_COLLECTION_NAME)
    except Exception as exc:
        raise FileNotFoundError(
            "Vector index not found. Run 'staite serve' to build it."
        ) from exc
