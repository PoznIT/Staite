"""Hash-based description cache.

Persists a mapping of {relative_file_path → {hash, description}} as JSON.
Files whose content hash matches the cache entry are not re-described.
"""

import hashlib
import json
import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_CHUNK = 65_536  # 64 KiB read chunks for hashing


class CacheEntry(BaseModel):
    sha256: str
    description: str


class DescriptionCache:
    """Read/write cache mapping file paths to their AI-generated descriptions.

    Usage::

        cache = DescriptionCache.load(Path(".staite/cache.json"))
        entry = cache.get("src/foo.py", current_hash)
        if entry is None:
            description = await describe(...)
            cache.set("src/foo.py", current_hash, description)
        cache.save()
    """

    def __init__(self, path: Path, entries: dict[str, CacheEntry]) -> None:
        self._path = path
        self._entries = entries
        self._dirty = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "DescriptionCache":
        """Load cache from *path*, or return an empty cache if the file doesn't exist."""
        if path.exists():
            logger.debug("Loading cache from %s", path)
            raw: dict = json.loads(path.read_text(encoding="utf-8"))
            entries = {k: CacheEntry.model_validate(v) for k, v in raw.items()}
            logger.info("Cache loaded: %d entry/entries", len(entries))
        else:
            logger.debug("No cache found at %s — starting fresh", path)
            entries = {}
        return cls(path=path, entries=entries)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, rel_path: str, current_hash: str) -> str | None:
        """Return a cached description if the hash matches, else None."""
        entry = self._entries.get(rel_path)
        if entry is None:
            return None
        if entry.sha256 != current_hash:
            logger.debug("Cache miss (hash changed): %s", rel_path)
            return None
        logger.debug("Cache hit: %s", rel_path)
        return entry.description

    def set(self, rel_path: str, sha256: str, description: str) -> None:
        """Store or update a cache entry."""
        self._entries[rel_path] = CacheEntry(sha256=sha256, description=description)
        self._dirty = True

    def save(self) -> None:
        """Persist the cache to disk. Creates parent directories if needed."""
        if not self._dirty:
            logger.debug("Cache unchanged — skipping write")
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v.model_dump() for k, v in self._entries.items()}
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._dirty = False
        logger.info("Cache saved to %s (%d entries)", self._path, len(self._entries))

    @property
    def size(self) -> int:
        return len(self._entries)


# ------------------------------------------------------------------
# Hashing utility
# ------------------------------------------------------------------


def hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of *path*'s contents."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()
