from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from contextlib import suppress
from pathlib import Path


def cache_root() -> Path:
    """Return the persistent cache directory without creating it."""
    override = os.environ.get("METILE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Caches" / "metile"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    return Path(xdg_cache).expanduser() / "metile" if xdg_cache else Path.home() / ".cache/metile"


def stable_digest(payload) -> str:
    """Hash a JSON-serializable cache identity deterministically."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path, default):
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except (OSError, ValueError, TypeError):
        return default


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically replace a cache entry so readers never observe partial data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        with suppress(FileNotFoundError):
            os.unlink(temporary_path)


def atomic_write_json(path: Path, payload) -> None:
    content = json.dumps(payload, indent=2, sort_keys=True).encode() + b"\n"
    atomic_write_bytes(path, content)
