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


def _disk_cache_disabled() -> bool:
    return os.environ.get("METILE_DISABLE_DISK_CACHE") == "1"


def read_cached_selection(path: Path, key: str):
    """Return the payload persisted under `key`, or None if absent or caching is off."""
    if _disk_cache_disabled():
        return None
    payload = read_json(path, {}).get(key)
    return payload if isinstance(payload, dict) else None


def write_cached_selection(path: Path, key: str, payload) -> None:
    """Persist `payload` under `key`, unless caching is off."""
    if _disk_cache_disabled():
        return
    stored = read_json(path, {})
    stored[key] = payload
    atomic_write_json(path, stored)


def read_cached_config(path: Path, key: str, configs):
    """Recover a tuned config that was persisted as its full attribute dict.

    A config only matches when every attribute agrees, so a config family that gained or
    dropped a field misses the cache and re-tunes rather than restoring a stale choice.
    """
    payload = read_cached_selection(path, key)
    if payload is None:
        return None
    return next((config for config in configs if vars(config) == payload), None)


def write_cached_config(path: Path, key: str, config) -> None:
    """Persist `config` by its full attribute dict."""
    write_cached_selection(path, key, vars(config))


def read_cached_algorithm_config(path: Path, key: str, configs):
    """Recover a tuned config identified by algorithm and block size alone.

    Used by the kernel families whose configs carry measurement fields that must not take
    part in cache identity.
    """
    payload = read_cached_selection(path, key)
    if payload is None:
        return None
    return next(
        (
            config
            for config in configs
            if config.algorithm == payload.get("algorithm")
            and config.block == payload.get("block", 0)
        ),
        None,
    )


def write_cached_algorithm_config(path: Path, key: str, config) -> None:
    """Persist `config` by algorithm and block size alone."""
    write_cached_selection(path, key, {"algorithm": config.algorithm, "block": config.block})
