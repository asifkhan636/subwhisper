"""Utility functions for tracking pipeline state.

The module implements a very small manifest format used to cache
intermediate results of processing stages.  Each manifest is stored
inside ``<input_dir>/.subwhisper_cache/<stem>.manifest.json`` and
contains information about the source file, parameters and outputs of
completed stages.

Example manifest structure::

    {
        "source": {
            "path": "/abs/path/to/input.mp4",
            "size": 1234,
            "mtime_ns": 1700000000,
            "sha1": "..."
        },
        "stages": {
            "preprocess": {
                "params": "<sha1>",
                "outputs": ["/abs/path/to/output.wav"]
            }
        }
    }

"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence, Tuple, List

logger = logging.getLogger(__name__)

CACHE_DIR = ".subwhisper_cache"
MANIFEST_SUFFIX = ".manifest.json"
_MAX_HASH_SIZE = 16 * 1024 * 1024  # 16 MB


# ---------------------------------------------------------------------------
# Fingerprints and hashing
# ---------------------------------------------------------------------------

def compute_source_fingerprint(path: str) -> Dict[str, Any]:
    """Return a fingerprint for *path*.

    Parameters
    ----------
    path:
        File to fingerprint.

    Returns
    -------
    dict
        ``{"size": int, "mtime_ns": int, "sha1": str}``
    """
    p = Path(path)
    stat = p.stat()
    with p.open("rb") as handle:
        data = handle.read(_MAX_HASH_SIZE)
    sha1 = hashlib.sha1(data).hexdigest()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha1": sha1}


def hash_params(mapping: Mapping[str, Any]) -> str:
    """Hash *mapping* using sorted JSON and return a SHA1 hexdigest."""
    encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def _manifest_path(input_dir: Path, stem: str) -> Path:
    input_dir = Path(input_dir).resolve()
    cache_dir = input_dir / CACHE_DIR
    return cache_dir / f"{stem}{MANIFEST_SUFFIX}"


def load_manifest(input_dir: Path, stem: str) -> Dict[str, Any]:
    """Load manifest for *stem* inside *input_dir*.

    Returns an empty dictionary when the manifest does not exist.
    """
    manifest_file = _manifest_path(input_dir, stem)
    try:
        with manifest_file.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return {}


def save_manifest(input_dir: Path, stem: str, manifest: MutableMapping[str, Any]) -> Path:
    """Write *manifest* to disk and return its path."""
    manifest_file = _manifest_path(input_dir, stem)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with manifest_file.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    return manifest_file


def stage_complete(
    input_dir: Path,
    stem: str,
    stage: str,
    source_path: str,
    params: Mapping[str, Any],
    outputs: Sequence[str],
) -> Dict[str, Any]:
    """Record completion of *stage* and update the manifest.

    Parameters
    ----------
    input_dir:
        Directory holding the source file.
    stem:
        Base name used for manifest file.
    stage:
        Name of the completed stage.
    source_path:
        Path to the original input whose fingerprint is tracked.
    params:
        Parameters used for the stage.
    outputs:
        Sequence of file paths produced by the stage.

    Returns
    -------
    dict
        The updated manifest.
    """
    manifest = load_manifest(input_dir, stem)

    source_abs = str(Path(source_path).resolve())
    manifest["source"] = {"path": source_abs, **compute_source_fingerprint(source_path)}

    stage_info = {
        "params": hash_params(dict(params)),
        "outputs": [str(Path(p).resolve()) for p in outputs],
    }
    manifest.setdefault("stages", {})[stage] = stage_info

    save_manifest(input_dir, stem, manifest)
    return manifest


def is_stage_reusable(
    input_dir: Path,
    stem: str,
    stage: str,
    source_path: str,
    params: Mapping[str, Any],
) -> Tuple[bool, List[str]]:
    """Check whether *stage* results can be reused.

    Returns ``(True, outputs)`` when reusable, otherwise ``(False, [])``.
    """
    manifest = load_manifest(input_dir, stem)
    if not manifest:
        return False, []

    stored_source = manifest.get("source")
    if not stored_source:
        return False, []
    current_fp = compute_source_fingerprint(source_path)
    source_abs = str(Path(source_path).resolve())
    if stored_source.get("path") != source_abs:
        return False, []
    for key, value in current_fp.items():
        if stored_source.get(key) != value:
            return False, []

    stage_entry = manifest.get("stages", {}).get(stage)
    if not stage_entry:
        return False, []
    if stage_entry.get("params") != hash_params(dict(params)):
        return False, []

    outputs = stage_entry.get("outputs", [])
    if not all(Path(p).exists() for p in outputs):
        return False, []
    return True, list(outputs)


# ---------------------------------------------------------------------------
# Locking
# ---------------------------------------------------------------------------

def _lock_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(manifest_path.suffix + ".lock")


def acquire_lock(manifest_path: Path, timeout: float = 10.0, poll_interval: float = 0.1) -> Path:
    """Acquire an exclusive lock for *manifest_path*.

    The function creates ``<manifest>.lock`` and waits up to *timeout*
    seconds if the lock already exists.  The path to the lock file is
    returned when the lock is acquired.
    """
    lock_path = _lock_path(manifest_path)
    start = time.monotonic()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return lock_path
        except FileExistsError:
            if time.monotonic() - start >= timeout:
                raise TimeoutError(f"Could not acquire lock: {lock_path}")
            time.sleep(poll_interval)


def release_lock(manifest_path: Path) -> None:
    """Release the lock for *manifest_path*."""
    lock_path = _lock_path(manifest_path)
    try:
        lock_path.unlink()
    except FileNotFoundError:  # pragma: no cover - nothing to release
        pass


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def clean_old_manifests(cache_root: Path, days: int) -> int:
    """Remove manifests older than *days* and return the number removed."""
    cutoff = time.time() - days * 86400
    removed = 0
    for manifest in cache_root.rglob(f"*{MANIFEST_SUFFIX}"):
        try:
            if manifest.stat().st_mtime < cutoff:
                manifest.unlink()
                lock_file = _lock_path(manifest)
                if lock_file.exists():
                    lock_file.unlink()
                removed += 1
        except FileNotFoundError:  # pragma: no cover - file disappeared
            continue
    return removed

