"""Subtitle handling pipeline utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pysubs2


def load_segments(path: Path) -> pysubs2.SSAFile:
    """Load segments from a Whisper JSON output.

    Parameters
    ----------
    path:
        Path to a JSON file containing either a list of segment dictionaries
        or the full Whisper transcription result with a ``segments`` field.

    Returns
    -------
    pysubs2.SSAFile
        Subtitle data parsed by :func:`pysubs2.load_from_whisper`.
    """
    with path.open() as f:
        data: Any = json.load(f)

    # `pysubs2.load_from_whisper` can work with either a list of segment
    # dictionaries or a full Whisper result containing a ``segments`` field.
    if isinstance(data, dict) and "segments" in data:
        data = data["segments"]

    return pysubs2.load_from_whisper(data)
