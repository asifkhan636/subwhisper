"""Subtitle handling pipeline utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pysubs2
import textwrap


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


def enforce_limits(
    subs: pysubs2.SSAFile,
    max_chars: int,
    max_lines: int,
    max_duration: float,
    min_gap: float,
) -> pysubs2.SSAFile:
    """Enforce basic formatting limits on ``subs`` in-place.

    Parameters
    ----------
    subs:
        Subtitle collection to mutate.
    max_chars:
        Maximum characters allowed per line.
    max_lines:
        Maximum number of lines per event.
    max_duration:
        Maximum duration for a single event in seconds.
    min_gap:
        Minimum required gap between consecutive events in seconds.

    Returns
    -------
    pysubs2.SSAFile
        The modified subtitle file (same object as ``subs``).
    """

    max_ms = int(max_duration * 1000)
    gap_ms = int(min_gap * 1000)

    # First, wrap the text of each event and clip overly long durations.
    for ev in subs.events:
        lines = textwrap.wrap(ev.plaintext, width=max_chars)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        ev.text = "\\N".join(lines)
        if ev.end - ev.start > max_ms:
            ev.end = ev.start + max_ms

    # Then enforce minimum gaps and re-check durations. Merge events that
    # would end up empty after shifting for the gap.
    i = 1
    while i < len(subs.events):
        prev = subs.events[i - 1]
        curr = subs.events[i]
        required_start = prev.end + gap_ms
        if curr.start < required_start:
            curr.start = required_start
            if curr.start >= curr.end:
                prev.text = prev.plaintext + "\\N" + curr.plaintext
                prev.end = max(prev.end, curr.end)
                subs.events.pop(i)
                lines = textwrap.wrap(prev.plaintext, width=max_chars)
                if len(lines) > max_lines:
                    lines = lines[:max_lines]
                prev.text = "\\N".join(lines)
                if prev.end - prev.start > max_ms:
                    prev.end = prev.start + max_ms
                continue
        if curr.end - curr.start > max_ms:
            curr.end = curr.start + max_ms
        i += 1

    return subs


def spellcheck_lines(subs: pysubs2.SSAFile, lang: str = "en-US") -> pysubs2.SSAFile:
    """Spell-check subtitle lines using `language_tool_python`.

    Parameters
    ----------
    subs:
        Subtitle collection to mutate.
    lang:
        Language code understood by `language_tool_python.LanguageTool`.

    Returns
    -------
    pysubs2.SSAFile
        The modified subtitle file (same object as ``subs``).
    """

    import language_tool_python

    tool = language_tool_python.LanguageTool(lang)
    for ev in subs.events:
        corrected = tool.correct(ev.plaintext)
        ev.text = corrected.replace("\n", "\\N")
    return subs
