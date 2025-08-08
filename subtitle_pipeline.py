"""Subtitle handling pipeline utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import pysubs2
import textwrap
from corrections import apply_corrections, load_corrections


logger = logging.getLogger(__name__)


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
    cps_limit = 17

    # First enforce minimum gaps by shifting start times. When shifting would
    # invalidate an event (start >= end), keep the latter half of the event
    # after the required gap.
    i = 1
    while i < len(subs.events):
        prev = subs.events[i - 1]
        curr = subs.events[i]
        required_start = prev.end + gap_ms
        if curr.start < required_start:
            logger.debug(
                "Shifting start of event at %.2fs by %.2fs to enforce gap",
                curr.start / 1000,
                (required_start - curr.start) / 1000,
            )
            original_start = curr.start
            original_end = curr.end
            curr.start = required_start
            if curr.start >= curr.end:
                mid = (original_start + original_end) // 2
                duration = original_end - mid
                curr.start = required_start
                curr.end = required_start + duration
        i += 1

    # Next, split events that violate CPS or maximum duration limits.
    i = 0
    while i < len(subs.events):
        ev = subs.events[i]
        duration = ev.end - ev.start
        dur_sec = duration / 1000 if duration > 0 else 0.001
        cps = len(ev.plaintext) / dur_sec
        ev.event_cps = cps
        if duration > max_ms or cps > cps_limit:
            text = ev.plaintext
            if not text:
                i += 1
                continue
            mid_time = ev.start + duration // 2
            mid_index = len(text) // 2
            split_idx = text.rfind(" ", 0, mid_index)
            if split_idx == -1:
                split_idx = text.find(" ", mid_index)
            if split_idx == -1:
                split_idx = mid_index
            left = text[:split_idx].strip()
            right = text[split_idx:].strip()
            ev1 = pysubs2.SSAEvent(start=ev.start, end=mid_time, text=left)
            ev2 = pysubs2.SSAEvent(start=mid_time, end=ev.end, text=right)
            subs.events[i] = ev1
            subs.events.insert(i + 1, ev2)
            continue
        i += 1

    # Finally, wrap text and clamp line counts and durations.
    for ev in subs.events:
        if ev.end - ev.start > max_ms:
            logger.debug(
                "Trimming duration of event at %.2fs from %.2fs to %.2fs",
                ev.start / 1000,
                (ev.end - ev.start) / 1000,
                max_ms / 1000,
            )
            ev.end = ev.start + max_ms
        lines = textwrap.wrap(ev.plaintext, width=max_chars)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        ev.text = "\\N".join(lines)
        duration = ev.end - ev.start
        dur_sec = duration / 1000 if duration > 0 else 0.001
        ev.event_cps = len(ev.plaintext) / dur_sec

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


def write_outputs(
    subs: pysubs2.SSAFile, out_srt: Path, out_txt: Optional[Path]
) -> None:
    """Write subtitle collection to ``out_srt`` and optionally ``out_txt``.

    Parameters
    ----------
    subs:
        Subtitle collection to write.
    out_srt:
        Destination path for the SRT subtitle file.
    out_txt:
        Optional path where plain text lines are written, one per subtitle
        event. When ``None`` the text file is skipped.
    """

    subs.save(out_srt, format_="srt")
    if out_txt is not None:
        lines = [ev.plaintext for ev in subs.events]
        out_txt.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:  # pragma: no cover - CLI entry point
    """Command-line interface for the subtitle pipeline."""

    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Format Whisper segments into subtitle files",
    )
    parser.add_argument("--segments", help="Path to segments.json file")
    parser.add_argument(
        "--transcript",
        action="store_true",
        help="Also write a plain text transcript next to the SRT output",
    )
    parser.add_argument(
        "--corrections",
        help="JSON or YAML file with text correction rules",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output SRT file (or directory when using --batch-dir)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=45,
        help="Maximum characters per line",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=2,
        help="Maximum number of lines per event",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=6.0,
        help="Maximum duration for an event in seconds",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.15,
        help="Minimum gap between events in seconds",
    )
    parser.add_argument(
        "--skip-music",
        action="store_true",
        help="Skip segments marked as music",
    )
    parser.add_argument(
        "--spellcheck",
        action="store_true",
        help="Run LanguageTool spell check on output",
    )
    parser.add_argument(
        "--batch-dir",
        help="Process all segments.json files under this directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--log-file",
        help="Optional file to write logs to",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)
    if getattr(args, "log_file", None):
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    rules = load_corrections(Path(args.corrections)) if args.corrections else None

    def process_file(seg_path: Path, out_srt: Path) -> None:
        with seg_path.open("r", encoding="utf-8") as f:
            data: Any = json.load(f)
        if isinstance(data, dict) and "segments" in data:
            data = data["segments"]
        if args.skip_music:
            filtered = []
            for seg in data:
                if seg.get("is_music"):
                    logger.debug(
                        "Skipping music segment %.2f-%.2f",
                        seg.get("start", 0.0),
                        seg.get("end", 0.0),
                    )
                    continue
                filtered.append(seg)
            data = filtered
        subs = pysubs2.load_from_whisper(data)
        corrections = 0
        if rules:
            for ev in subs.events:
                original = ev.plaintext
                fixed = apply_corrections(original, rules)
                if fixed != original:
                    corrections += 1
                    logger.debug("Replaced text: %r -> %r", original, fixed)
                ev.text = fixed.replace("\n", "\\N")
        enforce_limits(
            subs,
            max_chars=args.max_chars,
            max_lines=args.max_lines,
            max_duration=args.max_duration,
            min_gap=args.min_gap,
        )
        if args.spellcheck:
            spellcheck_lines(subs)
        out_txt = out_srt.with_suffix(".txt") if args.transcript else None
        write_outputs(subs, out_srt, out_txt)

        total = len(subs.events)
        avg_lines = (
            sum(ev.text.count("\\N") + 1 for ev in subs.events) / total
            if total
            else 0.0
        )
        logger.info(
            "Processed %s: subtitles=%d avg_lines=%.2f corrections=%d",
            seg_path,
            total,
            avg_lines,
            corrections,
        )

    if args.batch_dir:
        batch_root = Path(args.batch_dir)
        out_root = Path(args.output)
        for seg_file in batch_root.rglob("segments.json"):
            rel = seg_file.relative_to(batch_root)
            dest_dir = out_root / rel.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            out_srt = dest_dir / (seg_file.stem + ".srt")
            process_file(seg_file, out_srt)
    else:
        if not args.segments:
            parser.error("--segments is required unless --batch-dir is used")
        process_file(Path(args.segments), Path(args.output))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
