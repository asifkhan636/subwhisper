"""Subtitle handling pipeline utilities."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Optional

import pysubs2
import textwrap
from corrections import apply_corrections, load_corrections
from qc import collect_metrics


logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 45
DEFAULT_MAX_LINES = 2
DEFAULT_MAX_DURATION = 6.0
DEFAULT_MIN_GAP = 0.15


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


def _ns_len(text: str) -> int:
    """Length of ``text`` excluding all whitespace characters."""

    return len(re.sub(r"\s+", "", text))


def _cps(ev: pysubs2.SSAEvent) -> float:
    """Characters per second based on non-space characters."""

    duration = max(ev.end - ev.start, 1) / 1000
    return _ns_len(ev.plaintext) / duration


def _split_event_textually(ev: pysubs2.SSAEvent) -> tuple[pysubs2.SSAEvent, pysubs2.SSAEvent]:
    """Split an event in two, allocating time proportional to text length."""

    text = ev.plaintext
    if not text:
        mid = (ev.start + ev.end) // 2
        left = pysubs2.SSAEvent(start=ev.start, end=mid, text="")
        right = pysubs2.SSAEvent(start=mid, end=ev.end, text="")
        return left, right

    mid = len(text) // 2
    split = text.rfind(" ", 0, mid)
    if split == -1:
        m = re.search(r"\s", text[mid:])
        split = mid + m.start() if m else mid
    left_text = text[:split].strip()
    right_text = text[split:].strip()

    total_dur = ev.end - ev.start
    left_len = _ns_len(left_text)
    right_len = _ns_len(right_text)
    total_len = left_len + right_len or 1
    left_dur = int(round(total_dur * left_len / total_len))
    right_dur = total_dur - left_dur

    ev1 = pysubs2.SSAEvent(start=ev.start, end=ev.start + left_dur, text=left_text)
    ev2 = pysubs2.SSAEvent(start=ev.start + left_dur, end=ev.end, text=right_text)
    return ev1, ev2


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

    i = 0
    while i < len(subs.events):
        ev = subs.events[i]
        if (ev.end - ev.start) > max_ms or _cps(ev) > cps_limit:
            left, right = _split_event_textually(ev)
            subs.events[i] = left
            subs.events.insert(i + 1, right)
            continue
        i += 1

    for ev in subs.events:
        if ev.end - ev.start > max_ms:
            ev.end = ev.start + max_ms

    i = 1
    while i < len(subs.events):
        prev = subs.events[i - 1]
        curr = subs.events[i]
        required_start = prev.end + gap_ms
        if curr.start < required_start:
            shift = required_start - curr.start
            duration = curr.end - curr.start
            curr.start += shift
            curr.end = curr.start + duration
            if curr.start >= curr.end:
                left, right = _split_event_textually(curr)
                subs.events[i] = left
                subs.events.insert(i + 1, right)
                i += 1
                continue
        i += 1

    for ev in subs.events:
        lines = textwrap.wrap(ev.plaintext, width=max_chars)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
        ev.text = "\\N".join(lines)
        ev.event_cps = _cps(ev)

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
        default=DEFAULT_MAX_CHARS,
        help="Maximum characters per line",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=DEFAULT_MAX_LINES,
        help="Maximum number of lines per event",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=DEFAULT_MAX_DURATION,
        help="Maximum duration for an event in seconds",
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=DEFAULT_MIN_GAP,
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

        with NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
            subs.save(tmp.name, format_="srt")
            pre_metrics = collect_metrics(tmp.name)
        os.unlink(tmp.name)

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

        post_metrics = collect_metrics(str(out_srt))
        if (
            post_metrics["avg_cps"] > pre_metrics["avg_cps"]
            or post_metrics["pct_cps_gt_17"] > pre_metrics["pct_cps_gt_17"]
        ):
            raise ValueError("Formatting increased CPS metrics")
        metrics_out = out_srt.with_suffix(".metrics.json")
        metrics_out.write_text(
            json.dumps({"before": pre_metrics, "after": post_metrics}, indent=2),
            encoding="utf-8",
        )

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
