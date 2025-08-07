"""Quality control utilities."""
from __future__ import annotations

import argparse
import logging
import re
import statistics
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable

import jiwer
import pysubs2

logger = logging.getLogger(__name__)


def _strip_markup(text: str) -> str:
    """Remove simple markup tags from ``text``."""
    return re.sub(r"<[^>]+>", "", text)


def _load_text(path: Path) -> str:
    """Load subtitle or plain text from ``path``.

    Parameters
    ----------
    path:
        Location of ``.srt`` or ``.txt`` file.
    """
    if path.suffix.lower() == ".srt":
        subs = pysubs2.load(str(path))
        return " ".join(event.plaintext for event in subs)
    if path.suffix.lower() == ".txt":
        return _strip_markup(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported file type: {path.suffix}")


def collect_metrics(srt_path: str) -> Dict[str, Any]:
    """Collect basic statistics from a subtitle file.

    Parameters
    ----------
    srt_path:
        Path to the ``.srt`` subtitle file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing subtitle count, average duration, average
        number of lines, and any warnings about unusual entries.
    """
    subs = pysubs2.load(srt_path)

    durations: list[float] = []
    line_counts: list[int] = []
    warnings: list[str] = []

    for i, event in enumerate(subs, start=1):
        duration = (event.end - event.start) / 1000.0  # convert ms to seconds
        durations.append(duration)

        line_counts.append(len(event.plaintext.splitlines()))

        if duration < 0.5:
            warnings.append(f"subtitle {i} very short ({duration:.2f}s)")
        if duration > 10.0:
            warnings.append(f"subtitle {i} very long ({duration:.2f}s)")

    metrics: Dict[str, Any] = {
        "subtitle_count": len(subs),
        "avg_duration": statistics.mean(durations) if durations else 0.0,
        "avg_lines": statistics.mean(line_counts) if line_counts else 0.0,
        "warnings": warnings,
    }

    logger.info("Subtitle metrics: %s", metrics)
    return metrics


def compute_wer(hyp_path: str, ref_path: str) -> float:
    """Compute word error rate between two files.

    Parameters
    ----------
    hyp_path, ref_path:
        Paths to the hypothesis and reference files.

    Returns
    -------
    float
        The word error rate between ``hyp_path`` and ``ref_path``.
    """
    hyp = _load_text(Path(hyp_path))
    ref = _load_text(Path(ref_path))
    score = jiwer.wer(ref, hyp)
    logger.info("WER(%s, %s) = %.3f", hyp_path, ref_path, score)
    return score


def validate_sync(srt_path: str, audio_path: str) -> Dict[str, Any]:
    """Validate subtitle timing against ``audio_path`` using forced alignment.

    Parameters
    ----------
    srt_path, audio_path:
        Paths to the subtitle ``.srt`` file and corresponding audio file.

    Returns
    -------
    Dict[str, Any]
        Metrics describing alignment offsets.
    """
    # Import aeneas lazily to avoid hard dependency when unused
    from aeneas.executetask import ExecuteTask
    from aeneas.task import Task

    subs = pysubs2.load(srt_path)
    words: list[str] = []
    expected: list[float] = []
    for event in subs:
        tokens = event.plaintext.strip().split()
        if not tokens:
            continue
        duration = event.end - event.start
        for i, token in enumerate(tokens):
            words.append(token)
            # Distribute words uniformly across the subtitle interval
            expected.append((event.start + (duration * i) / len(tokens)) / 1000.0)

    with NamedTemporaryFile("w", suffix=".txt", delete=False) as txt:
        txt.write("\n".join(words))
        text_path = txt.name

    config = "task_language=eng|is_text_type=plain|os_task_file_format=json"
    task = Task(config_string=config)
    task.audio_file_path_absolute = audio_path
    task.text_file_path_absolute = text_path
    ExecuteTask(task).execute()

    n = min(len(expected), len(task.sync_map_leaves))
    offsets = [abs(float(item.begin) - expected[i]) for i, item in zip(range(n), task.sync_map_leaves)]

    metrics: Dict[str, Any] = {
        "word_count": n,
        "mean_offset": statistics.mean(offsets) if offsets else 0.0,
        "median_offset": statistics.median(offsets) if offsets else 0.0,
        "max_offset": max(offsets) if offsets else 0.0,
    }
    logger.info("Sync metrics: %s", metrics)
    return metrics


def main(argv: Iterable[str] | None = None):
    """Command-line interface for quality control utilities."""
    parser = argparse.ArgumentParser(description="Quality control utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    wer_p = subparsers.add_parser("wer", help="Compute word error rate")
    wer_p.add_argument("hyp")
    wer_p.add_argument("ref")

    sync_p = subparsers.add_parser("sync", help="Validate subtitle sync")
    sync_p.add_argument("srt")
    sync_p.add_argument("audio")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    if args.command == "wer":
        return compute_wer(args.hyp, args.ref)
    if args.command == "sync":
        return validate_sync(args.srt, args.audio)
    raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    main()
