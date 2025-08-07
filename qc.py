"""Quality control utilities."""
from __future__ import annotations

import argparse
import logging
import re
import statistics
import json
import csv
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Optional

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


def _gather_srt_files(paths: List[str], recursive: bool) -> List[Path]:
    """Expand ``paths`` into a list of subtitle files."""
    results: List[Path] = []
    for p in map(Path, paths):
        if p.is_file():
            results.append(p)
        elif p.is_dir():
            pattern = "**/*.srt" if recursive else "*.srt"
            results.extend(sorted(p.glob(pattern)))
    return results


def _match_file(base: Optional[str], target: Path, extensions: List[str]) -> Optional[Path]:
    """Return a file in ``base`` matching ``target``'s stem."""
    if base is None:
        return None
    base_path = Path(base)
    if base_path.is_file():
        return base_path
    stem = target.stem
    if base_path.is_dir():
        for ext in extensions:
            candidate = base_path / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    return None


def _aggregate_numeric(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute mean of numeric fields across ``rows``."""
    agg: Dict[str, float] = {}
    numeric_keys: set[str] = set()
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (int, float)):
                numeric_keys.add(k)
    for key in numeric_keys:
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if values:
            agg[key] = statistics.mean(values)
    return agg


def _write_csv(path: str, rows: List[Dict[str, Any]], aggregate: Dict[str, float]) -> None:
    """Write per-episode metrics and aggregate summary to CSV."""
    fieldnames = sorted({k for row in rows for k in row if k != "file"})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file"] + fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
            writer.writerow(out)
        agg_row = {"file": "MEAN"}
        for k, v in aggregate.items():
            agg_row[k] = v
        writer.writerow(agg_row)


def main(argv: Iterable[str] | None = None) -> Dict[str, Any]:
    """Command-line interface for quality control utilities.

    Parameters
    ----------
    argv:
        Optional command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Quality control utilities")
    parser.add_argument("generated", nargs="+", help="Generated subtitle file(s) or directory")
    parser.add_argument("--reference", "-r", help="Reference transcript file or directory")
    parser.add_argument("--audio", "-a", help="Audio file or directory")
    parser.add_argument("--wer", action=argparse.BooleanOptionalAction, default=True, help="Enable WER computation")
    parser.add_argument("--sync", action=argparse.BooleanOptionalAction, default=True, help="Enable sync validation")
    parser.add_argument("--metrics", action=argparse.BooleanOptionalAction, default=True, help="Collect subtitle metrics")
    parser.add_argument("--json", dest="json_path", help="Write JSON summary to PATH")
    parser.add_argument("--csv", dest="csv_path", help="Write CSV summary to PATH")
    parser.add_argument("--recursive", action="store_true", help="Recurse into directories")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    srt_files = _gather_srt_files(args.generated, args.recursive)
    episodes: List[Dict[str, Any]] = []
    for srt in srt_files:
        episode: Dict[str, Any] = {"file": str(srt)}
        ref_path = _match_file(args.reference, srt, [".srt", ".txt"])
        audio_path = _match_file(args.audio, srt, [".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"])

        if args.metrics:
            episode.update(collect_metrics(str(srt)))
        if args.wer and ref_path:
            episode["wer"] = compute_wer(str(srt), str(ref_path))
        if args.sync and audio_path:
            sync_metrics = validate_sync(str(srt), str(audio_path))
            episode.update({f"sync_{k}": v for k, v in sync_metrics.items()})

        logger.info("Metrics for %s: %s", srt, episode)
        episodes.append(episode)

    aggregate = _aggregate_numeric(episodes)

    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump({"episodes": episodes, "aggregate": aggregate}, fh, indent=2)
    if args.csv_path:
        Path(args.csv_path).parent.mkdir(parents=True, exist_ok=True)
        _write_csv(args.csv_path, episodes, aggregate)

    return {"episodes": episodes, "aggregate": aggregate}


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    main()
