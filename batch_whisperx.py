#!/usr/bin/env python3
"""Extract audio and run Faster-Whisper-compatible subtitle jobs.

This script preserves the historical ``batch_whisperx.py`` CLI, but it no
longer requires the external ``whisperx`` executable. Legacy command-file
entries that begin with ``whisperx`` are translated into calls to
``transcribe.py`` and ``subtitle_pipeline.py``.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Iterable


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SUPPORTED_FLAGS = {
    "--model",
    "--device",
    "--compute_type",
    "--batch_size",
    "--beam_size",
    "--output_dir",
    "--output_format",
    "--max_line_width",
    "--max_line_count",
    "--language",
    "--task",
}
SCRIPT_DIR = Path(__file__).resolve().parent


def run_command(cmd: Iterable[str], description: str) -> None:
    """Execute a command list and raise informative errors on failure."""

    logger.info("[%s]", description)
    logger.info("Command: %s", " ".join(cmd))
    try:
        completed = subprocess.run(
            list(cmd),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if completed.stdout:
            logger.info(completed.stdout)
        if completed.stderr:
            logger.info(completed.stderr)
        logger.info("%s completed successfully.", description)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        if exc.stdout:
            logger.error(exc.stdout)
        if exc.stderr:
            logger.error(exc.stderr)
        raise RuntimeError(
            f"{description} failed with exit code {exc.returncode}"
        ) from exc


def _build_internal_job(
    audio: str,
    outdir: str,
    model: str,
    device: str | None,
    language: str | None,
    compute_type: str | None,
    batch_size: str | None,
    beam_size: str | None,
    max_line_width: str | None,
    max_line_count: str | None,
) -> list[list[str]]:
    """Return the internal command sequence for one subtitle job."""

    outdir_path = Path(outdir)
    seg_path = outdir_path / "segments.json"
    srt_path = outdir_path / (Path(audio).stem + ".srt")

    transcribe_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "transcribe.py"),
        audio,
        "--outdir",
        outdir,
        "--model",
        model,
    ]
    if device:
        transcribe_cmd.extend(["--device", device])
    if language and language.lower() != "auto":
        transcribe_cmd.extend(["--language", language])
    if compute_type:
        transcribe_cmd.extend(["--compute-type", compute_type])
    if batch_size:
        transcribe_cmd.extend(["--batch-size", batch_size])
    if beam_size:
        transcribe_cmd.extend(["--beam-size", beam_size])

    subtitle_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "subtitle_pipeline.py"),
        "--segments",
        str(seg_path),
        "--output",
        str(srt_path),
    ]
    if max_line_width:
        subtitle_cmd.extend(["--max-chars", max_line_width])
    if max_line_count:
        subtitle_cmd.extend(["--max-lines", max_line_count])

    return [transcribe_cmd, subtitle_cmd]


def _translate_whisperx_command(command: str) -> list[list[str]]:
    """Translate a legacy whisperx CLI command into internal commands."""

    tokens = shlex.split(command, posix=(os.name != "nt"))
    if not tokens or tokens[0] != "whisperx":
        return [tokens]

    args = {
        "audio": None,
        "model": "large-v3-turbo",
        "device": None,
        "compute_type": None,
        "batch_size": None,
        "beam_size": None,
        "output_dir": None,
        "output_format": "srt",
        "max_line_width": None,
        "max_line_count": None,
        "language": "en",
        "task": "transcribe",
    }

    idx = 1
    while idx < len(tokens):
        token = tokens[idx]
        if not token.startswith("--"):
            if args["audio"] is None:
                args["audio"] = token
                idx += 1
                continue
            raise ValueError(f"Unexpected positional argument '{token}'.")

        if token not in SUPPORTED_FLAGS:
            raise ValueError(
                f"Unsupported WhisperX-only flag '{token}'. "
                "Supported flags: " + ", ".join(sorted(SUPPORTED_FLAGS))
            )
        if idx + 1 >= len(tokens):
            raise ValueError(f"Flag '{token}' requires a value.")

        value = tokens[idx + 1]
        key = token[2:]
        args[key] = value
        idx += 2

    if args["audio"] is None:
        raise ValueError("Legacy whisperx command must include an audio file.")
    if args["output_dir"] is None:
        raise ValueError("Legacy whisperx command must include --output_dir.")
    if args["output_format"] != "srt":
        raise ValueError("Only '--output_format srt' is supported.")
    if args["task"] != "transcribe":
        raise ValueError("Only '--task transcribe' is supported.")

    return _build_internal_job(
        audio=args["audio"],
        outdir=os.path.normpath(args["output_dir"]),
        model=args["model"],
        device=args["device"],
        language=args["language"],
        compute_type=args["compute_type"],
        batch_size=args["batch_size"],
        beam_size=args["beam_size"],
        max_line_width=args["max_line_width"],
        max_line_count=args["max_line_count"],
    )


def _default_jobs(audio: str, outdir: str) -> list[list[list[str]]]:
    """Return the built-in compatibility jobs."""

    return [
        _build_internal_job(
            audio=audio,
            outdir=os.path.join(outdir, "large"),
            model="large-v3-turbo",
            device="cuda",
            language=None,
            compute_type="float16",
            batch_size="4",
            beam_size="5",
            max_line_width="45",
            max_line_count="2",
        ),
        _build_internal_job(
            audio=audio,
            outdir=os.path.join(outdir, "medium"),
            model="medium",
            device="cuda",
            language=None,
            compute_type="float16",
            batch_size="4",
            beam_size="5",
            max_line_width=None,
            max_line_count=None,
        ),
        _build_internal_job(
            audio=audio,
            outdir=os.path.join(outdir, "cpu_float32"),
            model="medium",
            device="cpu",
            language=None,
            compute_type="float32",
            batch_size="1",
            beam_size=None,
            max_line_width="45",
            max_line_count="2",
        ),
        _build_internal_job(
            audio=audio,
            outdir=os.path.join(outdir, "cpu_int8"),
            model="small",
            device="cpu",
            language=None,
            compute_type="int8",
            batch_size="1",
            beam_size=None,
            max_line_width=None,
            max_line_count=None,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract second audio track and run Faster-Whisper compatibility jobs."
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--audio",
        "-a",
        default="extracted.wav",
        help="Where to write the extracted WAV (default: extracted.wav)",
    )
    parser.add_argument(
        "--outdir",
        "-o",
        default="whisperx_output",
        help="Directory for outputs (default: whisperx_output)",
    )
    parser.add_argument(
        "--commands-file",
        "-c",
        help="Optional file with legacy whisperx commands. Lines may use {audio} and {outdir}.",
    )
    args = parser.parse_args()

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        args.video,
        "-map",
        "0:a:1",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        args.audio,
    ]
    try:
        run_command(ffmpeg_cmd, "ffmpeg extraction")
    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    if args.commands_file:
        with open(args.commands_file, "r", encoding="utf-8") as fh:
            jobs = [
                _translate_whisperx_command(
                    line.strip().format(audio=args.audio, outdir=args.outdir)
                )
                for line in fh
                if line.strip() and not line.strip().startswith("#")
            ]
    else:
        jobs = _default_jobs(args.audio, args.outdir)

    total = len(jobs)
    for idx, steps in enumerate(jobs, 1):
        try:
            for step_no, cmd in enumerate(steps, 1):
                run_command(cmd, f"Compatibility job {idx}/{total} step {step_no}/{len(steps)}")
        except (RuntimeError, ValueError) as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
