#!/usr/bin/env python3
"""Extract audio and run multiple WhisperX CLI commands.

This script uses ffmpeg to grab the second audio track (index 1) from a video,
converts it to a mono 16 kHz WAV file for optimal WhisperX accuracy, and then
runs a series of WhisperX command lines on the audio to generate SRT subtitles.

Usage:
    python batch_whisperx.py input.mkv --audio audio.wav --outdir subtitles

Pass a text file with `--commands-file` where each line is a WhisperX command
containing `{audio}` and `{outdir}` placeholders to customise behaviour.
"""

import argparse
import logging
import os
import shlex
import subprocess
import sys
from typing import Iterable


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd: Iterable[str], description: str) -> None:
    """Execute a command list and raise informative errors on failure."""
    logger.info("[%s]", description)
    logger.info("Command: %s", " ".join(cmd))
    try:
        completed = subprocess.run(
            cmd,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract second audio track and run multiple WhisperX commands."
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
        help="Directory for WhisperX outputs (default: whisperx_output)",
    )
    parser.add_argument(
        "--commands-file",
        "-c",
        help="Optional file with WhisperX commands. Lines may use {audio} and {outdir}.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Extract second audio track with ffmpeg (mono, 16 kHz PCM)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Build WhisperX command templates
    # ------------------------------------------------------------------
    if args.commands_file:
        with open(args.commands_file, "r", encoding="utf-8") as fh:
            templates = [
                line.strip()
                for line in fh
                if line.strip() and not line.strip().startswith("#")
            ]
    else:
        templates = [
            # 1. Highest accuracy on GPU (large model)
            (
                "whisperx {audio} "
                "--model large-v2 "
                "--align_model WAV2VEC2_ASR_LARGE_LV60K_960H "
                "--language en "
                "--batch_size 4 "
                "--compute_type float16 "
                "--beam_size 5 --best_of 5 "
                "--max_line_width 45 --max_line_count 2 "
                "--output_format srt "
                "--output_dir {outdir}/large "
                "--task transcribe"
            ),
            # 2. GPU medium English model
            (
                "whisperx {audio} "
                "--model medium.en "
                "--align_model WAV2VEC2_ASR_LARGE_LV60K_960H "
                "--language en "
                "--batch_size 4 "
                "--compute_type float16 "
                "--beam_size 5 "
                "--output_format srt "
                "--output_dir {outdir}/medium "
                "--task transcribe"
            ),
            # 3. CPU high accuracy (float32)
            (
                "whisperx {audio} "
                "--model medium.en "
                "--device cpu "
                "--compute_type float32 "
                "--language en "
                "--batch_size 1 "
                "--align_model WAV2VEC2_ASR_LARGE_LV60K_960H "
                "--max_line_width 45 --max_line_count 2 "
                "--output_format srt "
                "--output_dir {outdir}/cpu_float32 "
                "--task transcribe"
            ),
            # 4. CPU memory-efficient (int8)
            (
                "whisperx {audio} "
                "--model small.en "
                "--device cpu "
                "--compute_type int8 "
                "--language en "
                "--batch_size 1 "
                "--align_model WAV2VEC2_ASR_LARGE_LV60K_960H "
                "--output_format srt "
                "--output_dir {outdir}/cpu_int8 "
                "--task transcribe"
            ),
        ]

    # ------------------------------------------------------------------
    # 3. Run WhisperX commands
    # ------------------------------------------------------------------
    for idx, tmpl in enumerate(templates, 1):
        cmd = shlex.split(tmpl.format(audio=args.audio, outdir=args.outdir))
        try:
            run_command(cmd, f"WhisperX command {idx}/{len(templates)}")
        except RuntimeError as exc:
            logger.error("%s", exc)
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
