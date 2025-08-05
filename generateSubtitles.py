#!/usr/bin/env python3
"""Utility to extract audio from videos and generate subtitles using WhisperX.

The script demonstrates a modular design with small, testable functions and
extensive logging.  It aims to be a starting point for more sophisticated
subtitle workflows.  Ideas for future enhancements are sprinkled throughout
as comments to encourage experimentation (e.g. diarization, word-level
timestamps, parallelism, or explicit language overrides).
"""
from __future__ import annotations

import argparse
import gc
import logging
import subprocess
import textwrap
from pathlib import Path
from typing import Iterable, List, Dict, Any

import torch
import whisperx
from whisperx.vads.pyannote import load_vad_model


def extract_audio(video_path: Path, audio_track: int, tmp_dir: Path) -> Path:
    """Extract an audio track from *video_path* using ``ffmpeg``.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    audio_track:
        Zero-based index of the audio track to extract.
    tmp_dir:
        Directory where the temporary WAV file will be placed.

    Returns
    -------
    Path
        Path to the generated temporary WAV file (16 kHz mono).
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    audio_path = tmp_dir / (video_path.stem + ".wav")
    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-map",
        f"0:a:{audio_track}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-y",
        str(audio_path),
    ]
    logging.debug("Running ffmpeg: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_path


def transcribe_file(
    audio_path: Path,
    model: Any,
    vad_model: Any | None,
    device: torch.device,
    options: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Transcribe *audio_path* with a preloaded WhisperX *model*.

    ``load_vad_model`` from ``whisperx.vads.pyannote`` can be used to detect
    speech regions prior to transcription with voice-activity-detection (VAD).
    """
    options = options or {}
    logging.debug("Loading audio for transcription: %s", audio_path)
    audio = whisperx.load_audio(audio_path)

    speech_segments = None
    if vad_model is not None:
        logging.debug("Running VAD to obtain speech segments")
        try:
            speech_segments = vad_model(audio)
        except Exception as exc:  # pragma: no cover - best effort for varied APIs
            logging.warning("VAD model failed, proceeding without VAD: %s", exc)

    logging.info("Transcribing %s", audio_path)
    if speech_segments is not None:
        try:
            result = model.transcribe(audio, segments=speech_segments, **options)
        except TypeError:
            # some versions of WhisperX expect ``speech_chunks`` instead
            result = model.transcribe(audio, speech_chunks=speech_segments, **options)
    else:
        result = model.transcribe(audio, **options)

    segments = result.get("segments", [])
    return segments


def _format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_subtitles(
    segments: Iterable[Dict[str, Any]],
    output_path: Path,
    fmt: str = "srt",
    max_line_width: int = 42,
    max_lines: int = 2,
) -> Path:
    """Write *segments* to *output_path* using the requested subtitle *fmt*.

    Parameters
    ----------
    segments:
        Iterable of dictionaries containing ``start``, ``end`` and ``text`` keys.
    output_path:
        Target path for the subtitle file.
    fmt:
        Output format. Currently ``"srt"`` and ``"vtt"`` are supported.
    max_line_width, max_lines:
        Controls simple line wrapping of subtitle text.
    """
    output_path = output_path.with_suffix(f".{fmt}")
    logging.debug("Writing subtitles: %s", output_path)

    if fmt not in {"srt", "vtt"}:
        raise ValueError(f"Unsupported subtitle format: {fmt}")

    with output_path.open("w", encoding="utf-8") as f:
        if fmt == "srt":
            for idx, seg in enumerate(segments, start=1):
                start = _format_timestamp(seg["start"])
                end = _format_timestamp(seg["end"])
                text = textwrap.fill(seg["text"].strip(), width=max_line_width)
                lines = text.splitlines()[:max_lines]
                f.write(f"{idx}\n{start} --> {end}\n")
                f.write("\n".join(lines) + "\n\n")
        else:  # WEBVTT
            f.write("WEBVTT\n\n")
            for seg in segments:
                start = _format_timestamp(seg["start"]).replace(",", ".")
                end = _format_timestamp(seg["end"]).replace(",", ".")
                text = textwrap.fill(seg["text"].strip(), width=max_line_width)
                lines = text.splitlines()[:max_lines]
                f.write(f"{start} --> {end}\n")
                f.write("\n".join(lines) + "\n\n")

    return output_path


def discover_videos(directory: Path, extensions: List[str]) -> List[Path]:
    """Recursively collect video files matching *extensions* under *directory*."""
    files: List[Path] = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate subtitles for videos")
    parser.add_argument("directory", help="Directory to recursively scan for videos")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".mp4", ".mkv", ".mov", ".avi"],
        help="Video file extensions to include",
    )
    parser.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Audio track index to extract (default: 0)",
    )
    parser.add_argument("--model-size", default="medium", help="Whisper model size")
    parser.add_argument(
        "--vad-model",
        default="pyannote/segmentation",
        help="Pyannote VAD model to use",
    )
    parser.add_argument(
        "--vad-onset",
        type=float,
        default=0.5,
        help="Onset probability threshold for VAD",
    )
    parser.add_argument(
        "--vad-offset",
        type=float,
        default=0.363,
        help="Offset probability threshold for VAD",
    )
    parser.add_argument(
        "--output-format",
        default="srt",
        choices=["srt", "vtt"],
        help="Subtitle output format",
    )
    parser.add_argument(
        "--max-line-width",
        type=int,
        default=42,
        help="Maximum characters per subtitle line",
    )
    parser.add_argument(
        "--max-lines", type=int, default=2, help="Maximum lines per subtitle"
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Override language detection (e.g. 'en')",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    failed_log = open("failed_subtitles.log", "a", encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device.type == "cuda" else "int8"
    logging.info("Using device: %s", device)

    model = whisperx.load_model(args.model_size, device, compute_type=compute_type)
    # ``load_vad_model`` initializes the pyannote VAD pipeline with optional thresholds.
    vad_model = load_vad_model(
        args.vad_model,
        vad_options={"vad_onset": args.vad_onset, "vad_offset": args.vad_offset},
        device=device,
    )

    options: Dict[str, Any] = {}
    if args.language:
        options["language"] = args.language

    videos = discover_videos(Path(args.directory), args.extensions)
    if not videos:
        logging.warning("No videos found under %s", args.directory)

    tmp_dir = Path(".tmp_audio")
    for video in videos:
        logging.info("Processing %s", video)
        try:
            audio_path = extract_audio(video, args.audio_track, tmp_dir)
            segments = transcribe_file(audio_path, model, vad_model, device, options)
            write_subtitles(
                segments,
                video.with_suffix("." + args.output_format),
                fmt=args.output_format,
                max_line_width=args.max_line_width,
                max_lines=args.max_lines,
            )
        except Exception as exc:  # pragma: no cover - pragmatic logging
            logging.exception("Failed to generate subtitles for %s", video)
            failed_log.write(f"{video}: {exc}\n")
        finally:
            # Housekeeping: remove temp audio and free up GPU memory.
            if 'audio_path' in locals() and audio_path.exists():
                audio_path.unlink(missing_ok=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    failed_log.close()
    # Cleaning up temp directory is intentionally left out to aid debugging.  In
    # production one could remove it or place it under ``TemporaryDirectory``.


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
