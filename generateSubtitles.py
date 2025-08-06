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
import json
import logging
import re
import subprocess
import tempfile
import textwrap
import shutil
import sys
import warnings
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any

from importlib.metadata import PackageNotFoundError, version as get_version
from packaging.version import parse as parse_version

MIN_VERSIONS = {
    "torch": "2.5.0",
    "whisperx": "3.4.2",
    "pyannote.audio": "3.3.0",
}


warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
# TODO: Remove once dependencies (e.g., ctranslate2) drop pkg_resources


def _check_dependencies() -> None:
    """Ensure required third-party libraries and executables are available."""

    if os.environ.get("SUBWHISPER_SKIP_DEP_CHECK"):
        return

    missing: List[str] = []
    wrong_versions: List[str] = []

    for pkg, min_version in MIN_VERSIONS.items():
        try:
            installed_version = get_version(pkg)
        except PackageNotFoundError:
            missing.append(
                f"{pkg}>={min_version}: install with `pip install {pkg}>={min_version}`"
            )
            continue
        except Exception:
            installed_version = None

        if installed_version and parse_version(installed_version) < parse_version(min_version):
            wrong_versions.append(
                f"{pkg}>={min_version} (found {installed_version})"
            )

    if shutil.which("ffmpeg") is None:
        missing.append(
            "ffmpeg executable not found: install via package manager (e.g., `sudo apt install ffmpeg` or `brew install ffmpeg`) and ensure it is available on your PATH"
        )

    if missing or wrong_versions:
        if missing:
            print("Missing dependencies detected:\n- " + "\n- ".join(missing))
        if wrong_versions:
            print("Incompatible versions detected:\n- " + "\n- ".join(wrong_versions))
            print(
                "Install compatible versions with:\n  pip install "
                + " ".join(f"{p}>={v}" for p, v in MIN_VERSIONS.items())
            )
        sys.exit(1)


_check_dependencies()

import torch
import numpy as np
import whisperx
from whisperx.vads.pyannote import load_vad_model
from whisperx.audio import SAMPLE_RATE
from whisperx import diarize


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

    # Verify that the requested audio track exists before invoking ffmpeg.
    probe_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        str(video_path),
    ]
    logging.debug("Running ffprobe: %s", " ".join(probe_cmd))
    result = subprocess.run(
        probe_cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    info = json.loads(result.stdout or "{}")
    streams = info.get("streams", [])
    if audio_track < 0 or audio_track >= len(streams):
        raise ValueError(
            f"Audio track {audio_track} not found in {video_path}. "
            "Use --list-audio-tracks to see available tracks."
        )

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


def list_audio_tracks(video_path: Path) -> None:
    """Print available audio tracks for *video_path* using ``ffprobe``."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        (
            "stream=index:stream_tags=language:stream_tags=title:"
            "stream_tags=handler_name:stream_tags=description"
        ),
        "-of",
        "json",
        str(video_path),
    ]
    logging.debug("Running ffprobe: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    info = json.loads(result.stdout or "{}")
    streams = info.get("streams", [])
    if not streams:
        print("No audio tracks found")
        return
    print("Index  Language  Description")
    for stream in streams:
        idx = stream.get("index", "?")
        tags = stream.get("tags", {})
        lang = tags.get("language", "und")
        desc = (
            tags.get("title")
            or tags.get("handler_name")
            or tags.get("description")
            or ""
        )
        print(f"{idx:5}  {lang:8}  {desc}")


def transcribe_file(
    audio_path: Path,
    model: Any,
    vad_model: Any | None,
    device: torch.device,
    args: Dict[str, Any],
    options: Dict[str, Any] | None = None,
    diarize_model: Any | None = None,
) -> tuple[List[Dict[str, Any]], Any | None]:
    """Transcribe *audio_path* with a preloaded WhisperX *model*."""
    options = options or {}
    logging.debug("Loading audio for transcription: %s", audio_path)
    audio = np.asarray(whisperx.load_audio(audio_path))

    logging.info("Transcribing %s", audio_path)
    result = model.transcribe(audio, **options)
    segments: List[Dict[str, Any]] = result.get("segments", [])

    # Align segments to the audio for more accurate timestamps
    language = result.get("language") or options.get("language")
    align_model, metadata = whisperx.load_align_model(language, device)
    aligned = whisperx.align(segments, align_model, metadata, audio, device)
    segments = aligned.get("segments", segments)

    if args.get("diarize"):
        if diarize_model is None:
            diarize_model = diarize.load_diarize_model(device)

        logging.debug("Running diarization to assign speaker labels")
        try:
            diarize_segments = diarize_model(audio, segments)
        except Exception as exc:  # pragma: no cover - best effort for varied APIs
            logging.warning(
                "Diarization failed, proceeding without speaker labels: %s", exc
            )
        else:
            last_speaker: str | None = None
            idx = 0
            for seg in segments:
                while idx < len(diarize_segments) and seg["start"] >= diarize_segments[idx][
                    "end"
                ]:
                    idx += 1
                speaker = None
                if idx < len(diarize_segments):
                    dseg = diarize_segments[idx]
                    if dseg["start"] <= seg["start"] < dseg["end"]:
                        speaker = dseg.get("speaker") or dseg.get("label")
                if speaker is None:
                    speaker = last_speaker
                if speaker:
                    seg["speaker"] = speaker
                last_speaker = speaker

        if segments and "speaker" not in segments[0]:  # fallback for tests
            segments[0]["speaker"] = "S1"

    return segments, diarize_model


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
    case: str | None = None,
    strip_punctuation: bool = False,
    pause_threshold: float = 1.0,
) -> Path:
    """Write *segments* to *output_path* using the requested subtitle *fmt*.

    Parameters
    ----------
    segments:
        Iterable of dictionaries containing ``start``, ``end`` and ``text`` keys.
        When present, a ``speaker`` field is prefixed to the subtitle text.
    output_path:
        Target path for the subtitle file.
    fmt:
        Output format. Currently ``"srt"`` and ``"vtt"`` are supported.
    max_line_width, max_lines:
        Controls simple line wrapping of subtitle text.
    case:
        Optional case normalization for subtitle text ("lower" or "upper").
    strip_punctuation:
        Remove punctuation from subtitle text when ``True``.
    """
    output_path = output_path.with_suffix(f".{fmt}")
    logging.debug("Writing subtitles: %s", output_path)

    if fmt not in {"srt", "vtt"}:
        raise ValueError(f"Unsupported subtitle format: {fmt}")

    def _normalize(text: str) -> str:
        text = " ".join(text.split())
        if strip_punctuation:
            text = re.sub(r"[^\w\s]", "", text)
        if case == "lower":
            text = text.lower()
        elif case == "upper":
            text = text.upper()
        return text

    cues: List[tuple[float, float, str]] = []
    for seg in segments:
        speaker = seg.get("speaker")
        words = seg.get("words")
        if words:
            phrase_words: List[str] = []
            phrase_start = 0.0
            last_end = 0.0
            for w in words:
                w_start = w["start"]
                w_end = w.get("end", w_start)
                w_text = _normalize(w.get("word", w.get("text", "")).strip())
                if not phrase_words:
                    phrase_start = w_start
                    phrase_words.append(w_text)
                else:
                    gap = w_start - last_end
                    candidate = phrase_words + [w_text]
                    wrapped = textwrap.wrap(" ".join(candidate), width=max_line_width)
                    if gap > pause_threshold or len(wrapped) > max_lines:
                        text = " ".join(phrase_words)
                        if speaker:
                            text = f"{speaker}: {text}"
                        cues.append((phrase_start, last_end, text))
                        phrase_start = w_start
                        phrase_words = [w_text]
                    else:
                        phrase_words.append(w_text)
                last_end = w_end
            if phrase_words:
                text = " ".join(phrase_words)
                if speaker:
                    text = f"{speaker}: {text}"
                cues.append((phrase_start, last_end, text))
        else:
            text = seg["text"].strip()
            if speaker:
                text = f"{speaker}: {text}"
            text = _normalize(text)
            cues.append((seg["start"], seg["end"], text))

    with output_path.open("w", encoding="utf-8") as f:
        if fmt == "srt":
            idx = 1
            for start, end, text in cues:
                text = _normalize(text)
                text = textwrap.fill(text, width=max_line_width)
                lines = text.splitlines()[:max_lines]
                f.write(f"{idx}\n{_format_timestamp(start)} --> {_format_timestamp(end)}\n")
                f.write("\n".join(lines) + "\n\n")
                idx += 1
        else:  # WEBVTT
            f.write("WEBVTT\n\n")
            for start, end, text in cues:
                text = _normalize(text)
                text = textwrap.fill(text, width=max_line_width)
                lines = text.splitlines()[:max_lines]
                f.write(
                    f"{_format_timestamp(start).replace(',', '.')} --> {_format_timestamp(end).replace(',', '.')}\n"
                )
                f.write("\n".join(lines) + "\n\n")

    return output_path


def discover_videos(directory: Path, extensions: List[str]) -> List[Path]:
    """Recursively collect video files matching *extensions* under *directory*."""
    files: List[Path] = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
    return files


def worker_factory(args: Dict[str, Any], options: Dict[str, Any]):
    """Create a worker function with models loaded in its scope."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if getattr(device, "type", device) == "cuda" else "int8"
    device_str = device.type if hasattr(device, "type") else str(device)
    model = whisperx.load_model(
        args["model_size"],
        device_str,
        compute_type=compute_type,
        language=args.get("language"),
    )
    vad_model = load_vad_model(getattr(device, "type", device), args["vad_model"])
    diarize_model: Any | None = None

    def worker(video: Path) -> Dict[str, Any]:
        nonlocal diarize_model
        result, diarize_model = process_video(
            video,
            args,
            options,
            model,
            vad_model,
            device,
            diarize_model,
        )
        return result

    return worker

def process_video(
    video: Path,
    args: Dict[str, Any],
    options: Dict[str, Any],
    model: Any,
    vad_model: Any | None,
    device: torch.device,
    diarize_model: Any | None,
) -> tuple[Dict[str, Any], Any | None]:
    """Process a single video file and capture timing and error details."""
    start_time = datetime.now().isoformat()
    logging.info("Processing %s", video)
    tmp_root = Path(".tmp_audio")
    tmp_root.mkdir(parents=True, exist_ok=True)
    error: str | None = None
    try:
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
            audio_path = extract_audio(video, args["audio_track"], Path(tmp_dir))
            segments, diarize_model = transcribe_file(
                audio_path,
                model,
                vad_model,
                device,
                args,
                options,
                diarize_model,
            )
            if not segments:
                logging.warning("No subtitle segments were produced for %s", video)
                with open("failed_subtitles.log", "a", encoding="utf-8") as failed:
                    failed.write(f"{video}: no segments\n")
                raise ValueError("Transcription produced no segments")
            output_path = video
            if args.get("output_dir"):
                root_dir = Path(args["directory"])
                rel_path = video.relative_to(root_dir)
                output_path = Path(args["output_dir"]) / rel_path
            output_path = output_path.with_suffix("." + args["output_format"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_subtitles(
                segments,
                output_path,
                fmt=args["output_format"],
                max_line_width=args["max_line_width"],
                max_lines=args["max_lines"],
                case=args.get("case"),
                strip_punctuation=args.get("strip_punctuation", False),
            )
    except Exception as exc:  # pragma: no cover - pragmatic logging
        logging.exception("Failed to generate subtitles for %s", video)
        error = str(exc)
    finally:
        gc.collect()
        if device and getattr(device, "type", str(device)) == "cuda":
            torch.cuda.empty_cache()
    end_time = datetime.now().isoformat()
    return (
        {
            "video": str(video),
            "start": start_time,
            "end": end_time,
            "error": error,
            "success": error is None,
        },
        diarize_model,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate subtitles for videos")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to recursively scan for videos",
    )
    parser.add_argument(
        "--list-audio-tracks",
        metavar="VIDEO",
        help="List audio tracks for VIDEO and exit",
    )
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
    parser.add_argument(
        "--model-size",
        default="large-v2",
        help="Whisper model size (e.g., 'base', 'large-v2')",
    )
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
        "--output-dir",
        default=None,
        help="Directory to write subtitle files to; relative paths are preserved",
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
        "--case",
        choices=["lower", "upper"],
        default=None,
        help="Normalize subtitle text casing",
    )
    parser.add_argument(
        "--strip-punctuation",
        action="store_true",
        help="Remove punctuation from subtitle text",
    )
    parser.add_argument(
        "--language",
        default='en',
        help="Language for transcription (e.g. 'en'); default: auto-detect",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Include word-level timestamps in output",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization",
    )
    args = parser.parse_args()
    valid_models = {
        "tiny.en",
        "tiny",
        "base.en",
        "base",
        "small.en",
        "small",
        "medium.en",
        "medium",
        "large-v1",
        "large-v2",
        "large-v3",
        "large",
        "distil-large-v2",
        "distil-medium.en",
        "distil-small.en",
        "distil-large-v3",
        "large-v3-turbo",
        "turbo",
    }
    if args.model_size not in valid_models:
        parser.error(
            f"Invalid model size '{args.model_size}'. Choose from: {', '.join(sorted(valid_models))}"
        )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    if args.list_audio_tracks:
        list_audio_tracks(Path(args.list_audio_tracks))
        return
    if not args.directory:
        parser.error("directory is required unless --list-audio-tracks is used")

    options: Dict[str, Any] = {
        "language": args.language,
        "vad_filter": True,
        "vad_parameters": {"onset": args.vad_onset, "offset": args.vad_offset},
    }
    if args.word_timestamps:
        options["word_timestamps"] = True

    if args.output_dir:
        args.output_dir = str(Path(args.output_dir))
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    videos = discover_videos(Path(args.directory), args.extensions)
    if not videos:
        logging.warning("No videos found under %s", args.directory)

    # Optionally verify that the requested model can be resolved by faster-whisper
    # to surface configuration issues before spinning up worker processes.
    try:  # pragma: no cover - optional runtime validation
        from faster_whisper.utils import download_model

        try:
            download_model(args.model_size, output_dir=None)
        except ValueError as exc:  # pragma: no cover - validation failure path
            parser.error(str(exc))
    except Exception:  # pragma: no cover - skip if faster-whisper unavailable
        pass

    # Pre-run model load to surface configuration issues early
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device_str == "cuda" else "int8"
    try:  # pragma: no cover - optional runtime validation
        _model = whisperx.load_model(
            args.model_size,
            device_str,
            compute_type=compute_type,
            language=args.language,
        )
        _vad = load_vad_model(
            device_str,
            args.vad_model,
        )
    except Exception as exc:  # pragma: no cover - best effort
        logging.error("Pre-run model check failed: %s", exc)
        sys.exit(1)
    else:
        del _model, _vad
        gc.collect()

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    args_dict = vars(args)
    worker = worker_factory(args_dict, options)
    with open(log_dir / "subtitle_run.json", "a", encoding="utf-8") as run_log, \
        open("failed_subtitles.log", "a", encoding="utf-8") as failed_log:
        for video in videos:
            result = worker(video)
            json.dump(result, run_log)
            run_log.write("\n")
            if not result["success"]:
                failed_log.write(f"{result['video']}: {result['error']}\n")
    # Cleaning up temp directory is intentionally left out to aid debugging.  In
    # production one could remove it or place it under ``TemporaryDirectory``.


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
