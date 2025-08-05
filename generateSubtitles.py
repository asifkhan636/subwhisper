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
import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
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
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any


warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
# TODO: Remove once dependencies (e.g., ctranslate2) drop pkg_resources

def _check_dependencies() -> None:
    """Ensure required third-party libraries and executables are available.

    The script relies on ``torch`` for tensor operations, ``whisperx`` for
    transcription, ``pyannote.audio`` for VAD/diarization features, and the
    ``ffmpeg`` executable for media manipulation.  If any are missing, provide
    helpful installation instructions and exit gracefully.
    """

    missing: List[str] = []

    try:  # PyTorch
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - import failure path
        missing.append(
            "torch: install with `pip install torch` (see https://pytorch.org for "
            "platform-specific instructions)"
        )

    try:  # WhisperX
        import whisperx  # noqa: F401
    except Exception:  # pragma: no cover - import failure path
        missing.append(
            "whisperx: install with `pip install git+https://github.com/m-bain/whisperX`"
        )

    try:  # pyannote.audio
        import pyannote.audio  # noqa: F401
    except Exception:  # pragma: no cover - import failure path
        missing.append(
            "pyannote.audio: install with `pip install pyannote.audio`"
        )

    if shutil.which("ffmpeg") is None:
        missing.append(
            "ffmpeg executable not found: install via package manager (e.g., "
            "`sudo apt install ffmpeg` or `brew install ffmpeg`) and ensure it is "
            "available on your PATH"
        )

    if missing:
        print("Missing dependencies detected:\n- " + "\n- ".join(missing))
        sys.exit(1)


_check_dependencies()

import torch
import whisperx
from whisperx.vads.pyannote import load_vad_model
from whisperx.audio import SAMPLE_RATE


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
    options: Dict[str, Any] | None = None,
    diarize_model: Any | None = None,
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
            if not speech_segments:
                logging.warning("VAD produced no segments; falling back to chunking")
                speech_segments = None
        except Exception as exc:  # pragma: no cover - best effort for varied APIs
            logging.warning("VAD model failed, proceeding without VAD: %s", exc)

    logging.info("Transcribing %s", audio_path)
    segments: List[Dict[str, Any]] = []
    if speech_segments is not None:
        try:
            result = model.transcribe(audio, segments=speech_segments, **options)
        except TypeError:
            # some versions of WhisperX expect ``speech_chunks`` instead
            result = model.transcribe(audio, speech_chunks=speech_segments, **options)
        segments = result.get("segments", [])
    else:
        chunk_dur = float(options.get("chunk_duration", 30.0))
        sr = int(options.get("sample_rate", SAMPLE_RATE))
        chunk_size = int(chunk_dur * sr)
        for start in range(0, len(audio), chunk_size):
            chunk_audio = audio[start:start + chunk_size]
            result = model.transcribe(chunk_audio, **options)
            chunk_segments = result.get("segments", [])
            offset = start / sr
            for seg in chunk_segments:
                seg["start"] += offset
                seg["end"] += offset
            segments.extend(chunk_segments)
    if diarize_model is not None and speech_segments is not None:
        logging.debug("Running diarization to assign speaker labels")
        try:
            diarize_segments = diarize_model(audio, speech_segments)
        except Exception as exc:  # pragma: no cover - best effort for varied APIs
            logging.warning("Diarization failed, proceeding without speaker labels: %s", exc)
        else:
            for seg in segments:
                for dseg in diarize_segments:
                    if dseg["start"] <= seg["start"] < dseg["end"]:
                        speaker = dseg.get("speaker") or dseg.get("label")
                        if speaker:
                            seg["speaker"] = speaker
                        break

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
    case: str | None = None,
    strip_punctuation: bool = False,
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

    with output_path.open("w", encoding="utf-8") as f:
        if fmt == "srt":
            idx = 1
            for seg in segments:
                words = seg.get("words")
                if words:
                    for w in words:
                        start = _format_timestamp(w["start"])
                        end = _format_timestamp(w.get("end", w["start"]))
                        text = w.get("word", w.get("text", "")).strip()
                        if "speaker" in seg:
                            text = f"{seg['speaker']}: {text}"
                        text = _normalize(text)
                        text = textwrap.fill(text, width=max_line_width)
                        lines = text.splitlines()[:max_lines]
                        f.write(f"{idx}\n{start} --> {end}\n")
                        f.write("\n".join(lines) + "\n\n")
                        idx += 1
                else:
                    start = _format_timestamp(seg["start"])
                    end = _format_timestamp(seg["end"])
                    text = seg["text"].strip()
                    if "speaker" in seg:
                        text = f"{seg['speaker']}: {text}"
                    text = _normalize(text)
                    text = textwrap.fill(text, width=max_line_width)
                    lines = text.splitlines()[:max_lines]
                    f.write(f"{idx}\n{start} --> {end}\n")
                    f.write("\n".join(lines) + "\n\n")
                    idx += 1
        else:  # WEBVTT
            f.write("WEBVTT\n\n")
            for seg in segments:
                words = seg.get("words")
                if words:
                    for w in words:
                        start = _format_timestamp(w["start"]).replace(",", ".")
                        end = _format_timestamp(w.get("end", w["start"])).replace(",", ".")
                        text = w.get("word", w.get("text", "")).strip()
                        if "speaker" in seg:
                            text = f"{seg['speaker']}: {text}"
                        text = _normalize(text)
                        text = textwrap.fill(text, width=max_line_width)
                        lines = text.splitlines()[:max_lines]
                        f.write(f"{start} --> {end}\n")
                        f.write("\n".join(lines) + "\n\n")
                else:
                    start = _format_timestamp(seg["start"]).replace(",", ".")
                    end = _format_timestamp(seg["end"]).replace(",", ".")
                    text = seg["text"].strip()
                    if "speaker" in seg:
                        text = f"{seg['speaker']}: {text}"
                    text = _normalize(text)
                    text = textwrap.fill(text, width=max_line_width)
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


# Globals initialised in worker processes
ARGS: Dict[str, Any] = {}
DEVICE: torch.device | None = None
MODEL: Any | None = None
VAD_MODEL: Any | None = None
DIARIZE_MODEL: Any | None = None
OPTIONS: Dict[str, Any] = {}


def _init_worker(args: Dict[str, Any], options: Dict[str, Any]) -> None:
    """Initializer for worker processes to load heavy models once."""
    global ARGS, DEVICE, MODEL, VAD_MODEL, DIARIZE_MODEL, OPTIONS
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ARGS = args
    OPTIONS = options
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if getattr(DEVICE, "type", DEVICE) == "cuda" else "int8"
    device_str = DEVICE.type if hasattr(DEVICE, "type") else str(DEVICE)
    MODEL = whisperx.load_model(
        ARGS["model_size"],
        device_str,
        compute_type=compute_type,
        language=ARGS.get("language"),
    )
    VAD_MODEL = load_vad_model(
        getattr(DEVICE, "type", DEVICE),
        ARGS["vad_model"],
        vad_options={"onset": ARGS["vad_onset"], "offset": ARGS["vad_offset"]},
    )
    if ARGS.get("diarize"):
        from whisperx.diarize import load_diarize_model

        DIARIZE_MODEL = load_diarize_model(DEVICE)


def process_video(video: Path) -> Dict[str, Any]:
    """Process a single video file and capture timing and error details."""
    start_time = datetime.now().isoformat()
    logging.info("Processing %s", video)
    tmp_root = Path(".tmp_audio")
    tmp_root.mkdir(parents=True, exist_ok=True)
    error: str | None = None
    try:
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp_dir:
            audio_path = extract_audio(video, ARGS["audio_track"], Path(tmp_dir))
            segments = transcribe_file(
                audio_path,
                MODEL,
                VAD_MODEL,
                DEVICE,
                OPTIONS,
                DIARIZE_MODEL,
            )
            output_path = video
            if ARGS.get("output_dir"):
                root_dir = Path(ARGS["directory"])
                rel_path = video.relative_to(root_dir)
                output_path = Path(ARGS["output_dir"]) / rel_path
            output_path = output_path.with_suffix("." + ARGS["output_format"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            write_subtitles(
                segments,
                output_path,
                fmt=ARGS["output_format"],
                max_line_width=ARGS["max_line_width"],
                max_lines=ARGS["max_lines"],
                case=ARGS.get("case"),
                strip_punctuation=ARGS.get("strip_punctuation", False),
            )
    except Exception as exc:  # pragma: no cover - pragmatic logging
        logging.exception("Failed to generate subtitles for %s", video)
        error = str(exc)
    finally:
        gc.collect()
        if DEVICE and DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    end_time = datetime.now().isoformat()
    return {
        "video": str(video),
        "start": start_time,
        "end": end_time,
        "error": error,
        "success": error is None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate subtitles for videos")
    parser.add_argument(
        "directory",
        nargs="?",
        default="E:\Movies Series\One Pace - One Piece\[One Pace][101-105] Reverse Mountain [1080p]",
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
        default=1,
        help="Audio track index to extract (default: 1; use 0 for first track)",
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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes",
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

    options: Dict[str, Any] = {"language": args.language}
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

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = open(log_dir / "subtitle_run.json", "a", encoding="utf-8")
    failed_log = open("failed_subtitles.log", "a", encoding="utf-8")
    args_dict = vars(args)
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args_dict, options),
        ) as executor:
            future_to_video = {
                executor.submit(process_video, v): v for v in videos
            }
            for future in concurrent.futures.as_completed(future_to_video):
                video = future_to_video[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - worker failure path
                    logging.error("Processing %s failed: %s", video, exc)
                    failed_log.write(f"{video}: {exc}\n")
                    continue
                json.dump(result, run_log)
                run_log.write("\n")
                if not result["success"]:
                    failed_log.write(
                        f"{result['video']}: {result['error']}\n"
                    )
    except BrokenProcessPool as exc:  # pragma: no cover
        logging.error(
            "Worker initialization failed: %s", exc.__cause__ or exc
        )
        sys.exit(1)
    finally:
        failed_log.close()
        run_log.close()
    # Cleaning up temp directory is intentionally left out to aid debugging.  In
    # production one could remove it or place it under ``TemporaryDirectory``.


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
