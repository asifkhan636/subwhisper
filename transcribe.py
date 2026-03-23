"""Utilities for transcribing audio with Faster-Whisper word timestamps.

The script expects the mono 16 kHz WAV produced by ``preproc.py`` and,
optionally, a ``music_segments.json`` file containing ``[start, end]`` pairs
to mark regions with background music.

The transcription pipeline writes two JSON files to the output directory. By
default they are named ``transcript.json`` and ``segments.json`` but a custom
``--stem`` can prefix the filenames (for example ``MyEp.transcript.json``).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable, List, Optional, Tuple

import ctranslate2
import pysubs2
from faster_whisper import BatchedInferencePipeline, WhisperModel

from subtitle_pipeline import spellcheck_lines


logger = logging.getLogger(__name__)

SUPPORTED_COMPUTE_TYPES = {
    "auto",
    "default",
    "int8",
    "int8_float16",
    "float16",
    "float32",
}
CPU_COMPUTE_TYPES = {"auto", "default", "int8", "float32"}
CUDA_COMPUTE_TYPES = {
    "auto",
    "default",
    "int8",
    "int8_float16",
    "float16",
    "float32",
}


def _overlaps(
    seg_start: float, seg_end: float, music_segments: List[Tuple[float, float]]
) -> bool:
    """Return True when a segment overlaps any music interval."""

    for m_start, m_end in music_segments:
        if seg_end > m_start and seg_start < m_end:
            return True
    return False


def _default_device() -> str:
    """Return ``cuda`` when available through CTranslate2, else ``cpu``."""

    try:
        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:  # pragma: no cover - depends on runtime
        return "cpu"


def _validate_runtime_options(device: str, compute_type: str) -> None:
    """Fail early on invalid Faster-Whisper runtime options."""

    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{device}'. Use 'cpu' or 'cuda'.")

    if compute_type not in SUPPORTED_COMPUTE_TYPES:
        valid = ", ".join(sorted(SUPPORTED_COMPUTE_TYPES))
        raise ValueError(
            f"Unsupported compute type '{compute_type}'. Supported values: {valid}."
        )

    allowed = CUDA_COMPUTE_TYPES if device == "cuda" else CPU_COMPUTE_TYPES
    if compute_type not in allowed:
        valid = ", ".join(sorted(allowed))
        raise ValueError(
            f"Compute type '{compute_type}' is not supported on {device}. "
            f"Supported values: {valid}."
        )


def _normalize_segment(segment) -> dict:
    """Convert a Faster-Whisper segment into the repo's JSON schema."""

    words = []
    if getattr(segment, "words", None):
        for word in segment.words:
            if word.start is None or word.end is None:
                continue
            words.append(
                {
                    "word": word.word,
                    "start": float(word.start),
                    "end": float(word.end),
                }
            )

    start = float(segment.start)
    end = float(segment.end)
    if words:
        start = words[0]["start"]
        end = words[-1]["end"]

    return {
        "start": start,
        "end": end,
        "text": segment.text.strip(),
        "words": words,
    }


def _transcribe_segments(
    audio_path: str,
    model_name: str,
    device: str,
    compute_type: str,
    batch_size: int,
    beam_size: Optional[int],
) -> List[dict]:
    """Run Faster-Whisper and return normalized segments."""

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    engine = BatchedInferencePipeline(model=model) if batch_size > 1 else model

    transcribe_kwargs = {
        "language": "en",
        "word_timestamps": True,
        "vad_filter": True,
    }
    if beam_size is not None:
        transcribe_kwargs["beam_size"] = beam_size
    if batch_size > 1:
        transcribe_kwargs["batch_size"] = batch_size

    try:
        segments, info = engine.transcribe(audio_path, **transcribe_kwargs)
        raw_segments = list(segments)
    except Exception as exc:  # pragma: no cover - depends on backend
        logger.error("Transcription failed: %s", exc)
        if "out of memory" in str(exc).lower():
            logger.error(
                "Out of memory during transcription. Try smaller batch or beam size."
            )
        raise

    logger.info(
        "Detected language: %s (probability %.3f)",
        getattr(info, "language", "unknown"),
        getattr(info, "language_probability", 0.0),
    )
    return [_normalize_segment(segment) for segment in raw_segments]


def transcribe_and_align(
    audio_path: str,
    outdir: str,
    model: str = "large-v3-turbo",
    compute_type: str = "float32",
    device: Optional[str] = None,
    batch_size: int = 8,
    beam_size: Optional[int] = None,
    music_segments: Optional[List[Tuple[float, float]]] = None,
    skip_music: bool = False,
    spellcheck: bool = False,
    stem: Optional[str] = None,
    resume_outputs: Optional[dict] = None,
) -> dict:
    """Transcribe ``audio_path`` and emit the existing JSON output schema."""

    if resume_outputs and all(
        resume_outputs.get(k) and os.path.exists(resume_outputs[k])
        for k in ("transcript_json", "segments_json")
    ):
        logger.info("resume: using existing")
        return {
            "transcript_json": resume_outputs["transcript_json"],
            "segments_json": resume_outputs["segments_json"],
        }

    if device is None:
        device = _default_device()
    _validate_runtime_options(device, compute_type)
    logger.info("Device: %s", device)

    segments = _transcribe_segments(
        audio_path=audio_path,
        model_name=model,
        device=device,
        compute_type=compute_type,
        batch_size=batch_size,
        beam_size=beam_size,
    )

    final_segments: List[dict] = []
    for seg in segments:
        is_music = bool(music_segments) and _overlaps(
            seg["start"], seg["end"], music_segments
        )
        if is_music and skip_music:
            continue

        transcript_seg = {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "is_music": is_music,
        }
        if not is_music:
            transcript_seg["words"] = seg["words"]
        final_segments.append(transcript_seg)

    os.makedirs(outdir, exist_ok=True)
    transcript_path = (
        os.path.join(outdir, f"{stem}.transcript.json")
        if stem
        else os.path.join(outdir, "transcript.json")
    )
    with open(transcript_path, "w", encoding="utf-8") as fh:
        json.dump({"segments": final_segments}, fh, ensure_ascii=False, indent=2)

    simple_segments = [
        {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "words": [] if seg["is_music"] else list(seg.get("words", [])),
        }
        for seg in final_segments
    ]

    if spellcheck:
        subs = pysubs2.load_from_whisper(simple_segments)
        spellcheck_lines(subs)
        for seg, ev in zip(simple_segments, subs.events):
            seg["text"] = ev.plaintext

    segments_path = (
        os.path.join(outdir, f"{stem}.segments.json")
        if stem
        else os.path.join(outdir, "segments.json")
    )
    with open(segments_path, "w", encoding="utf-8") as fh:
        json.dump(simple_segments, fh, ensure_ascii=False, indent=2)
    logger.info("Transcription complete. JSON output at %s", segments_path)

    return {"transcript_json": transcript_path, "segments_json": segments_path}


def main() -> None:
    """Command-line interface for ``transcribe_and_align``."""

    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe audio and emit word timestamps using Faster-Whisper."
    )
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument(
        "--outdir", required=True, help="Directory for the resulting JSON file"
    )
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model name")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=_default_device(),
        help="Device to run the model on",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for processing"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size (default: model default)",
    )
    parser.add_argument(
        "--compute-type", default="float32", help="Precision for Faster-Whisper"
    )
    parser.add_argument(
        "--music-segments",
        help="JSON file containing a list of [start, end] music ranges",
    )
    parser.add_argument(
        "--skip-music",
        action="store_true",
        help="Drop segments overlapping music ranges",
    )
    parser.add_argument(
        "--spellcheck",
        action="store_true",
        help="Run LanguageTool spell check on output (slow; requires Java)",
    )
    parser.add_argument(
        "--stem",
        help="Filename stem for output JSON files",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Model: %s", args.model)
    logger.info("Compute type: %s", args.compute_type)
    logger.info("Device: %s", args.device)
    logger.info("Beam size: %s", args.beam_size)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Output directory: %s", args.outdir)
    logger.info("Music segments: %s", args.music_segments or "None")
    logger.info("Filename stem: %s", args.stem or "None")

    music_ranges = None
    if args.music_segments:
        with open(args.music_segments, "r", encoding="utf-8") as fh:
            music_ranges = json.load(fh)

    outputs = transcribe_and_align(
        args.audio_path,
        args.outdir,
        model=args.model,
        compute_type=args.compute_type,
        device=args.device,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        music_segments=music_ranges,
        skip_music=args.skip_music,
        spellcheck=args.spellcheck,
        stem=args.stem,
    )
    logger.info("JSON output written to %s", outputs["segments_json"])
    print(outputs["segments_json"])


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
