# Summary of changes
# ------------------
# - Removed unsupported ``batch_size`` argument from ``whisperx.load_model`` to
#   avoid ``TypeError`` with newer WhisperX versions. ``batch_size`` is now only
#   passed to ``transcribe`` and ``align``.
# - Added optional ``device`` parameter with automatic default and CLI flag.
# - Added optional ``stem`` parameter to customize output file names.

"""Utilities for transcribing audio with WhisperX and word-level alignment.

The script expects the mono 16 kHz WAV produced by ``preproc.py`` and,
optionally, a ``music_segments.json`` file containing ``[start, end]`` pairs
to mark regions with background music.

The transcription pipeline writes two JSON files to the output directory. By
default they are named ``transcript.json`` and ``segments.json`` but a custom
``--stem`` can prefix the filenames (e.g., ``MyEp.transcript.json``).

``transcript.json`` / ``<stem>.transcript.json``
    The raw WhisperX segments enriched with an ``is_music`` flag.

``segments.json`` / ``<stem>.segments.json``
    A simplified representation used by downstream tooling.  It contains a
    list of segments following the schema::

        [
            {
                "start": float,
                "end": float,
                "text": str,
                "words": [
                    {"word": str, "start": float, "end": float},
                    ...
                ],
            },
            ...
        ]

    ``words`` is empty when a segment was skipped during alignment (e.g. for
    music sections).

Command line usage::

    python transcribe.py cleaned.wav --outdir transcript \
        --music-segments preproc/music_segments.json --device cuda \
        [--model large-v3-turbo] [--batch-size 8] [--beam-size 5] \
        [--compute-type float32] [--stem MyEp]
"""

from __future__ import annotations

import json
import logging
import os
import inspect
from typing import List, Optional, Tuple

from audio_backend import setup_torchaudio_backend

setup_torchaudio_backend()

import torch
import whisperx
import pysubs2

from vad import warn_if_incompatible_pyannote

from subtitle_pipeline import spellcheck_lines


ALIGN_MODEL_NAME = "WAV2VEC2_ASR_LARGE_LV60K_960H"

logger = logging.getLogger(__name__)


def _overlaps(seg_start: float, seg_end: float, music_segments: List[Tuple[float, float]]) -> bool:
    """Return True when a segment overlaps any music interval."""
    for m_start, m_end in music_segments:
        if seg_end > m_start and seg_start < m_end:
            return True
    return False




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
    """Transcribe ``audio_path`` and align words with WhisperX.

    Parameters
    ----------
    audio_path:
        Path to the audio file to transcribe.
    outdir:
        Directory where the resulting JSON file is written.
    model:
        Whisper model name to use for transcription.
    compute_type:
        Precision for WhisperX (e.g., ``float32`` or ``float16``).
    device:
        Torch device on which to run the model. Defaults to ``"cuda"`` if
        available, otherwise ``"cpu"``.
    batch_size:
        Batch size used for both transcription and alignment.
    beam_size:
        Optional beam size for decoder during transcription. When ``None``, the
        model's default beam size is used.
    music_segments:
        Optional ``(start, end)`` pairs marking regions containing music.
    skip_music:
        When ``True`` segments overlapping ``music_segments`` are removed
        entirely. Otherwise they are kept with ``is_music`` set to ``True``.
    spellcheck:
        Run a LanguageTool spell check on the final segment texts when ``True``.
    stem:
        Optional filename stem for output JSON files. When provided, files are
        named ``<stem>.transcript.json`` and ``<stem>.segments.json``.
    resume_outputs:
        Optional mapping of expected output files. When both ``transcript_json``
        and ``segments_json`` exist, transcription and alignment are skipped and
        these paths are returned.

    Returns
    -------
    dict
        Mapping containing ``transcript_json`` and ``segments_json`` paths.
    """
    warn_if_incompatible_pyannote()

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
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    asr_model = whisperx.load_model(
        model, device, language="en", compute_type=compute_type
    )
    # NOTE: ``batch_size`` is not accepted by ``whisperx.load_model`` in current
    # versions. Specify ``batch_size`` only in ``transcribe`` and ``align`` calls.

    audio = whisperx.load_audio(audio_path)
    transcribe_kwargs = {"batch_size": batch_size, "language": "en"}
    if beam_size is not None:
        sig = inspect.signature(asr_model.transcribe)
        if "beam_size" in sig.parameters:
            transcribe_kwargs["beam_size"] = beam_size
    try:
        result = asr_model.transcribe(audio, **transcribe_kwargs)
    except Exception as exc:  # pragma: no cover - depends on backend
        logger.error("Transcription failed: %s", exc)
        if "out of memory" in str(exc).lower():
            logger.error(
                "Out of memory during transcription. Try smaller batch or beam size."
            )
        raise

    logger.info("Alignment model: %s", ALIGN_MODEL_NAME)
    align_model, metadata = whisperx.load_align_model(
        model_name=ALIGN_MODEL_NAME, language_code="en", device=device
    )

    segments = result["segments"]
    processed: List[dict] = []
    to_align: List[dict] = []
    for seg in segments:
        is_music = bool(music_segments) and _overlaps(
            seg["start"], seg["end"], music_segments
        )
        if is_music and skip_music:
            continue
        seg["is_music"] = is_music
        processed.append(seg)
        if not is_music:
            to_align.append(seg)

    aligned_segments: List[dict] = []
    if to_align:
        try:
            aligned_result = whisperx.align(
                to_align, align_model, metadata, audio, batch_size=batch_size
            )
        except Exception as exc:  # pragma: no cover - depends on backend
            logger.error("Alignment failed: %s", exc)
            if "out of memory" in str(exc).lower():
                logger.error(
                    "Out of memory during alignment. Try smaller batch or beam size."
                )
            else:
                logger.error("Consider retrying with smaller settings.")
            raise
        else:
            aligned_segments = aligned_result["segments"]
            for seg in aligned_segments:
                if seg.get("words"):
                    seg["start"] = seg["words"][0]["start"]
                    seg["end"] = seg["words"][-1]["end"]

    final_segments: List[dict] = []
    aligned_iter = iter(aligned_segments)
    for seg in processed:
        if seg["is_music"]:
            final_segments.append(seg)
        else:
            aligned_seg = next(aligned_iter)
            aligned_seg["is_music"] = False
            final_segments.append(aligned_seg)

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
            "words": [
                {"word": w["word"], "start": w["start"], "end": w["end"]}
                for w in seg.get("words", [])
            ],
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
        description="Transcribe audio and align words using WhisperX."
    )
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument(
        "--outdir", required=True, help="Directory for the resulting JSON file"
    )
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model name")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=default_device,
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
        "--compute-type", default="float32", help="Precision for WhisperX"
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
    logger.info("Alignment model: %s", ALIGN_MODEL_NAME)
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
