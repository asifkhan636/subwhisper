"""Utilities for transcribing audio with WhisperX and word-level alignment.

The transcription pipeline writes two JSON files to the output directory:

``transcript.json``
    The raw WhisperX segments enriched with an ``is_music`` flag.

``segments.json``
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
"""

from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import whisperx


def _overlaps(seg_start: float, seg_end: float, music_segments: List[Tuple[float, float]]) -> bool:
    """Return True when a segment overlaps any music interval."""
    for m_start, m_end in music_segments:
        if seg_end > m_start and seg_start < m_end:
            return True
    return False


def transcribe_and_align(
    audio_path: str,
    outdir: str,
    model: str = "large-v2",
    compute_type: str = "float32",
    batch_size: int = 8,
    beam_size: int = 5,
    music_segments: Optional[List[Tuple[float, float]]] = None,
    skip_music: bool = False,
) -> str:
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
    batch_size:
        Batch size used for both transcription and alignment.
    beam_size:
        Beam size for decoder during transcription.
    music_segments:
        Optional ``(start, end)`` pairs marking regions containing music.
    skip_music:
        When ``True`` segments overlapping ``music_segments`` are removed
        entirely. Otherwise they are kept with ``is_music`` set to ``True``.

    Returns
    -------
    str
        Path to the JSON file containing aligned segments.
    """
    asr_model = whisperx.load_model(
        model, language="en", compute_type=compute_type, batch_size=batch_size
    )

    audio = whisperx.load_audio(audio_path)
    result = asr_model.transcribe(
        audio, batch_size=batch_size, beam_size=beam_size, language="en"
    )

    align_model, metadata = whisperx.load_align_model(
        model_name="WAV2VEC2_ASR_LARGE_LV60K_960H", language_code="en"
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
        aligned_result = whisperx.align(
            to_align, align_model, metadata, audio, batch_size=batch_size
        )
        aligned_segments = aligned_result["segments"]

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
    transcript_path = os.path.join(outdir, "transcript.json")
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

    segments_path = os.path.join(outdir, "segments.json")
    with open(segments_path, "w", encoding="utf-8") as fh:
        json.dump(simple_segments, fh, ensure_ascii=False, indent=2)

    return segments_path


def main() -> None:
    """Command-line interface for ``transcribe_and_align``."""
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Transcribe audio and align words using WhisperX."
    )
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument(
        "--outdir", required=True, help="Directory for the resulting JSON file"
    )
    parser.add_argument("--model", default="large-v2", help="Whisper model name")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for processing"
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size")
    parser.add_argument(
        "--compute-type", default="float32", help="Precision for WhisperX"
    )
    parser.add_argument(
        "--music-segments",
        help="JSON file containing a list of [start, end] music ranges",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info("Model: %s", args.model)
    logging.info("Batch size: %d", args.batch_size)
    logging.info("Beam size: %d", args.beam_size)
    logging.info("Compute type: %s", args.compute_type)
    logging.info("Output directory: %s", args.outdir)
    logging.info("Music segments: %s", args.music_segments or "None")

    music_ranges = None
    if args.music_segments:
        with open(args.music_segments, "r", encoding="utf-8") as fh:
            music_ranges = json.load(fh)

    outpath = transcribe_and_align(
        args.audio_path,
        args.outdir,
        model=args.model,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        beam_size=args.beam_size,
        music_segments=music_ranges,
    )
    print(outpath)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
