"""Utilities for transcribing audio with WhisperX and word-level alignment."""

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
        Optional ``(start, end)`` pairs marking regions to exclude.

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
    aligned = whisperx.align(
        result["segments"], align_model, metadata, audio, batch_size=batch_size
    )

    segments = aligned["segments"]
    if music_segments:
        segments = [
            seg
            for seg in segments
            if not _overlaps(seg["start"], seg["end"], music_segments)
        ]

    os.makedirs(outdir, exist_ok=True)
    output_path = os.path.join(outdir, "transcript.json")
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump({"segments": segments}, fh, ensure_ascii=False, indent=2)

    return output_path
