"""Utilities for loading Faster-Whisper's Silero VAD."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Iterable, Optional

import numpy as np
import soundfile as sf
from faster_whisper.vad import VadOptions, get_speech_timestamps


logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """Simple speech span used by preprocessing tests and callers."""

    start: float
    end: float
    confidence: float = 1.0


def warn_if_incompatible_pyannote(models: Iterable[str] | None = None) -> None:
    """Deprecated compatibility shim kept for import stability."""

    return None


class SileroVADPipeline:
    """Thin wrapper around Faster-Whisper's Silero VAD helpers."""

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 0,
        min_silence_duration_ms: int = 2000,
        speech_pad_ms: int = 400,
    ) -> None:
        self.options = VadOptions(
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

    @staticmethod
    def _load_audio(inp) -> tuple[np.ndarray, int]:
        if isinstance(inp, dict):
            if inp.get("waveform") is not None:
                waveform = np.asarray(inp["waveform"], dtype=np.float32)
                if waveform.ndim > 1:
                    waveform = waveform.mean(axis=0)
                return waveform, int(inp.get("sample_rate", 16000))
            inp = inp.get("audio")

        if isinstance(inp, np.ndarray):
            waveform = np.asarray(inp, dtype=np.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=0)
            return waveform, 16000

        waveform, sampling_rate = sf.read(str(inp), dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        return waveform, int(sampling_rate)

    def __call__(self, inp) -> list[SpeechSegment]:
        waveform, sampling_rate = self._load_audio(inp)
        chunks = get_speech_timestamps(
            waveform,
            vad_options=self.options,
            sampling_rate=sampling_rate,
        )
        return [
            SpeechSegment(
                start=chunk["start"] / sampling_rate,
                end=chunk["end"] / sampling_rate,
            )
            for chunk in chunks
        ]


def load_vad_model(
    device: Optional[str] = None,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 0,
    min_silence_duration_ms: int = 2000,
    speech_pad_ms: int = 400,
):
    """Load a Silero VAD wrapper.

    ``device`` is accepted for compatibility but ignored since the bundled
    Silero model runs through ONNX Runtime on CPU.
    """

    if device:
        logger.debug("Ignoring unsupported VAD device override: %s", device)
    logger.info("Loading Silero VAD (threshold=%s)", threshold)
    return SileroVADPipeline(
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=speech_pad_ms,
    )
