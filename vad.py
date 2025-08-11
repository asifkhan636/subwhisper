"""Utilities for loading and validating Pyannote VAD models."""

from __future__ import annotations

import logging
from packaging import version

VAD_MODEL = "pyannote/voice-activity-detection"

logger = logging.getLogger(__name__)


def warn_if_incompatible_pyannote() -> None:
    """Warn when installed pyannote.audio is incompatible with ``VAD_MODEL``.

    The ``pyannote/voice-activity-detection`` pipeline requires
    ``pyannote.audio`` 2.x. Older releases (0.x/1.x) are incompatible and will
    likely fail at runtime. This check emits a warning when such a mismatch is
    detected but does not raise an exception so the caller can decide how to
    proceed.
    """

    try:
        import pyannote.audio  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        logger.warning("pyannote.audio not available: %s", exc)
        return

    installed = version.parse(pyannote.audio.__version__)
    if installed.major < 2:  # pragma: no cover - simple branch
        logger.warning(
            "pyannote.audio %s detected; upgrade to >=2.0 for %s",
            pyannote.audio.__version__,
            VAD_MODEL,
        )


def load_vad_model(device: str | None = None):
    """Load the pretrained Pyannote VAD pipeline.

    Parameters
    ----------
    device:
        Optional device string (e.g., ``"cpu"`` or ``"cuda"``) to move the
        pipeline to after loading.
    """
    from pyannote.audio import Pipeline  # imported lazily

    logger.info("Loading VAD pipeline: %s", VAD_MODEL)
    pipeline = Pipeline.from_pretrained(VAD_MODEL)
    if device:
        pipeline.to(device)
    return pipeline

