"""Utilities for loading and validating Pyannote VAD models."""

from __future__ import annotations

import logging
from packaging import version
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
import urllib.request

VAD_MODEL = "pyannote/voice-activity-detection"

logger = logging.getLogger(__name__)


def warn_if_incompatible_pyannote() -> str | None:
    """Check whether ``pyannote.audio`` matches the model expectation.

    Attempts to read the ``requirements.txt`` for ``VAD_MODEL`` from the
    HuggingFace Hub to determine the expected ``pyannote.audio`` version. When
    the requirement cannot be determined (e.g. due to missing network
    connectivity) the function falls back to a simple major version check
    requiring ``>=2``.

    Returns
    -------
    Optional[str]
        Warning message when an incompatibility is detected, otherwise ``None``.
    """

    try:
        import pyannote.audio  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        return f"pyannote.audio not available: {exc}"

    installed = version.parse(pyannote.audio.__version__)

    expected: SpecifierSet | None = None
    try:  # pragma: no cover - network dependent
        url = f"https://huggingface.co/{VAD_MODEL}/raw/main/requirements.txt"
        with urllib.request.urlopen(url, timeout=5) as resp:  # type: ignore[arg-type]
            text = resp.read().decode()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("pyannote.audio"):
                req = Requirement(line)
                expected = req.specifier
                break
    except Exception:
        # Unable to fetch requirement; fall back to major version check
        pass

    if expected and installed not in expected:
        return (
            f"{VAD_MODEL} expects pyannote.audio {expected}, "
            f"but {pyannote.audio.__version__} is installed"
        )

    if installed.major < 2:  # pragma: no cover - simple branch
        return (
            f"pyannote.audio {pyannote.audio.__version__} detected; "
            f"upgrade to >=2.0 for {VAD_MODEL}"
        )

    return None


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

