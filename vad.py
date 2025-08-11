"""Utilities for loading and validating Pyannote VAD models."""

from __future__ import annotations

import logging
from packaging import version
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
import urllib.request

# Requires ``pyannote.audio`` >=2 according to the model card
VAD_MODEL = "pyannote/vad"

logger = logging.getLogger(__name__)


def warn_if_incompatible_pyannote() -> None:
    """Ensure ``pyannote.audio`` matches the model expectation.

    The function attempts to read the ``requirements.txt`` for ``VAD_MODEL``
    from the HuggingFace Hub to determine the expected ``pyannote.audio``
    version. When the requirement cannot be determined (e.g. due to missing
    network connectivity) the function falls back to a simple major version
    check requiring ``>=2``.  When an incompatibility is detected the function
    logs an actionable error message and raises ``RuntimeError`` to halt
    execution.
    """

    try:
        import pyannote.audio  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        msg = f"pyannote.audio not available: {exc}"
        logger.error(msg)
        raise RuntimeError(msg) from exc

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
        msg = (
            f"{VAD_MODEL} expects pyannote.audio {expected}, "
            f"but {pyannote.audio.__version__} is installed."
            " Please install a compatible version."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    if installed.major < 2:  # pragma: no cover - simple branch
        msg = (
            f"pyannote.audio {pyannote.audio.__version__} detected; "
            f"upgrade to >=2.0 for {VAD_MODEL}."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    logger.debug(
        "pyannote.audio %s is compatible with %s", pyannote.audio.__version__, VAD_MODEL
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

