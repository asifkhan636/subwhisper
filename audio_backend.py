"""Helpers for configuring audio backends.

This module ensures that torchaudio uses the ``soundfile`` backend on
Windows.  It also disables ``sox_io``-related warnings by setting the
``TORCHAUDIO_ENABLE_SOX_IO_BACKEND`` environment variable.  The selection is
deferred to a helper so that startup scripts can call it before importing
libraries like :mod:`speechbrain` which inspect the backend at import time.
"""

from __future__ import annotations

import logging
import os
import platform
import warnings

logger = logging.getLogger(__name__)


def setup_torchaudio_backend() -> None:
    """Configure torchaudio to use the ``soundfile`` backend on Windows.

    The function is intentionally silent and best-effort: if torchaudio is
    unavailable or the backend cannot be set, the failure is logged at debug
    level and the program continues with torchaudio's default behaviour.
    """

    if platform.system() != "Windows":
        return

    # ``torchaudio`` warns about the ``sox_io`` backend when the C++
    # extension is unavailable.  Setting this environment variable silences
    # the warning and makes the behaviour deterministic for callers.
    os.environ.setdefault("TORCHAUDIO_ENABLE_SOX_IO_BACKEND", "0")

    try:  # pragma: no cover - depends on environment
        import torchaudio  # type: ignore
    except Exception as exc:  # pragma: no cover - torchaudio missing
        logger.debug("torchaudio not available: %s", exc)
        return

    try:  # pragma: no cover - backend may be unavailable
        torchaudio.set_audio_backend("soundfile")
    except Exception as exc:  # pragma: no cover - backend may fail
        logger.debug("failed to set torchaudio backend: %s", exc)
    else:
        # ``speechbrain.utils.torch_audio_backend`` prints warnings about the
        # selected backend.  Suppress them since "soundfile" is expected on
        # Windows.
        warnings.filterwarnings(
            "ignore", module="speechbrain.utils.torch_audio_backend"
        )
        logging.getLogger(
            "speechbrain.utils.torch_audio_backend"
        ).setLevel(logging.ERROR)

