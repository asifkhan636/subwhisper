"""Quality control utilities."""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import jiwer
import pysubs2

logger = logging.getLogger(__name__)


def _strip_markup(text: str) -> str:
    """Remove simple markup tags from ``text``."""
    return re.sub(r"<[^>]+>", "", text)


def _load_text(path: Path) -> str:
    """Load subtitle or plain text from ``path``.

    Parameters
    ----------
    path:
        Location of ``.srt`` or ``.txt`` file.
    """
    if path.suffix.lower() == ".srt":
        subs = pysubs2.load(str(path))
        return " ".join(event.plaintext for event in subs)
    if path.suffix.lower() == ".txt":
        return _strip_markup(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported file type: {path.suffix}")


def compute_wer(hyp_path: str, ref_path: str) -> float:
    """Compute word error rate between two files.

    Parameters
    ----------
    hyp_path, ref_path:
        Paths to the hypothesis and reference files.

    Returns
    -------
    float
        The word error rate between ``hyp_path`` and ``ref_path``.
    """
    hyp = _load_text(Path(hyp_path))
    ref = _load_text(Path(ref_path))
    score = jiwer.wer(ref, hyp)
    logger.info("WER(%s, %s) = %.3f", hyp_path, ref_path, score)
    return score


def main(argv: Iterable[str] | None = None) -> float:
    """Command-line interface for :func:`compute_wer`."""
    parser = argparse.ArgumentParser(description="Compute word error rate")
    parser.add_argument("hyp")
    parser.add_argument("ref")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    return compute_wer(args.hyp, args.ref)


if __name__ == "__main__":  # pragma: no cover - convenience CLI
    main()
