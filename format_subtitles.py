"""Subtitle formatting utilities."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

logger = logging.getLogger(__name__)


class ConfigDict(dict):
    """Helper dict allowing attribute access for tests."""

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - trivial
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


def _as_dict(config: Any) -> Dict[str, Any]:
    """Return ``config`` as a dictionary."""
    if isinstance(config, dict):
        return config
    # Try to handle SimpleNamespace or similar objects
    return {k: getattr(config, k) for k in dir(config) if not k.startswith("__")}


def load_replacements(path: str) -> Dict[str, str]:
    """Load replacement rules from a JSON or YAML file.

    Parameters
    ----------
    path:
        Path to a JSON or YAML file containing a mapping of patterns to
        replacement strings.

    Returns
    -------
    dict
        Mapping of patterns to replacement strings.
    """
    with open(path, "r", encoding="utf-8") as fh:
        if path.lower().endswith((".yml", ".yaml")):
            if yaml is None:  # pragma: no cover - depends on optional package
                raise RuntimeError("PyYAML is required to load YAML files")
            data = yaml.safe_load(fh) or {}
        else:
            data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Replacement rules must be a mapping")
    return {str(k): str(v) for k, v in data.items()}


def apply_corrections(text: str, rules: Dict[str, str], use_regex: bool = False) -> str:
    """Apply replacement ``rules`` to ``text``.

    When ``use_regex`` is ``True`` the keys in ``rules`` are treated as regular
    expressions. Each performed replacement is logged at debug level allowing
    callers to enable a ``--debug`` flag in command-line interfaces.

    Parameters
    ----------
    text:
        The input text to transform.
    rules:
        Mapping of patterns to replacement strings.
    use_regex:
        Interpret keys in ``rules`` as regular expressions.

    Returns
    -------
    str
        The transformed text.
    """
    for pattern, repl in rules.items():
        if use_regex:
            text, count = re.subn(pattern, repl, text)
        else:
            count = text.count(pattern)
            if count:
                text = text.replace(pattern, repl)
        if count:
            logger.debug("Replaced %r -> %r (%d matches)", pattern, repl, count)
    return text


def format_subtitles(segments: List[Dict[str, Any]], config: Any) -> List[Dict[str, Any]]:
    """Format segments into subtitle blocks.

    Parameters
    ----------
    segments:
        Sequence of segment dictionaries. Each segment must contain ``start``,
        ``end``, ``text`` and optionally ``words`` and ``is_music``.
    config:
        Mapping or object providing ``max_line_length``, ``max_line_count``,
        ``max_duration``, ``skip_music`` and ``gap``.

    Returns
    -------
    list of dict
        Subtitle blocks with ``start``, ``end`` and ``lines`` keys.
    """
    cfg = _as_dict(config)
    max_line_length = int(cfg["max_line_length"])
    max_line_count = int(cfg["max_line_count"])
    max_duration = float(cfg["max_duration"])
    skip_music = bool(cfg.get("skip_music", False))
    merge_gap = float(cfg.get("gap", 0.0))

    results: List[Dict[str, Any]] = []

    current_lines: List[str] = []
    current_line = ""
    block_start: float | None = None
    block_end: float | None = None

    def finalize() -> None:
        nonlocal current_lines, current_line, block_start, block_end
        if block_start is None:
            return
        if current_line:
            current_lines.append(current_line)
        start = block_start
        if results:
            prev_end = results[-1]["end"]
            if start < prev_end + 0.1:
                start = prev_end + 0.1
        results.append({"start": start, "end": block_end, "lines": current_lines})
        current_lines = []
        current_line = ""
        block_start = None
        block_end = None

    for seg in segments:
        if skip_music and seg.get("is_music"):
            continue
        words = seg.get("words") or []
        if not words:
            words = [{"word": seg.get("text", ""), "start": seg["start"], "end": seg["end"]}]

        for word in words:
            w_text = word.get("word", "").strip()
            if not w_text:
                continue
            w_start = float(word.get("start", seg["start"]))
            w_end = float(word.get("end", w_start))

            if block_start is None:
                block_start = w_start
                current_line = w_text
                block_end = w_end
                continue

            # start new block when gap between words is large
            if w_start - block_end > merge_gap:
                finalize()
                block_start = w_start
                current_line = w_text
                block_end = w_end
                continue

            # start new block if adding word exceeds max duration
            if w_end - block_start > max_duration:
                finalize()
                block_start = w_start
                current_line = w_text
                block_end = w_end
                continue

            # try to append to current line
            candidate = f"{current_line} {w_text}" if current_line else w_text
            if len(candidate) <= max_line_length:
                current_line = candidate
                block_end = w_end
            else:
                # line full -> push current line to lines
                current_lines.append(current_line)
                if len(current_lines) >= max_line_count:
                    finalize()
                    block_start = w_start
                    current_line = w_text
                    block_end = w_end
                else:
                    current_line = w_text
                    block_end = w_end

    finalize()
    return results
