"""Utilities for loading and applying subtitle correction rules."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Mapping

try:  # pragma: no cover - optional dependency
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None

logger = logging.getLogger(__name__)


def load_corrections(path: Path) -> Dict[str, str]:
    """Load correction rules from a JSON or YAML file.

    Parameters
    ----------
    path:
        Path to the JSON/YAML file.

    Returns
    -------
    dict
        Mapping of patterns to replacement strings.
    """
    with path.open("r", encoding="utf-8") as fh:
        if path.suffix.lower() in {".yml", ".yaml"}:
            if yaml is None:  # pragma: no cover - depends on optional package
                raise RuntimeError("PyYAML is required to load YAML files")
            data = yaml.safe_load(fh) or {}
        else:
            data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Correction rules must be a mapping")
    return {str(k): str(v) for k, v in data.items()}


def apply_corrections(text: str, rules: Mapping[str, str], use_regex: bool = False) -> str:
    """Apply correction ``rules`` to ``text``.

    Replacements are applied independently to each line in ``text``. When
    ``use_regex`` is ``True`` the keys in ``rules`` are treated as regular
    expressions. When debug logging is enabled, the original and corrected line
    are logged whenever a change is made.

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
    result_lines: List[str] = []
    for raw_line in text.splitlines(keepends=True):
        if raw_line.endswith("\n"):
            line, end = raw_line[:-1], "\n"
        else:
            line, end = raw_line, ""
        original = line
        for pattern, repl in rules.items():
            if use_regex:
                line = re.sub(pattern, repl, line)
            else:
                line = line.replace(pattern, repl)
        if logger.isEnabledFor(logging.DEBUG) and line != original:
            logger.debug("apply_corrections: %r -> %r", original, line)
        result_lines.append(line + end)
    return "".join(result_lines)
