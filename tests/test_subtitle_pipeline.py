import json
import os
import subprocess
import sys
import types
import pathlib
import pytest
import pysubs2

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subtitle_pipeline import load_segments, enforce_limits
from format_subtitles import load_corrections, apply_corrections

# Stub for language_tool_python to avoid external dependency
lt_stub = types.ModuleType("language_tool_python")
class _DummyTool:
    def __init__(self, *args, **kwargs):
        pass

    def correct(self, text: str) -> str:
        return text

lt_stub.LanguageTool = _DummyTool
sys.modules.setdefault("language_tool_python", lt_stub)


def _write_segments(tmp_path, segments):
    seg_path = tmp_path / "segments.json"
    seg_path.write_text(json.dumps(segments))
    return seg_path


def test_enforce_limits_line_split_and_duration(tmp_path):
    segs = [{"start": 0.0, "end": 10.0, "text": "hello world it's me"}]
    seg_path = _write_segments(tmp_path, segs)
    subs = load_segments(seg_path)
    enforce_limits(subs, max_chars=5, max_lines=2, max_duration=2.0, min_gap=0.0)
    ev = subs.events[0]
    assert ev.text == "hello\\Nworld"
    assert ev.end - ev.start == 2000


def test_apply_corrections(tmp_path):
    segs = [{"start": 0.0, "end": 1.0, "text": "teh cat"}]
    seg_path = _write_segments(tmp_path, segs)
    corr_path = tmp_path / "corr.json"
    corr_path.write_text(json.dumps({"teh": "the"}))
    subs = load_segments(seg_path)
    rules = load_corrections(corr_path)
    for ev in subs.events:
        ev.text = apply_corrections(ev.plaintext, rules).replace("\n", "\\N")
    assert subs.events[0].text == "the cat"


def test_cli_generates_srt(tmp_path):
    segs = [{"start": 0.0, "end": 1.0, "text": "hello"}]
    seg_path = _write_segments(tmp_path, segs)
    corr_path = tmp_path / "corr.json"
    corr_path.write_text(json.dumps({"hello": "hi"}))
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "language_tool_python.py").write_text(
        "class LanguageTool:\n"
        "    def __init__(self, *a, **k):\n        pass\n"
        "    def correct(self, text):\n        return text\n"
    )
    out_srt = tmp_path / "out.srt"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(stub_dir) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            "python",
            "subtitle_pipeline.py",
            "--segments",
            str(seg_path),
            "--output",
            str(out_srt),
            "--corrections",
            str(corr_path),
            "--spellcheck",
        ],
        check=True,
        cwd=ROOT,
        env=env,
    )
    assert out_srt.exists()
    assert "hi" in out_srt.read_text(encoding="utf-8")
