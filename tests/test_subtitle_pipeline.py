import json
import os
import subprocess
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
from corrections import load_corrections, apply_corrections

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
    segs = [{"start": 0.0, "end": 2.0, "text": "hello world it's me"}]
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


def test_pipeline_skips_music_and_enforces_limits(tmp_path):
    segments = [
        {"start": 0.0, "end": 2.0, "text": "hello world it's me"},
        {"start": 2.2, "end": 4.0, "text": "teh cat"},
        {"start": 4.0, "end": 5.0, "text": "\u266a\u266a", "is_music": True},
    ]
    # Simulate skip_music by filtering out segments marked as music
    filtered = [seg for seg in segments if not seg.get("is_music")]
    seg_path = _write_segments(tmp_path, filtered)
    subs = load_segments(seg_path)
    rules = {"teh": "the"}
    for ev in subs.events:
        ev.text = apply_corrections(ev.plaintext, rules).replace("\n", "\\N")
    enforce_limits(subs, max_chars=5, max_lines=2, max_duration=2.0, min_gap=0.5)
    assert len(subs.events) == 2
    first, second = subs.events
    # Line wrapping and duration capping on the first event
    assert first.text == "hello\\Nworld"
    assert first.end - first.start == 2000
    # Correction application and gap insertion for the second event
    # The second event was corrected and shifted to ensure a gap
    assert second.text.startswith("the")
    assert "teh" not in second.text
    assert second.start - first.end == 500


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
    metrics_file = out_srt.with_suffix(".metrics.json")
    assert metrics_file.exists()
    metrics = json.loads(metrics_file.read_text())
    assert metrics["after"]["avg_cps"] <= metrics["before"]["avg_cps"]
    assert metrics["after"]["pct_cps_gt_17"] <= metrics["before"]["pct_cps_gt_17"]


def test_cli_default_output_path(tmp_path):
    segs = [{"start": 0.0, "end": 1.0, "text": "hello"}]
    seg_path = _write_segments(tmp_path, segs)
    subprocess.run(
        ["python", "subtitle_pipeline.py", "--segments", str(seg_path), "--transcript"],
        check=True,
        cwd=ROOT,
    )
    out_srt = seg_path.with_suffix(".srt")
    out_txt = seg_path.with_suffix(".txt")
    assert out_srt.exists()
    assert out_txt.exists()
    metrics_file = out_srt.with_suffix(".metrics.json")
    assert metrics_file.exists()
