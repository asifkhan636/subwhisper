import pytest
import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qc import collect_metrics, compute_wer, validate_sync


def test_compute_wer_txt(tmp_path):
    ref = tmp_path / "ref.txt"
    ref.write_text("hello world")
    hyp = tmp_path / "hyp.txt"
    hyp.write_text("hello there world")
    assert compute_wer(str(hyp), str(ref)) == pytest.approx(0.5)


def test_compute_wer_srt(tmp_path):
    ref = tmp_path / "ref.srt"
    ref.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n<i>Hello</i> world\n\n",
        encoding="utf-8",
    )
    hyp = tmp_path / "hyp.srt"
    hyp.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n<i>Hello</i> there world\n\n",
        encoding="utf-8",
    )
    assert compute_wer(str(hyp), str(ref)) == pytest.approx(0.5)


def test_collect_metrics(tmp_path):
    srt = tmp_path / "test.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello\nWorld\n\n"
        "2\n00:00:01,000 --> 00:00:03,000\nAnother line\n\n",
        encoding="utf-8",
    )
    metrics = collect_metrics(str(srt))
    assert metrics["subtitle_count"] == 2
    assert metrics["avg_duration"] == pytest.approx(1.5)
    assert metrics["avg_lines"] == pytest.approx(1.5)
    assert metrics["warnings"] == []


def test_validate_sync(tmp_path, monkeypatch):
    srt = tmp_path / "tiny.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello world\n\n",
        encoding="utf-8",
    )
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"dummy")

    task_module = types.ModuleType("aeneas.task")

    class DummyTask:
        def __init__(self, config_string):
            self.config_string = config_string
            self.audio_file_path_absolute = ""
            self.text_file_path_absolute = ""
            self.sync_map_leaves = []

    task_module.Task = DummyTask

    exec_module = types.ModuleType("aeneas.executetask")

    class DummyExecuteTask:
        def __init__(self, task):
            self.task = task

        def execute(self):
            class Leaf:
                def __init__(self, begin):
                    self.begin = begin

            self.task.sync_map_leaves = [Leaf(0.0), Leaf(0.7)]

    exec_module.ExecuteTask = DummyExecuteTask

    pkg_module = types.ModuleType("aeneas")
    pkg_module.task = task_module
    pkg_module.executetask = exec_module

    monkeypatch.setitem(sys.modules, "aeneas", pkg_module)
    monkeypatch.setitem(sys.modules, "aeneas.task", task_module)
    monkeypatch.setitem(sys.modules, "aeneas.executetask", exec_module)

    metrics = validate_sync(str(srt), str(audio))
    assert metrics["word_count"] == 2
    assert metrics["mean_offset"] == pytest.approx(0.1)
    assert metrics["median_offset"] == pytest.approx(0.1)
    assert metrics["max_offset"] == pytest.approx(0.2)
