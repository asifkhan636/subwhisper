import pytest
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qc import collect_metrics, compute_wer


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
