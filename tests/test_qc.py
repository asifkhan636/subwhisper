import pytest

from qc import compute_wer


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
