from pathlib import Path
import os
import tarfile
import time

from maintenance import prune_raw_intermediates, compress_old_runs


def test_prune_removes_intermediate_files(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "audio.wav").write_text("a", encoding="utf-8")
    (run_dir / "keep.srt").write_text("b", encoding="utf-8")

    removed = prune_raw_intermediates(str(runs), patterns=["*.wav"], dry_run=False)
    assert removed == 1
    assert not (run_dir / "audio.wav").exists()
    assert (run_dir / "keep.srt").exists()


def test_compress_old_runs(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "out.srt").write_text("data", encoding="utf-8")
    (run_dir / "audio.wav").write_text("raw", encoding="utf-8")
    prune_raw_intermediates(str(runs), patterns=["*.wav"], dry_run=False)
    old_time = time.time() - 40 * 86400
    os.utime(run_dir, (old_time, old_time))

    count = compress_old_runs(str(runs), days=30, dry_run=False)
    assert count == 1
    archive = runs / "r1.tar.gz"
    assert archive.exists()
    assert not run_dir.exists()
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "audio.wav" not in names

def test_prune_dry_run_keeps_files(tmp_path):
    runs = tmp_path / "runs"
    run_dir = runs / "r1"
    run_dir.mkdir(parents=True)
    target = run_dir / "audio.wav"
    target.write_text("data", encoding="utf-8")

    removed = prune_raw_intermediates(str(runs), patterns=["*.wav"], dry_run=True)
    assert removed == 1
    assert target.exists()


def test_compress_missing_root(tmp_path):
    missing = tmp_path / "none"
    count = compress_old_runs(str(missing), days=30, dry_run=False)
    assert count == 0
