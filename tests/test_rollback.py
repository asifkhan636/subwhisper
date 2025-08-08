from pathlib import Path

from rollback import restore


def test_restore_copies_srt(tmp_path):
    run_id = "oldrun"
    output_root = tmp_path / "runs"
    src_dir = output_root / run_id / "episode"
    src_dir.mkdir(parents=True)
    srt_file = src_dir / f"episode_{run_id}.srt"
    srt_file.write_text("data", encoding="utf-8")

    dest = tmp_path / "restored"
    restore(run_id, str(output_root), str(dest))

    restored = dest / "episode" / f"episode_{run_id}.srt"
    assert restored.exists()
    assert restored.read_text() == "data"
