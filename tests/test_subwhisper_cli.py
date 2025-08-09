from pathlib import Path
import sys
import types

preproc_stub = types.ModuleType("preproc")
preproc_stub.preprocess_pipeline = lambda *a, **k: None
sys.modules.setdefault("preproc", preproc_stub)

transcribe_stub = types.ModuleType("transcribe")
transcribe_stub.transcribe_and_align = lambda *a, **k: None
sys.modules.setdefault("transcribe", transcribe_stub)

subtitle_stub = types.ModuleType("subtitle_pipeline")
subtitle_stub.load_segments = lambda *a, **k: None
subtitle_stub.enforce_limits = lambda *a, **k: None
subtitle_stub.write_outputs = lambda *a, **k: None
sys.modules.setdefault("subtitle_pipeline", subtitle_stub)

corrections_stub = types.ModuleType("corrections")
corrections_stub.load_corrections = lambda *a, **k: None
corrections_stub.apply_corrections = lambda *a, **k: None
sys.modules.setdefault("corrections", corrections_stub)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subwhisper_cli import _resolve_outputs


def test_no_sync_flag_absent():
    cli_path = Path(__file__).resolve().parents[1] / "subwhisper_cli.py"
    content = cli_path.read_text()
    assert "--no-sync" not in content
    assert "no_sync" not in content
    assert "--skip-music" in content


def test_resolve_outputs_same_stem(tmp_path):
    media1 = tmp_path / "dir1" / "foo.mp4"
    media2 = tmp_path / "dir2" / "foo.mkv"
    media1.parent.mkdir(parents=True)
    media2.parent.mkdir(parents=True)
    media1.touch()
    media2.touch()

    output_root = tmp_path / "out"
    srt1, txt1, _ = _resolve_outputs(media1, output_root)
    srt2, txt2, _ = _resolve_outputs(media2, output_root)

    expected_dir = output_root / "foo"
    assert srt1 == expected_dir / "foo.srt"
    assert txt1 == expected_dir / "foo.txt"
    assert srt2 == srt1
    assert txt2 == txt1
    assert expected_dir.is_dir()
