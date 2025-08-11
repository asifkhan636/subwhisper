from pathlib import Path
import sys
import types
import json

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

import subwhisper_cli
from subwhisper_cli import _resolve_outputs, _process_one


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


def test_music_args_present():
    cli_path = Path(__file__).resolve().parents[1] / "subwhisper_cli.py"
    content = cli_path.read_text()
    assert "--music-threshold" in content
    assert "--music-min-gap" in content


def test_process_one_filters_short_music_segments(monkeypatch, tmp_path):
    seg_path = tmp_path / "segs.json"
    seg_path.write_text(json.dumps([[0.0, 0.4], [0.5, 1.5]]))

    def fake_preprocess_pipeline(**kwargs):
        return {"audio_wav": str(tmp_path / "audio.wav"), "music_segments": str(seg_path)}

    captured = {}

    def fake_transcribe_and_align(**kwargs):
        captured["segments"] = kwargs.get("music_segments")
        return {"transcript_json": str(tmp_path / "t.json"), "segments_json": str(tmp_path / "s.json")}

    monkeypatch.setattr(subwhisper_cli, "preprocess_pipeline", fake_preprocess_pipeline)
    monkeypatch.setattr(subwhisper_cli, "transcribe_and_align", fake_transcribe_and_align)
    monkeypatch.setattr(subwhisper_cli, "compute_source_fingerprint", lambda p: "fp")
    monkeypatch.setattr(subwhisper_cli, "load_manifest", lambda p, s: {})
    monkeypatch.setattr(subwhisper_cli, "stage_complete", lambda *a, **k: None)
    monkeypatch.setattr(subwhisper_cli, "is_stage_reusable", lambda *a, **k: (False, []))
    monkeypatch.setattr(subwhisper_cli, "acquire_lock", lambda p: None)
    monkeypatch.setattr(subwhisper_cli, "release_lock", lambda p: None)
    monkeypatch.setattr(subwhisper_cli, "clean_old_manifests", lambda *a, **k: 0)
    monkeypatch.setattr(subwhisper_cli, "load_segments", lambda *a, **k: types.SimpleNamespace(events=[]))
    monkeypatch.setattr(subwhisper_cli, "enforce_limits", lambda *a, **k: None)
    monkeypatch.setattr(subwhisper_cli, "write_outputs", lambda *a, **k: None)

    media = tmp_path / "video.mp4"
    media.write_text("x")

    code = _process_one(
        media=media,
        output_root=None,
        device="cpu",
        skip_music=False,
        enhanced_music_detection=False,
        music_threshold=0.6,
        music_min_duration=0.5,
        music_min_gap=0.5,
        music_count_warning=1000,
        beam_size=None,
        clean_intermediates=False,
        purge_all_on_success=False,
        write_transcript_flag=False,
        max_chars=45,
        max_lines=2,
        max_duration=6.0,
        min_gap=0.15,
        corrections_path=None,
        resume="off",
        force=False,
        resume_clean_days=None,
        cleaned_roots=set(),
    )

    assert code == 0
    assert captured["segments"] == [[0.5, 1.5]]
