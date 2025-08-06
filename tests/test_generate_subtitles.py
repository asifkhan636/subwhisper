import importlib
import types
import sys
from pathlib import Path

import pytest


@pytest.fixture
def gs(monkeypatch):
    """Provide the generateSubtitles module with stubbed heavy deps."""
    import shutil
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    # Stub external modules before importing generateSubtitles
    dummy_torch = types.ModuleType("torch")
    dummy_torch.device = lambda *a, **k: "cpu"
    dummy_torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    dummy_torch.tensor = lambda x: x
    dummy_torch.from_numpy = lambda arr: types.SimpleNamespace(unsqueeze=lambda dim: arr)
    sys.modules.setdefault("torch", dummy_torch)

    dummy_whisperx = types.ModuleType("whisperx")
    dummy_whisperx.load_audio = lambda path: [0.0] * 16000
    dummy_whisperx.audio = types.SimpleNamespace(SAMPLE_RATE=16000)
    dummy_whisperx.load_align_model = lambda language, device: (None, None)
    dummy_whisperx.align = lambda segments, model, metadata, audio, device: {"segments": segments}
    sys.modules.setdefault("whisperx", dummy_whisperx)
    sys.modules.setdefault("whisperx.audio", dummy_whisperx.audio)
    dummy_vads_pyannote = types.ModuleType("whisperx.vads.pyannote")
    dummy_vads_pyannote.load_vad_model = lambda *a, **k: None
    sys.modules.setdefault("whisperx.vads", types.ModuleType("whisperx.vads"))
    sys.modules.setdefault("whisperx.vads.pyannote", dummy_vads_pyannote)
    sys.modules.setdefault("pyannote", types.ModuleType("pyannote"))
    sys.modules.setdefault("pyannote.audio", types.ModuleType("pyannote.audio"))

    monkeypatch.setattr(shutil, "which", lambda cmd: f"/usr/bin/{cmd}")

    module = importlib.import_module("generateSubtitles")
    return module


def test_extract_audio(gs, tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    called = {"cmds": []}

    def fake_run(cmd, check, stdout, stderr, text=False):
        called["cmds"].append(cmd)
        if cmd[0] == "ffprobe":
            return types.SimpleNamespace(returncode=0, stdout='{"streams": [{"index": 0}]}')
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(gs.subprocess, "run", fake_run)
    result = gs.extract_audio(video, 0, tmp_path)
    assert result == tmp_path / "clip.wav"
    ffmpeg_cmd = called["cmds"][-1]
    assert ffmpeg_cmd[0] == "ffmpeg"
    assert f"0:a:0" in ffmpeg_cmd


def test_extract_audio_missing_track(gs, tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"

    def fake_run(cmd, check, stdout, stderr, text=False):
        if cmd[0] == "ffprobe":
            return types.SimpleNamespace(returncode=0, stdout='{"streams": []}')
        raise AssertionError("ffmpeg should not be called")

    monkeypatch.setattr(gs.subprocess, "run", fake_run)
    with pytest.raises(ValueError):
        gs.extract_audio(video, 0, tmp_path)


def test_transcribe_file(gs, monkeypatch):
    audio_path = Path("dummy.wav")
    dummy_audio = [0.0] * 8000
    monkeypatch.setattr(gs.whisperx, "load_audio", lambda p: dummy_audio)

    class FakeModel:
        def __init__(self):
            self.last_kwargs = None

        def transcribe(self, audio, **kwargs):
            self.last_kwargs = kwargs
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}

    calls = {}

    def fake_load_align_model(language, device):
        calls["load_align_model"] = (language, device)
        return "am", "meta"

    def fake_align(segs, align_model, metadata, audio, device):
        calls["align"] = True
        return {"segments": [{"start": 0.0, "end": 1.0, "text": "aligned"}]}

    def fake_diar(audio, segments):
        return [{"start": 0.0, "end": 1.0, "speaker": "S1"}]

    gs.ARGS["vad_onset"] = 0.3
    gs.ARGS["vad_offset"] = 0.5

    monkeypatch.setattr(gs.whisperx, "load_align_model", fake_load_align_model)
    monkeypatch.setattr(gs.whisperx, "align", fake_align)

    model = FakeModel()
    segments = gs.transcribe_file(
        audio_path,
        model,
        None,
        gs.torch.device("cpu"),
        {},
        fake_diar,
    )
    assert model.last_kwargs["vad_filter"] is True
    assert model.last_kwargs["vad_parameters"] == {"onset": 0.3, "offset": 0.5}
    assert segments[0]["text"] == "aligned"
    assert segments[0]["speaker"] == "S1"
    assert calls["load_align_model"][1] == gs.torch.device("cpu")
    assert calls["align"] is True


def test_write_subtitles(gs, tmp_path):
    segments = [{"start": 0.0, "end": 1.0, "text": "Hello, WORLD!"}]
    out = tmp_path / "out"
    result = gs.write_subtitles(segments, out, fmt="srt", case="lower", strip_punctuation=True)
    text = result.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,000" in text
    assert "hello world" in text


def test_write_subtitles_word_grouping(gs, tmp_path):
    segments = [
        {
            "speaker": "S1",
            "words": [
                {"start": 0.0, "end": 0.4, "word": "Hello"},
                {"start": 0.5, "end": 0.8, "word": "WORLD"},
                {"start": 2.0, "end": 2.3, "word": "Again"},
            ],
        }
    ]
    out = tmp_path / "out"
    result = gs.write_subtitles(segments, out, fmt="srt", case="lower", pause_threshold=1.0)
    text = result.read_text(encoding="utf-8")
    assert text.count("-->") == 2
    assert "s1: hello world" in text
    assert "s1: again" in text


def test_write_subtitles_width_limit(gs, tmp_path):
    segments = [
        {
            "speaker": "S1",
            "words": [
                {"start": 0.0, "end": 0.3, "word": "hello"},
                {"start": 0.31, "end": 0.6, "word": "world"},
                {"start": 0.61, "end": 0.9, "word": "again"},
            ],
        }
    ]
    out = tmp_path / "out"
    result = gs.write_subtitles(
        segments,
        out,
        fmt="srt",
        case="lower",
        max_line_width=10,
        max_lines=1,
        pause_threshold=10.0,
    )
    text = result.read_text(encoding="utf-8")
    assert text.count("-->") == 3
    assert "s1: hello" in text
    assert "s1: world" in text
    assert "s1: again" in text
