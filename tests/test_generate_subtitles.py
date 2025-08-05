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
    sys.modules.setdefault("torch", dummy_torch)

    dummy_whisperx = types.ModuleType("whisperx")
    dummy_whisperx.load_audio = lambda path: [0.0] * 16000
    dummy_whisperx.audio = types.SimpleNamespace(SAMPLE_RATE=16000)
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
    called = {}

    def fake_run(cmd, check, stdout, stderr):
        called["cmd"] = cmd
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(gs.subprocess, "run", fake_run)
    result = gs.extract_audio(video, 0, tmp_path)
    assert result == tmp_path / "clip.wav"
    assert called["cmd"][0] == "ffmpeg"
    assert f"0:a:0" in called["cmd"]


def test_transcribe_file(gs, monkeypatch):
    audio_path = Path("dummy.wav")
    dummy_audio = [0.0] * 8000
    monkeypatch.setattr(gs.whisperx, "load_audio", lambda p: dummy_audio)

    class FakeModel:
        def __init__(self):
            self.last_segments = None

        def transcribe(self, audio, segments=None, **options):
            self.last_segments = segments
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "hi"}]}

    called = {}

    def fake_vad(audio, onset=None, offset=None):
        called["onset"] = onset
        called["offset"] = offset
        return [{"start": 0.0, "end": 1.0}]

    def fake_diar(audio, segments):
        return [{"start": 0.0, "end": 1.0, "speaker": "S1"}]

    gs.ARGS["vad_onset"] = 0.3
    gs.ARGS["vad_offset"] = 0.5

    model = FakeModel()
    segments = gs.transcribe_file(
        audio_path,
        model,
        fake_vad,
        gs.torch.device("cpu"),
        {},
        fake_diar,
    )
    assert model.last_segments == [{"start": 0.0, "end": 1.0}]
    assert segments[0]["speaker"] == "S1"
    assert called["onset"] == 0.3
    assert called["offset"] == 0.5


def test_write_subtitles(gs, tmp_path):
    segments = [{"start": 0.0, "end": 1.0, "text": "Hello, WORLD!"}]
    out = tmp_path / "out"
    result = gs.write_subtitles(segments, out, fmt="srt", case="lower", strip_punctuation=True)
    text = result.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,000" in text
    assert "hello world" in text
