import importlib
import importlib
import json
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
    monkeypatch.setenv("SUBWHISPER_SKIP_DEP_CHECK", "1")

    # Stub external modules before importing generateSubtitles
    dummy_torch = types.ModuleType("torch")
    dummy_torch.device = lambda *a, **k: "cpu"
    dummy_torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    dummy_torch.tensor = lambda x: x
    dummy_torch.from_numpy = lambda arr: types.SimpleNamespace(unsqueeze=lambda dim: arr)
    dummy_torch.__version__ = "2.5.0"
    sys.modules["torch"] = dummy_torch

    dummy_whisperx = types.ModuleType("whisperx")
    dummy_whisperx.load_audio = lambda path: [0.0] * 16000
    dummy_whisperx.audio = types.SimpleNamespace(SAMPLE_RATE=16000)
    dummy_whisperx.load_align_model = lambda language, device: (None, None)
    dummy_whisperx.align = lambda segments, model, metadata, audio, device: {"segments": segments}
    dummy_diarize = types.ModuleType("whisperx.diarize")
    dummy_diarize.load_diarize_model = lambda *a, **k: (lambda *aa, **kk: [])
    dummy_whisperx.diarize = dummy_diarize
    dummy_whisperx.__version__ = "3.4.2"
    dummy_utils = types.ModuleType("whisperx.utils")

    def fake_write_srt(segments, path):
        with open(path, "w", encoding="utf-8") as f:
            for s in segments:
                f.write(f"{s['start']} --> {s['end']}\n{s['text']}\n\n")

    dummy_utils.write_srt = fake_write_srt
    dummy_utils.write_vtt = fake_write_srt
    dummy_whisperx.utils = dummy_utils
    sys.modules["whisperx"] = dummy_whisperx
    sys.modules["whisperx.audio"] = dummy_whisperx.audio
    sys.modules["whisperx.diarize"] = dummy_diarize
    sys.modules["whisperx.utils"] = dummy_utils

    dummy_numpy = types.ModuleType("numpy")
    dummy_numpy.asarray = lambda x: x
    sys.modules.setdefault("numpy", dummy_numpy)

    dummy_pyannote = types.ModuleType("pyannote")
    dummy_pyannote.__path__ = []
    dummy_pyannote_audio = types.ModuleType("pyannote.audio")
    dummy_pyannote_audio.__version__ = "3.3.0"
    sys.modules["pyannote"] = dummy_pyannote
    sys.modules["pyannote.audio"] = dummy_pyannote_audio

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


def test_detect_audio_track_pref_language(gs, tmp_path, monkeypatch):
    video = tmp_path / "clip.mkv"

    def fake_run(cmd, check, stdout, stderr, text=False):
        data = {"streams": [{"index": 0, "tags": {"language": "jpn"}}, {"index": 1, "tags": {"language": "eng"}}]}
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(data))

    monkeypatch.setattr(gs.subprocess, "run", fake_run)
    assert gs.detect_audio_track(video, "eng") == 1


def test_detect_audio_track_fallback(gs, tmp_path, monkeypatch):
    video = tmp_path / "clip.mkv"

    def fake_run(cmd, check, stdout, stderr, text=False):
        data = {"streams": [{"index": 2}, {"index": 4}]}
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(data))

    monkeypatch.setattr(gs.subprocess, "run", fake_run)
    assert gs.detect_audio_track(video, None) == 2


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
    calls["diar"] = False

    def fake_load_diarize_model(device):
        calls["load_diarize_model"] = device

        def diar_fn(audio, segments):
            calls["diar"] = True
            return [{"start": 0.0, "end": 1.0, "speaker": "S1"}]

        return diar_fn

    monkeypatch.setattr(gs.whisperx, "load_align_model", fake_load_align_model)
    monkeypatch.setattr(gs.whisperx, "align", fake_align)
    monkeypatch.setattr(gs.whisperx.diarize, "load_diarize_model", fake_load_diarize_model)

    model = FakeModel()
    args = {
        "diarize": True,
        "vad_filter": False,
        "vad_onset": 0.5,
        "vad_offset": 0.5,
    }
    options = {
        "language": "en",
        "vad_filter": False,
        "vad_onset": 0.5,
        "vad_offset": 0.5,
    }
    segments, _ = gs.transcribe_file(
        audio_path,
        model,
        gs.torch.device("cpu"),
        args,
        options,
    )
    assert model.last_kwargs == {
        "language": "en",
        "vad_filter": False,
        "vad_onset": 0.5,
        "vad_offset": 0.5,
    }
    assert segments[0]["text"] == "aligned"
    assert segments[0]["speaker"] == "S1"
    assert calls["load_align_model"][1] == gs.torch.device("cpu")
    assert calls["align"] is True
    assert calls["load_diarize_model"] == gs.torch.device("cpu")
    assert calls["diar"] is True


def test_write_subtitles(gs, tmp_path):
    segments = [{"start": 0.0, "end": 1.0, "text": "Hello, WORLD!", "speaker": "S1"}]
    out = tmp_path / "out"
    result = gs.write_subtitles(segments, out, fmt="srt", case="lower", strip_punctuation=True)
    text = result.read_text(encoding="utf-8")
    assert "s1 hello world" in text


def test_write_subtitles_from_words(gs, tmp_path):
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "S1",
            "words": [
                {"start": 0.0, "word": "Hello"},
                {"start": 0.5, "word": "WORLD"},
            ],
        }
    ]
    out = tmp_path / "out"
    result = gs.write_subtitles(segments, out, fmt="srt", case="lower")
    text = result.read_text(encoding="utf-8")
    assert "s1: hello world" in text
