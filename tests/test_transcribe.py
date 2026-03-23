import importlib
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules.pop("transcribe", None)
import transcribe


class DummyWord:
    def __init__(self, word, start, end):
        self.word = word
        self.start = start
        self.end = end


class DummySegment:
    def __init__(self, start, end, text, words=None):
        self.start = start
        self.end = end
        self.text = text
        self.words = words


def test_forwards_options_with_batched_pipeline(tmp_path, monkeypatch):
    calls = {}

    class FakeWhisperModel:
        def __init__(self, model, device, compute_type):
            calls["model_init"] = {
                "model": model,
                "device": device,
                "compute_type": compute_type,
            }

    class FakeBatchPipeline:
        def __init__(self, model):
            calls["pipeline_model"] = model

        def transcribe(self, audio_path, **kwargs):
            calls["transcribe"] = {"audio_path": audio_path, **kwargs}
            return (
                [
                    DummySegment(
                        0.0,
                        1.0,
                        " Hello ",
                        words=[DummyWord("Hello", 0.1, 0.9)],
                    )
                ],
                types.SimpleNamespace(language="en", language_probability=0.99),
            )

    monkeypatch.setattr(transcribe, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(transcribe, "BatchedInferencePipeline", FakeBatchPipeline)

    outputs = transcribe.transcribe_and_align(
        "dummy.wav",
        str(tmp_path),
        model="tiny",
        compute_type="float16",
        device="cuda",
        batch_size=4,
        beam_size=2,
    )

    assert calls["model_init"] == {
        "model": "tiny",
        "device": "cuda",
        "compute_type": "float16",
    }
    assert calls["transcribe"] == {
        "audio_path": "dummy.wav",
        "language": "en",
        "word_timestamps": True,
        "vad_filter": True,
        "beam_size": 2,
        "batch_size": 4,
    }
    assert outputs["segments_json"] == str(tmp_path / "segments.json")
    assert json.loads((tmp_path / "segments.json").read_text()) == [
        {
            "start": 0.1,
            "end": 0.9,
            "text": "Hello",
            "words": [{"word": "Hello", "start": 0.1, "end": 0.9}],
        }
    ]


def test_uses_model_directly_for_batch_size_one(tmp_path, monkeypatch):
    calls = {}

    class FakeWhisperModel:
        def __init__(self, model, device, compute_type):
            calls["init"] = (model, device, compute_type)

        def transcribe(self, audio_path, **kwargs):
            calls["transcribe"] = {"audio_path": audio_path, **kwargs}
            return (
                [DummySegment(0.0, 1.0, "Hello", words=None)],
                types.SimpleNamespace(language="en", language_probability=0.5),
            )

    class BoomBatchPipeline:
        def __init__(self, model):
            raise AssertionError("batched pipeline should not be used")

    monkeypatch.setattr(transcribe, "WhisperModel", FakeWhisperModel)
    monkeypatch.setattr(transcribe, "BatchedInferencePipeline", BoomBatchPipeline)

    outputs = transcribe.transcribe_and_align(
        "dummy.wav",
        str(tmp_path),
        device="cpu",
        batch_size=1,
        beam_size=None,
    )

    assert outputs["segments_json"] == str(tmp_path / "segments.json")
    assert calls["transcribe"] == {
        "audio_path": "dummy.wav",
        "language": "en",
        "word_timestamps": True,
        "vad_filter": True,
    }
    assert json.loads((tmp_path / "segments.json").read_text()) == [
        {"start": 0.0, "end": 1.0, "text": "Hello", "words": []}
    ]


def test_materializes_segment_generator(tmp_path, monkeypatch):
    consumed = {"value": False}

    class FakeWhisperModel:
        def __init__(self, model, device, compute_type):
            pass

        def transcribe(self, audio_path, **kwargs):
            def _segments():
                consumed["value"] = True
                yield DummySegment(0.0, 1.0, "Hello", [DummyWord("Hello", 0.0, 1.0)])

            return _segments(), types.SimpleNamespace(language="en", language_probability=1.0)

    monkeypatch.setattr(transcribe, "WhisperModel", FakeWhisperModel)

    transcribe.transcribe_and_align("dummy.wav", str(tmp_path), device="cpu", batch_size=1)

    assert consumed["value"] is True


def test_mark_music(tmp_path, monkeypatch):
    class FakeWhisperModel:
        def __init__(self, model, device, compute_type):
            pass

        def transcribe(self, audio_path, **kwargs):
            return (
                [
                    DummySegment(0.0, 1.0, "Intro", [DummyWord("Intro", 0.0, 1.0)]),
                    DummySegment(1.0, 2.0, "World", [DummyWord("World", 1.0, 2.0)]),
                ],
                types.SimpleNamespace(language="en", language_probability=1.0),
            )

    monkeypatch.setattr(transcribe, "WhisperModel", FakeWhisperModel)

    transcribe.transcribe_and_align(
        "dummy.wav", str(tmp_path), music_segments=[(0.0, 1.0)], skip_music=False, device="cpu", batch_size=1
    )

    transcript_data = json.loads((tmp_path / "transcript.json").read_text())
    simple_data = json.loads((tmp_path / "segments.json").read_text())

    assert transcript_data["segments"][0]["is_music"] is True
    assert "words" not in transcript_data["segments"][0]
    assert transcript_data["segments"][1]["words"][0]["word"] == "World"
    assert simple_data[0]["words"] == []
    assert simple_data[1]["words"][0]["word"] == "World"


def test_skip_music(tmp_path, monkeypatch):
    class FakeWhisperModel:
        def __init__(self, model, device, compute_type):
            pass

        def transcribe(self, audio_path, **kwargs):
            return (
                [
                    DummySegment(0.0, 1.0, "Intro", [DummyWord("Intro", 0.0, 1.0)]),
                    DummySegment(1.0, 2.0, "World", [DummyWord("World", 1.0, 2.0)]),
                ],
                types.SimpleNamespace(language="en", language_probability=1.0),
            )

    monkeypatch.setattr(transcribe, "WhisperModel", FakeWhisperModel)

    transcribe.transcribe_and_align(
        "dummy.wav", str(tmp_path), music_segments=[(0.0, 1.0)], skip_music=True, device="cpu", batch_size=1
    )

    transcript_data = json.loads((tmp_path / "transcript.json").read_text())
    simple_data = json.loads((tmp_path / "segments.json").read_text())

    assert transcript_data["segments"] == [
        {
            "start": 1.0,
            "end": 2.0,
            "text": "World",
            "is_music": False,
            "words": [{"word": "World", "start": 1.0, "end": 2.0}],
        }
    ]
    assert simple_data == [
        {
            "start": 1.0,
            "end": 2.0,
            "text": "World",
            "words": [{"word": "World", "start": 1.0, "end": 2.0}],
        }
    ]


def test_missing_words_fallback(tmp_path, monkeypatch):
    class FakeWhisperModel:
        def __init__(self, model, device, compute_type):
            pass

        def transcribe(self, audio_path, **kwargs):
            return (
                [DummySegment(0.0, 1.0, "Hello", words=None)],
                types.SimpleNamespace(language="en", language_probability=1.0),
            )

    monkeypatch.setattr(transcribe, "WhisperModel", FakeWhisperModel)

    transcribe.transcribe_and_align("dummy.wav", str(tmp_path), device="cpu", batch_size=1)

    assert json.loads((tmp_path / "transcript.json").read_text()) == {
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hello",
                "is_music": False,
                "words": [],
            }
        ]
    }


def test_resume_outputs_short_circuit(tmp_path, monkeypatch):
    transcript_path = tmp_path / "transcript.json"
    segments_path = tmp_path / "segments.json"
    transcript_path.write_text("{}")
    segments_path.write_text("[]")

    def boom(*args, **kwargs):
        raise AssertionError("backend should not be invoked")

    monkeypatch.setattr(transcribe, "_transcribe_segments", boom)

    outputs = transcribe.transcribe_and_align(
        "dummy.wav",
        str(tmp_path),
        resume_outputs={
            "transcript_json": str(transcript_path),
            "segments_json": str(segments_path),
        },
    )

    assert outputs == {
        "transcript_json": str(transcript_path),
        "segments_json": str(segments_path),
    }


def test_invalid_compute_type_for_cpu_raises(tmp_path):
    with pytest.raises(ValueError, match="not supported on cpu"):
        transcribe.transcribe_and_align(
            "dummy.wav",
            str(tmp_path),
            device="cpu",
            compute_type="float16",
        )


def test_default_device_uses_ctranslate2(monkeypatch):
    monkeypatch.setattr(transcribe.ctranslate2, "get_cuda_device_count", lambda: 0)
    assert transcribe._default_device() == "cpu"
    monkeypatch.setattr(transcribe.ctranslate2, "get_cuda_device_count", lambda: 2)
    assert transcribe._default_device() == "cuda"


def test_cli_main(tmp_path, monkeypatch, capsys, caplog):
    def fake_transcribe(audio_path, outdir, **kwargs):
        assert audio_path == "foo.wav"
        assert outdir == str(tmp_path)
        assert kwargs["model"] == "tiny"
        assert kwargs["batch_size"] == 4
        assert kwargs["beam_size"] == 2
        assert kwargs["compute_type"] == "float16"
        assert kwargs["device"] == "cpu"
        assert kwargs["music_segments"] == [[0.0, 1.0]]
        assert kwargs["spellcheck"] is False
        return {
            "segments_json": str(tmp_path / "segments.json"),
            "transcript_json": str(tmp_path / "transcript.json"),
        }

    monkeypatch.setattr(transcribe, "transcribe_and_align", fake_transcribe)

    music_file = tmp_path / "music.json"
    music_file.write_text("[[0.0, 1.0]]")

    argv = [
        "transcribe.py",
        "foo.wav",
        "--outdir",
        str(tmp_path),
        "--model",
        "tiny",
        "--batch-size",
        "4",
        "--beam-size",
        "2",
        "--compute-type",
        "float16",
        "--device",
        "cpu",
        "--music-segments",
        str(music_file),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    with caplog.at_level("INFO"):
        transcribe.main()
    captured = capsys.readouterr()

    assert str(tmp_path / "segments.json") in captured.out
    assert "Model: tiny" in caplog.text
    assert "Batch size: 4" in caplog.text
    assert "Device: cpu" in caplog.text


def test_cli_spellcheck_flag(tmp_path, monkeypatch):
    def fake_transcribe(audio_path, outdir, **kwargs):
        assert kwargs["spellcheck"] is True
        assert kwargs["device"] == "cpu"
        return {
            "segments_json": str(tmp_path / "segments.json"),
            "transcript_json": str(tmp_path / "transcript.json"),
        }

    monkeypatch.setattr(transcribe, "transcribe_and_align", fake_transcribe)

    argv = [
        "transcribe.py",
        "foo.wav",
        "--outdir",
        str(tmp_path),
        "--spellcheck",
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    transcribe.main()


def test_cli_stem_flag(tmp_path, monkeypatch, capsys):
    def fake_transcribe(audio_path, outdir, **kwargs):
        assert kwargs["stem"] == "MyEp"
        return {
            "segments_json": str(tmp_path / "MyEp.segments.json"),
            "transcript_json": str(tmp_path / "MyEp.transcript.json"),
        }

    monkeypatch.setattr(transcribe, "transcribe_and_align", fake_transcribe)

    argv = [
        "transcribe.py",
        "foo.wav",
        "--outdir",
        str(tmp_path),
        "--stem",
        "MyEp",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    transcribe.main()
    captured = capsys.readouterr()

    assert str(tmp_path / "MyEp.segments.json") in captured.out


def test_transcribe_imports_without_legacy_modules(monkeypatch):
    blocked = {"torch", "torchaudio", "whisperx", "pyannote", "pyannote.audio"}
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in blocked:
            raise ImportError(f"blocked import: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    for name in blocked:
        sys.modules.pop(name, None)
    sys.modules.pop("transcribe", None)
    module = importlib.import_module("transcribe")
    assert hasattr(module, "transcribe_and_align")
