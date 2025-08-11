import importlib
import json
import sys
import types
from pathlib import Path

torch_stub = types.SimpleNamespace(
    cuda=types.SimpleNamespace(is_available=lambda: False)
)
sys.modules.setdefault("torch", torch_stub)
import torch

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Create a stub whisperx module so ``transcribe`` can be imported without the
# heavy dependency.
stub = types.SimpleNamespace()
sys.modules.setdefault("whisperx", stub)
sys.modules.setdefault(
    "pysubs2", types.SimpleNamespace(load_from_whisper=lambda segments: None)
)
sys.modules.setdefault(
    "subtitle_pipeline", types.SimpleNamespace(spellcheck_lines=lambda subs: None)
)
transcribe = importlib.import_module("transcribe")


def _setup_stub(align_func):
    """Configure ``stub`` with simple whisperx functionality."""

    class DummyModel:
        def transcribe(self, audio, batch_size, language, beam_size=None):  # noqa: D401
            return {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Hello"},
                    {"start": 1.0, "end": 2.0, "text": "World"},
                ]
            }

    def load_model(model, device, language, compute_type):
        return DummyModel()

    calls = {}

    stub.load_model = load_model
    stub.load_audio = lambda path: "audio"

    def load_align_model(model_name, language_code, device):
        calls["load_align_model"] = {
            "model_name": model_name,
            "language_code": language_code,
            "device": device,
        }
        return ("align", "meta")

    stub.load_align_model = load_align_model
    stub.align = align_func

    return calls


def test_forwards_options(tmp_path):
    """``transcribe_and_align`` forwards core whisperx options."""

    calls = {}

    class DummyModel:
        def transcribe(self, audio, batch_size, beam_size, language):  # noqa: D401
            calls["transcribe"] = {
                "batch_size": batch_size,
                "beam_size": beam_size,
                "language": language,
            }
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}]}

    def load_model(model, device, language, compute_type):
        calls["load_model"] = {
            "device": device,
            "language": language,
            "compute_type": compute_type,
        }
        return DummyModel()

    def load_align_model(model_name, language_code, device):
        calls["load_align_model"] = {
            "model_name": model_name,
            "language_code": language_code,
            "device": device,
        }
        return ("align", "meta")

    def align(segs, align_model, metadata, audio, batch_size):
        calls["align"] = {"batch_size": batch_size}
        seg = segs[0].copy()
        seg["words"] = [{"start": 0.0, "end": 1.0, "word": "Hello"}]
        return {"segments": [seg]}

    stub.load_model = load_model
    stub.load_audio = lambda path: "audio"
    stub.load_align_model = load_align_model
    stub.align = align

    outputs = transcribe.transcribe_and_align(
        "dummy.wav",
        str(tmp_path),
        model="tiny",
        compute_type="float16",
        batch_size=4,
        beam_size=2,
    )
    outpath = outputs["segments_json"]

    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert calls["load_model"] == {
        "device": expected_device,
        "language": "en",
        "compute_type": "float16",
    }
    assert calls["transcribe"] == {
        "batch_size": 4,
        "beam_size": 2,
        "language": "en",
    }
    assert calls["load_align_model"] == {
        "model_name": transcribe.ALIGN_MODEL_NAME,
        "language_code": "en",
        "device": expected_device,
    }
    assert calls["align"] == {"batch_size": 4}
    assert json.loads(tmp_path.joinpath("segments.json").read_text())[0]["words"][0]["word"] == "Hello"
    assert outpath == str(tmp_path / "segments.json")


def test_mark_music(tmp_path):
    """Segments overlapping music are marked when not skipped."""

    def align_func(segs, align_model, metadata, audio, batch_size):
        assert len(segs) == 1 and segs[0]["text"] == "World"
        aligned = segs[0].copy()
        aligned["words"] = [
            {"start": 1.0, "end": 2.0, "word": "World"}
        ]
        return {"segments": [aligned]}

    calls = _setup_stub(align_func)

    outputs = transcribe.transcribe_and_align(
        "dummy.wav", str(tmp_path), music_segments=[(0.0, 1.0)], skip_music=False
    )
    data = json.loads(tmp_path.joinpath("transcript.json").read_text())
    simple = json.loads(tmp_path.joinpath("segments.json").read_text())

    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert calls["load_align_model"] == {
        "model_name": transcribe.ALIGN_MODEL_NAME,
        "language_code": "en",
        "device": expected_device,
    }

    assert len(data["segments"]) == 2
    assert data["segments"][0]["is_music"] is True
    assert "words" not in data["segments"][0]
    assert data["segments"][1]["is_music"] is False
    assert data["segments"][1]["words"][0]["word"] == "World"

    assert len(simple) == 2
    assert simple[0]["words"] == []
    assert simple[1]["words"][0]["word"] == "World"


def test_skip_music(tmp_path):
    """Music segments are removed when ``skip_music`` is True."""

    def align_func(segs, align_model, metadata, audio, batch_size):
        assert len(segs) == 1 and segs[0]["text"] == "World"
        aligned = segs[0].copy()
        aligned["words"] = [
            {"start": 1.0, "end": 2.0, "word": "World"}
        ]
        return {"segments": [aligned]}

    calls = _setup_stub(align_func)

    outputs = transcribe.transcribe_and_align(
        "dummy.wav", str(tmp_path), music_segments=[(0.0, 1.0)], skip_music=True
    )
    data = json.loads(tmp_path.joinpath("transcript.json").read_text())
    simple = json.loads(tmp_path.joinpath("segments.json").read_text())

    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert calls["load_align_model"] == {
        "model_name": transcribe.ALIGN_MODEL_NAME,
        "language_code": "en",
        "device": expected_device,
    }

    assert len(data["segments"]) == 1
    assert data["segments"][0]["is_music"] is False
    assert data["segments"][0]["text"] == "World"

    assert len(simple) == 1
    assert simple[0]["text"] == "World"


def test_cli_main(tmp_path, monkeypatch, capsys, caplog):
    """Command-line interface passes arguments and prints the output path."""

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


def test_cli_main_without_beam_size(tmp_path, monkeypatch, capsys):
    """CLI defaults to ``beam_size=None`` when not provided."""

    def fake_transcribe(audio_path, outdir, **kwargs):
        assert kwargs["beam_size"] is None
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
        "--device",
        "cpu",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    transcribe.main()
    captured = capsys.readouterr()

    assert str(tmp_path / "segments.json") in captured.out


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


def test_transcribe_without_beam_size(tmp_path, caplog):
    """``transcribe_and_align`` handles models without ``beam_size``."""

    calls = {}

    class DummyModel:
        def transcribe(self, audio, batch_size, language):
            calls["transcribe"] = {"batch_size": batch_size, "language": language}
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}]}

    stub.load_model = lambda model, device, language, compute_type: DummyModel()
    stub.load_audio = lambda path: "audio"

    def load_align_model(model_name, language_code, device):
        calls["load_align_model"] = {
            "model_name": model_name,
            "language_code": language_code,
            "device": device,
        }
        return ("align", "meta")

    stub.load_align_model = load_align_model
    stub.align = lambda segs, align_model, metadata, audio, batch_size: {"segments": segs}

    with caplog.at_level("INFO"):
        outputs = transcribe.transcribe_and_align("dummy.wav", str(tmp_path), beam_size=2)
        outpath = outputs["segments_json"]

    expected_device = "cuda" if torch.cuda.is_available() else "cpu"

    assert outpath == str(tmp_path / "segments.json")
    assert calls["transcribe"] == {"batch_size": 8, "language": "en"}
    assert calls["load_align_model"] == {
        "model_name": transcribe.ALIGN_MODEL_NAME,
        "language_code": "en",
        "device": expected_device,
    }
    assert json.loads(tmp_path.joinpath("segments.json").read_text()) == [
        {"start": 0.0, "end": 1.0, "text": "Hello", "words": []}
    ]
    assert "beam_size" not in caplog.text


def test_transcribe_default_beam_size(tmp_path):
    """When ``beam_size`` is ``None``, the model's default is used."""

    calls = {}

    class DummyModel:
        def transcribe(self, audio, batch_size, language, beam_size=None):
            calls["transcribe"] = {
                "batch_size": batch_size,
                "beam_size": beam_size,
                "language": language,
            }
            return {"segments": [{"start": 0.0, "end": 1.0, "text": "Hello"}]}

    stub.load_model = lambda model, device, language, compute_type: DummyModel()
    stub.load_audio = lambda path: "audio"
    stub.load_align_model = lambda model_name, language_code, device: ("align", "meta")
    stub.align = lambda segs, align_model, metadata, audio, batch_size: {"segments": segs}

    outputs = transcribe.transcribe_and_align("dummy.wav", str(tmp_path))
    outpath = outputs["segments_json"]

    assert outpath == str(tmp_path / "segments.json")
    assert calls["transcribe"] == {"batch_size": 8, "beam_size": None, "language": "en"}


def test_transcribe_with_stem(tmp_path):
    """Output filenames change when ``stem`` is provided."""

    def align_func(segs, align_model, metadata, audio, batch_size):
        aligned = []
        for s in segs:
            seg = s.copy()
            seg["words"] = [
                {"start": s["start"], "end": s["end"], "word": s["text"]}
            ]
            aligned.append(seg)
        return {"segments": aligned}

    _setup_stub(align_func)

    outputs = transcribe.transcribe_and_align(
        "dummy.wav", str(tmp_path), stem="MyEp"
    )

    assert outputs["segments_json"] == str(tmp_path / "MyEp.segments.json")
    assert json.loads(tmp_path.joinpath("MyEp.transcript.json").read_text())
    assert json.loads(tmp_path.joinpath("MyEp.segments.json").read_text())


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

