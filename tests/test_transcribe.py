import importlib
import json
import sys
import types


# Create a stub whisperx module so ``transcribe`` can be imported without the
# heavy dependency.
stub = types.SimpleNamespace()
sys.modules.setdefault("whisperx", stub)
transcribe = importlib.import_module("transcribe")


def _setup_stub(align_func):
    """Configure ``stub`` with simple whisperx functionality."""

    class DummyModel:
        def transcribe(self, audio, batch_size, beam_size, language):  # noqa: D401
            return {
                "segments": [
                    {"start": 0.0, "end": 1.0, "text": "Hello"},
                    {"start": 1.0, "end": 2.0, "text": "World"},
                ]
            }

    stub.load_model = lambda *a, **k: DummyModel()
    stub.load_audio = lambda path: "audio"
    stub.load_align_model = lambda **k: ("align", "meta")
    stub.align = align_func


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

    def load_model(model, language, compute_type, batch_size):
        calls["load_model"] = {
            "language": language,
            "compute_type": compute_type,
            "batch_size": batch_size,
        }
        return DummyModel()

    def load_align_model(model_name, language_code):
        calls["load_align_model"] = {
            "model_name": model_name,
            "language_code": language_code,
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

    outpath = transcribe.transcribe_and_align(
        "dummy.wav",
        str(tmp_path),
        model="tiny",
        compute_type="float16",
        batch_size=4,
        beam_size=2,
    )

    assert calls["load_model"] == {
        "language": "en",
        "compute_type": "float16",
        "batch_size": 4,
    }
    assert calls["transcribe"] == {
        "batch_size": 4,
        "beam_size": 2,
        "language": "en",
    }
    assert calls["load_align_model"] == {
        "model_name": transcribe.ALIGN_MODEL_NAME,
        "language_code": "en",
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

    _setup_stub(align_func)

    outpath = transcribe.transcribe_and_align(
        "dummy.wav", str(tmp_path), music_segments=[(0.0, 1.0)], skip_music=False
    )
    data = json.loads(tmp_path.joinpath("transcript.json").read_text())
    simple = json.loads(tmp_path.joinpath("segments.json").read_text())

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

    _setup_stub(align_func)

    outpath = transcribe.transcribe_and_align(
        "dummy.wav", str(tmp_path), music_segments=[(0.0, 1.0)], skip_music=True
    )
    data = json.loads(tmp_path.joinpath("transcript.json").read_text())
    simple = json.loads(tmp_path.joinpath("segments.json").read_text())

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
        assert kwargs["music_segments"] == [[0.0, 1.0]]
        assert kwargs["spellcheck"] is False
        return str(tmp_path / "segments.json")

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


def test_cli_spellcheck_flag(tmp_path, monkeypatch):
    def fake_transcribe(audio_path, outdir, **kwargs):
        assert kwargs["spellcheck"] is True
        return str(tmp_path / "segments.json")

    monkeypatch.setattr(transcribe, "transcribe_and_align", fake_transcribe)

    argv = [
        "transcribe.py",
        "foo.wav",
        "--outdir",
        str(tmp_path),
        "--spellcheck",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    transcribe.main()

