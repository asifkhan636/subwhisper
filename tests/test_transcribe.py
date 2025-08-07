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

    assert len(data["segments"]) == 2
    assert data["segments"][0]["is_music"] is True
    assert "words" not in data["segments"][0]
    assert data["segments"][1]["is_music"] is False
    assert data["segments"][1]["words"][0]["word"] == "World"


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

    assert len(data["segments"]) == 1
    assert data["segments"][0]["is_music"] is False
    assert data["segments"][0]["text"] == "World"

