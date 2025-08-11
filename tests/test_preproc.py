import json
import sys
import types
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import pathlib
import sys
import types

# Provide stub modules so ``preproc`` imports without heavy dependencies.
librosa_stub = types.ModuleType("librosa")
librosa_stub.load = lambda *a, **k: None
librosa_stub.stream = lambda *a, **k: iter([])
librosa_stub.effects = types.SimpleNamespace(hpss=lambda y: ([], []))
librosa_stub.feature = types.SimpleNamespace(rms=lambda *a, **k: np.array([[0]]))
librosa_stub.frames_to_time = lambda idx, sr, hop_length: float(idx)
sys.modules.setdefault("librosa", librosa_stub)

sf_stub = types.ModuleType("soundfile")
sf_stub.read = lambda *a, **k: ([], 16000)
sf_stub.info = lambda *a, **k: types.SimpleNamespace(samplerate=16000)
sf_stub.write = lambda *a, **k: None
sys.modules.setdefault("soundfile", sf_stub)

nr_stub = types.ModuleType("noisereduce")
nr_stub.reduce_noise = lambda *a, **k: None
sys.modules.setdefault("noisereduce", nr_stub)

# Ensure repository root on path for importing ``preproc``.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import preproc


def test_find_english_track_selects_english_stream(monkeypatch):
    # Pretend ffprobe is available
    monkeypatch.setattr(preproc.shutil, "which", lambda cmd: "/usr/bin/" + cmd)

    # Mock ffprobe output with an English track at position 1
    ffprobe_output = json.dumps(
        {
            "streams": [
                {"tags": {"language": "deu"}},
                {"tags": {"language": "eng"}},
                {"tags": {"language": "fra"}},
            ]
        }
    )
    run_mock = MagicMock(return_value=SimpleNamespace(stdout=ffprobe_output))
    monkeypatch.setattr(preproc.subprocess, "run", run_mock)

    index = preproc.find_english_track("dummy.mp4")

    assert index == 1
    run_mock.assert_called_once()


def test_denoise_audio_passes_aggressiveness(monkeypatch, tmp_path):
    # Dummy audio data
    data = np.array([0.1, 0.2])
    rate = 16000

    # Mock soundfile read/write
    monkeypatch.setattr(preproc.sf, "read", lambda path: (data, rate))
    write_mock = MagicMock()
    monkeypatch.setattr(preproc.sf, "write", write_mock)

    # Mock noise reduction
    reduce_mock = MagicMock(return_value=data)
    monkeypatch.setattr(preproc.nr, "reduce_noise", reduce_mock)

    input_wav = tmp_path / "in.wav"
    output_wav = tmp_path / "out.wav"

    result = preproc.denoise_audio(str(input_wav), str(output_wav), aggressiveness=0.9)

    assert result == str(output_wav)
    reduce_mock.assert_called_once()
    assert reduce_mock.call_args.kwargs.get("prop_decrease") == 0.9
    write_mock.assert_called_once_with(str(output_wav), data, rate)


def test_preprocess_pipeline_forwards_aggressiveness(monkeypatch, tmp_path):
    monkeypatch.setattr(preproc.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(preproc.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(preproc, "find_english_track", lambda p: 0)

    extract_mock = MagicMock(return_value=str(tmp_path / "audio.wav"))
    monkeypatch.setattr(preproc, "extract_audio", extract_mock)

    denoise_mock = MagicMock(return_value=str(tmp_path / "den.wav"))
    monkeypatch.setattr(preproc, "denoise_audio", denoise_mock)

    detect_mock = MagicMock(return_value=[])
    monkeypatch.setattr(preproc, "detect_music_segments", detect_mock)

    preproc.preprocess_pipeline(
        input_path="in.mp4",
        outdir=str(tmp_path),
        denoise=True,
        denoise_aggressiveness=0.9,
    )

    assert denoise_mock.call_args.kwargs["aggressiveness"] == 0.9
    detect_mock.assert_called_once()
    assert detect_mock.call_args.args[1] == str(tmp_path / "music_segments.json")


def test_preprocess_pipeline_stem_builds_filenames(monkeypatch, tmp_path):
    monkeypatch.setattr(preproc.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(preproc.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(preproc, "find_english_track", lambda p: 0)

    def fake_extract(src, dst, track):
        pathlib.Path(dst).touch()
        return dst

    def fake_denoise(src, dst, aggressiveness):
        pathlib.Path(dst).touch()
        return dst

    def fake_normalize(src, dst, enabled=True):
        pathlib.Path(dst).touch()
        return dst

    def fake_detect(audio, seg_file, **kwargs):
        pathlib.Path(seg_file).touch()
        return []

    monkeypatch.setattr(preproc, "extract_audio", fake_extract)
    monkeypatch.setattr(preproc, "denoise_audio", fake_denoise)
    monkeypatch.setattr(preproc, "normalize_audio", fake_normalize)
    monkeypatch.setattr(preproc, "detect_music_segments", fake_detect)

    preproc.preprocess_pipeline(
        input_path="in.mp4",
        outdir=str(tmp_path),
        denoise=True,
        normalize=True,
        stem="MyEp",
    )

    assert (tmp_path / "MyEp.audio.wav").is_file()
    assert (tmp_path / "MyEp.denoised.wav").is_file()
    assert (tmp_path / "MyEp.normalized.wav").is_file()
    assert (tmp_path / "MyEp.music_segments.json").is_file()


def test_preprocess_pipeline_handles_detection_failure(monkeypatch, tmp_path):
    """A failure in music detection should not abort preprocessing."""
    monkeypatch.setattr(preproc.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(preproc.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(preproc, "find_english_track", lambda p: 0)

    def fake_extract(src, dst, track):
        pathlib.Path(dst).touch()
        return dst

    def fake_normalize(src, dst, enabled=True):
        pathlib.Path(dst).touch()
        return dst

    monkeypatch.setattr(preproc, "extract_audio", fake_extract)
    monkeypatch.setattr(preproc, "normalize_audio", fake_normalize)

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(preproc, "detect_music_segments", boom)

    out = preproc.preprocess_pipeline(input_path="in.mp4", outdir=str(tmp_path))

    assert out["music_segments"] is None


def test_normalize_audio_copy_when_disabled(monkeypatch, tmp_path):
    copy_mock = MagicMock()
    monkeypatch.setattr(preproc.shutil, "copyfile", copy_mock)

    src = tmp_path / "src.wav"
    dst = tmp_path / "dst.wav"

    result = preproc.normalize_audio(str(src), str(dst), enabled=False)

    assert result == str(dst)
    copy_mock.assert_called_once_with(str(src), str(dst))


def test_normalize_audio_invokes_ffmpeg_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setattr(preproc.shutil, "which", lambda cmd: "/usr/bin/" + cmd)
    run_mock = MagicMock(return_value=SimpleNamespace())
    monkeypatch.setattr(preproc.subprocess, "run", run_mock)

    src = tmp_path / "src.wav"
    dst = tmp_path / "dst.wav"

    result = preproc.normalize_audio(str(src), str(dst), enabled=True)

    assert result == str(dst)
    run_mock.assert_called_once()


@pytest.mark.parametrize(
    "threshold,expected",
    [
        (0.5, [(2.0, 4.0)]),
        (0.7, [(3.0, 4.0)]),
    ],
)
def test_detect_music_segments_threshold(monkeypatch, tmp_path, threshold, expected):
    monkeypatch.setattr(preproc.sf, "info", lambda p: SimpleNamespace(samplerate=22050))

    def stream_stub(path, block_length, frame_length, hop_length, mono):
        yield np.zeros(4)

    monkeypatch.setattr(preproc.librosa, "stream", stream_stub)
    monkeypatch.setattr(
        preproc.librosa.effects,
        "hpss",
        lambda y: (np.zeros(4), np.zeros(4)),
    )
    harm_rms = np.array([[1.0, 1.0, 1.0, 1.0]])
    perc_rms = np.array([[0.1, 0.6, 0.4, 0.8]])
    rms_mock = MagicMock(side_effect=[harm_rms, perc_rms])
    monkeypatch.setattr(preproc.librosa.feature, "rms", rms_mock)
    monkeypatch.setattr(
        preproc.librosa, "frames_to_time", lambda idx, sr, hop_length: float(idx)
    )

    seg_file = tmp_path / "music_segments.json"
    segments = preproc.detect_music_segments("dummy.wav", str(seg_file), threshold)

    assert segments == expected
    # Ensure JSON file was written
    assert seg_file.is_file()


def test_detect_music_segments_merges_adjacent(monkeypatch, tmp_path):
    monkeypatch.setattr(preproc.sf, "info", lambda p: SimpleNamespace(samplerate=22050))

    def stream_stub(path, block_length, frame_length, hop_length, mono):
        yield np.zeros(5)

    monkeypatch.setattr(preproc.librosa, "stream", stream_stub)
    monkeypatch.setattr(
        preproc.librosa.effects,
        "hpss",
        lambda y: (np.zeros(5), np.zeros(5)),
    )
    harm_rms = np.array([[1, 1, 1, 1, 1]])
    perc_rms = np.array([[1, 1, 0, 1, 1]])
    rms_mock = MagicMock(side_effect=[harm_rms, perc_rms])
    monkeypatch.setattr(preproc.librosa.feature, "rms", rms_mock)
    monkeypatch.setattr(
        preproc.librosa,
        "frames_to_time",
        lambda idx, sr, hop_length: float(idx // 2),
    )

    seg_file = tmp_path / "music_segments.json"
    segments = preproc.detect_music_segments("dummy.wav", str(seg_file), threshold=0.6)

    assert segments == [(0.0, 2.0)]


def test_detect_music_segments_drops_short(monkeypatch, tmp_path):
    monkeypatch.setattr(preproc.sf, "info", lambda p: SimpleNamespace(samplerate=22050))

    def stream_stub(path, block_length, frame_length, hop_length, mono):
        yield np.zeros(7)

    monkeypatch.setattr(preproc.librosa, "stream", stream_stub)
    monkeypatch.setattr(
        preproc.librosa.effects,
        "hpss",
        lambda y: (np.zeros(7), np.zeros(7)),
    )
    harm_rms = np.array([[1, 1, 1, 1, 0, 1, 0]])
    perc_rms = np.array([[1, 1, 1, 1, 0, 1, 0]])
    rms_mock = MagicMock(side_effect=[harm_rms, perc_rms])
    monkeypatch.setattr(preproc.librosa.feature, "rms", rms_mock)
    monkeypatch.setattr(
        preproc.librosa,
        "frames_to_time",
        lambda idx, sr, hop_length: idx * 0.5,
    )

    seg_file = tmp_path / "music_segments.json"
    segments = preproc.detect_music_segments(
        "dummy.wav", str(seg_file), threshold=0.6, min_duration=1.0
    )

    assert segments == [(0.0, 2.5)]


def test_detect_music_segments_smooths_flips(monkeypatch, tmp_path):
    """Brief non-music flips inside a music region should be removed."""
    monkeypatch.setattr(preproc.sf, "info", lambda p: SimpleNamespace(samplerate=22050))

    def stream_stub(path, block_length, frame_length, hop_length, mono):
        yield np.zeros(3)

    monkeypatch.setattr(preproc.librosa, "stream", stream_stub)
    monkeypatch.setattr(
        preproc.librosa.effects,
        "hpss",
        lambda y: (np.zeros(3), np.zeros(3)),
    )
    harm_rms = np.array([[1, 1, 1]])
    perc_rms = np.array([[1, 0, 1]])
    rms_mock = MagicMock(side_effect=[harm_rms, perc_rms])
    monkeypatch.setattr(preproc.librosa.feature, "rms", rms_mock)
    monkeypatch.setattr(
        preproc.librosa, "frames_to_time", lambda idx, sr, hop_length: float(idx)
    )

    seg_file = tmp_path / "music_segments.json"
    segments = preproc.detect_music_segments("dummy.wav", str(seg_file), threshold=0.6)

    assert segments == [(0.0, 3.0)]


def test_detect_music_segments_merges_small_gaps(monkeypatch, tmp_path):
    """Segments separated by short gaps should be merged when min_gap is set."""
    monkeypatch.setattr(preproc.sf, "info", lambda p: SimpleNamespace(samplerate=22050))

    def stream_stub(path, block_length, frame_length, hop_length, mono):
        yield np.zeros(7)

    monkeypatch.setattr(preproc.librosa, "stream", stream_stub)
    monkeypatch.setattr(
        preproc.librosa.effects,
        "hpss",
        lambda y: (np.zeros(7), np.zeros(7)),
    )
    harm_rms = np.array([[1, 1, 1, 1, 1, 1, 1]])
    perc_rms = np.array([[1, 1, 0, 0, 0, 1, 1]])
    rms_mock = MagicMock(side_effect=[harm_rms, perc_rms])
    monkeypatch.setattr(preproc.librosa.feature, "rms", rms_mock)
    monkeypatch.setattr(
        preproc.librosa, "frames_to_time", lambda idx, sr, hop_length: idx * 0.5
    )

    seg_file = tmp_path / "music_segments.json"
    segments = preproc.detect_music_segments(
        "dummy.wav", str(seg_file), threshold=0.6, min_gap=2.0
    )

    assert segments == [(0.0, 3.5)]


def test_detect_music_segments_warns_on_many_segments(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(preproc.sf, "info", lambda p: SimpleNamespace(samplerate=22050))

    def stream_stub(path, block_length, frame_length, hop_length, mono):
        yield np.zeros(4)

    monkeypatch.setattr(preproc.librosa, "stream", stream_stub)
    monkeypatch.setattr(
        preproc.librosa.effects,
        "hpss",
        lambda y: (np.zeros(4), np.zeros(4)),
    )
    harm_rms = np.array([[1, 1, 1, 1]])
    perc_rms = np.array([[1, 0, 1, 0]])
    rms_mock = MagicMock(side_effect=[harm_rms, perc_rms])
    monkeypatch.setattr(preproc.librosa.feature, "rms", rms_mock)
    monkeypatch.setattr(
        preproc.librosa, "frames_to_time", lambda idx, sr, hop_length: float(idx)
    )

    seg_file = tmp_path / "music_segments.json"
    with caplog.at_level(logging.WARNING, logger=preproc.logger.name):
        preproc.detect_music_segments(
            "dummy.wav", str(seg_file), threshold=0.6, count_warning=0
        )

    assert any("exceeds warning threshold" in r.message for r in caplog.records)


def test_detect_music_segments_vad_suppresses(monkeypatch, tmp_path):
    monkeypatch.setattr(preproc.sf, "info", lambda p: SimpleNamespace(samplerate=22050))

    def stream_stub(path, block_length, frame_length, hop_length, mono):
        yield np.zeros(4)

    monkeypatch.setattr(preproc.librosa, "stream", stream_stub)
    monkeypatch.setattr(
        preproc.librosa.effects,
        "hpss",
        lambda y: (np.zeros(4), np.zeros(4)),
    )
    harm_rms = np.array([[1.0, 1.0, 1.0, 1.0]])
    perc_rms = np.array([[1.0, 1.0, 1.0, 1.0]])
    rms_mock = MagicMock(side_effect=[harm_rms, perc_rms])
    monkeypatch.setattr(preproc.librosa.feature, "rms", rms_mock)
    monkeypatch.setattr(
        preproc.librosa.feature,
        "spectral_centroid",
        lambda y, sr, hop_length: np.array([[0.1, 0.1, 0.1, 0.1]]),
    )
    monkeypatch.setattr(
        preproc.librosa.onset,
        "onset_strength",
        lambda y, sr, hop_length: np.array([0.1, 0.1, 0.1, 0.1]),
    )
    monkeypatch.setattr(
        preproc.librosa, "frames_to_time", lambda idx, sr, hop_length: float(idx)
    )
    monkeypatch.setattr(
        preproc.librosa, "time_to_frames", lambda t, sr, hop_length: int(t)
    )

    class DummyVAD:
        def __call__(self, inp):
            return [SimpleNamespace(start=1.0, end=3.0, confidence=1.0)]

    monkeypatch.setattr(preproc.vad, "load_vad_model", lambda: DummyVAD())

    seg_file = tmp_path / "music_segments.json"
    segments = preproc.detect_music_segments(
        "dummy.wav", str(seg_file), threshold=0.6, enhanced=True
    )

    assert segments == [(0.0, 1.0), (3.0, 4.0)]


def test_detect_music_segments_streams_large_files(monkeypatch, tmp_path):
    """Processing should iterate over multiple chunks without loading whole file."""
    monkeypatch.setattr(preproc.sf, "info", lambda p: SimpleNamespace(samplerate=22050))

    def stream_stub(path, block_length, frame_length, hop_length, mono):
        yield np.zeros(3)
        yield np.zeros(3)

    monkeypatch.setattr(preproc.librosa, "stream", stream_stub)
    monkeypatch.setattr(
        preproc.librosa.effects,
        "hpss",
        lambda y: (np.zeros_like(y), np.zeros_like(y)),
    )
    harm1 = np.array([[1, 1, 1]])
    perc1 = np.array([[1, 1, 1]])
    harm2 = np.array([[1, 1, 1]])
    perc2 = np.array([[1, 1, 1]])
    rms_mock = MagicMock(side_effect=[harm1, perc1, harm2, perc2])
    monkeypatch.setattr(preproc.librosa.feature, "rms", rms_mock)
    monkeypatch.setattr(
        preproc.librosa, "frames_to_time", lambda idx, sr, hop_length: float(idx)
    )

    seg_file = tmp_path / "music_segments.json"
    segments = preproc.detect_music_segments("dummy.wav", str(seg_file), threshold=0.6)

    assert segments == [(0.0, 6.0)]
    assert rms_mock.call_count == 4
