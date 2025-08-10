import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock


# Ensure repository root on path for importing ``run_state``.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_state import (
    is_stage_reusable,
    stage_complete,
    load_manifest,
    CACHE_DIR,
)


def run_dummy_pipeline(root, preproc_level=0.5, preproc_op=None, transcribe_op=None):
    """Run a minimal two-stage pipeline using the cache helpers.

    Heavy operations like audio extraction or WhisperX transcription are
    replaced by the callables ``preproc_op`` and ``transcribe_op`` which only
    write dummy output files. This keeps tests fast while still exercising the
    resume logic.
    """
    root = pathlib.Path(root)
    media = root / "input.mp4"
    if not media.exists():
        media.write_text("media")
    stem = media.stem

    work = root / CACHE_DIR / stem
    work.mkdir(parents=True, exist_ok=True)

    if preproc_op is None:
        preproc_op = lambda out: out.write_text("audio")
    if transcribe_op is None:
        transcribe_op = lambda out: out.write_text("{}")

    downstream_valid = True

    # Stage: preprocess
    preproc_params = {"level": preproc_level}
    reusable, outputs = is_stage_reusable(root, stem, "preproc", str(media), preproc_params)
    if reusable:
        audio = pathlib.Path(outputs[0])
    else:
        audio = work / "audio.wav"
        preproc_op(audio)
        stage_complete(root, stem, "preproc", str(media), preproc_params, [str(audio)])
    preproc_ran = not reusable
    downstream_valid = downstream_valid and reusable

    # Stage: transcribe
    trans_params = {"model": "dummy"}
    reusable, outputs = (
        is_stage_reusable(root, stem, "transcribe", str(media), trans_params)
        if downstream_valid
        else (False, [])
    )
    if reusable:
        transcript = pathlib.Path(outputs[0])
    else:
        transcript = work / "transcript.json"
        transcribe_op(transcript)
        stage_complete(root, stem, "transcribe", str(media), trans_params, [str(transcript)])
    trans_ran = not reusable

    return SimpleNamespace(
        audio=audio,
        transcript=transcript,
        preproc_ran=preproc_ran,
        trans_ran=trans_ran,
    )


def test_initial_run_writes_manifest_and_outputs(tmp_path):
    preproc_mock = MagicMock(side_effect=lambda p: p.write_text("audio"))
    trans_mock = MagicMock(side_effect=lambda p: p.write_text("{}"))

    result = run_dummy_pipeline(tmp_path, preproc_op=preproc_mock, transcribe_op=trans_mock)

    assert result.preproc_ran and result.trans_ran
    preproc_mock.assert_called_once()
    trans_mock.assert_called_once()
    assert result.audio.exists() and result.transcript.exists()

    manifest = load_manifest(tmp_path, "input")
    assert "preproc" in manifest.get("stages", {})
    assert "transcribe" in manifest.get("stages", {})
    assert manifest["stages"]["preproc"]["outputs"][0] == str(result.audio.resolve())


def test_second_run_skips_stages(tmp_path):
    run_dummy_pipeline(tmp_path)  # initial run to populate manifest

    preproc_mock = MagicMock(side_effect=lambda p: p.write_text("audio"))
    trans_mock = MagicMock(side_effect=lambda p: p.write_text("{}"))

    result = run_dummy_pipeline(tmp_path, preproc_op=preproc_mock, transcribe_op=trans_mock)

    assert not result.preproc_ran and not result.trans_ran
    preproc_mock.assert_not_called()
    trans_mock.assert_not_called()


def test_param_change_recomputes_downstream(tmp_path):
    run_dummy_pipeline(tmp_path, preproc_level=0.5)

    preproc_mock = MagicMock(side_effect=lambda p: p.write_text("audio"))
    trans_mock = MagicMock(side_effect=lambda p: p.write_text("{}"))

    result = run_dummy_pipeline(
        tmp_path, preproc_level=0.9, preproc_op=preproc_mock, transcribe_op=trans_mock
    )

    assert result.preproc_ran and result.trans_ran
    preproc_mock.assert_called_once()
    trans_mock.assert_called_once()


def test_missing_output_triggers_recompute(tmp_path):
    first = run_dummy_pipeline(tmp_path)
    first.transcript.unlink()  # simulate external deletion

    preproc_mock = MagicMock(side_effect=lambda p: p.write_text("audio"))
    trans_mock = MagicMock(side_effect=lambda p: p.write_text("{}"))

    result = run_dummy_pipeline(tmp_path, preproc_op=preproc_mock, transcribe_op=trans_mock)

    assert not result.preproc_ran and result.trans_ran
    preproc_mock.assert_not_called()
    trans_mock.assert_called_once()
