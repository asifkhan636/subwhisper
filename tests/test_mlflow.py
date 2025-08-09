import builtins
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.modules["whisperx"] = types.ModuleType("whisperx")
sys.modules["torch"] = types.ModuleType("torch")
sys.modules["noisereduce"] = types.ModuleType("noisereduce")
from experiment import SubtitleExperiment
sys.modules.pop("whisperx", None)
sys.modules.pop("torch", None)
sys.modules.pop("noisereduce", None)
sys.modules.pop("transcribe", None)


def _setup_pipeline(monkeypatch, tmp_path: Path) -> None:
    """Stub heavy dependencies used by ``SubtitleExperiment``."""

    def fake_preprocess(src, workdir, **kwargs):
        return src, []

    def fake_transcribe(audio_path, out_dir, **kwargs):
        return str(tmp_path / "segments.json")

    class DummySubs:
        def __init__(self):
            self.events = []

    def fake_load_segments(path):
        return DummySubs()

    def fake_enforce(subs, *args, **kwargs):
        pass

    def fake_write_outputs(subs, srt_path, _):
        Path(srt_path).write_text("dummy", encoding="utf-8")

    def fake_collect_metrics(path):
        return {"subtitle_count": 1}

    def fake_validate_sync(path, audio):
        return {"offset": 0.2}

    monkeypatch.setattr("experiment.preprocess_pipeline", fake_preprocess)
    monkeypatch.setattr("experiment.transcribe_and_align", fake_transcribe)
    monkeypatch.setattr("experiment.load_segments", fake_load_segments)
    monkeypatch.setattr("experiment.enforce_limits", fake_enforce)
    monkeypatch.setattr("experiment.write_outputs", fake_write_outputs)
    monkeypatch.setattr("experiment.qc.collect_metrics", fake_collect_metrics)
    monkeypatch.setattr("experiment.qc.validate_sync", fake_validate_sync)


def test_mlflow_missing_dependency(tmp_path, monkeypatch):
    cfg = {
        "run_id": "mlflowfail",
        "inputs": [],
        "output_root": str(tmp_path),
        "mlflow": {"tracking_uri": "file:/tmp", "experiment_name": "exp"},
    }

    exp = SubtitleExperiment(cfg)

    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("No module named mlflow")
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="MLflow logging requested"):
        exp.run()


def test_mlflow_logging(tmp_path, monkeypatch):
    cfg = {
        "run_id": "mlflowrun",
        "inputs": ["audio.wav"],
        "output_root": str(tmp_path),
        "mlflow": {"tracking_uri": "file:/tmp", "experiment_name": "exp"},
    }

    _setup_pipeline(monkeypatch, tmp_path)

    ml = types.SimpleNamespace()
    ml.tracking_uri = None
    ml.experiment_name = None
    ml.params = {}
    ml.metrics = []
    ml.run_name = None
    ml.ended = False

    def set_tracking_uri(uri):
        ml.tracking_uri = uri

    def set_experiment(name):
        ml.experiment_name = name

    def start_run(run_name=None):
        ml.run_name = run_name

    def log_params(params):
        ml.params = params

    def log_metrics(metrics, step=None):
        ml.metrics.append((metrics, step))

    def end_run():
        ml.ended = True

    ml.set_tracking_uri = set_tracking_uri
    ml.set_experiment = set_experiment
    ml.start_run = start_run
    ml.log_params = log_params
    ml.log_metrics = log_metrics
    ml.end_run = end_run

    monkeypatch.setitem(sys.modules, "mlflow", ml)

    exp = SubtitleExperiment(cfg)
    exp.run()

    assert ml.tracking_uri == "file:/tmp"
    assert ml.experiment_name == "exp"
    assert ml.run_name == cfg["run_id"]
    assert ml.params["run_id"] == cfg["run_id"]
    assert ml.metrics[0][0]["subtitle_count"] == 1
    assert ml.metrics[0][1] == 1
    assert ml.ended
