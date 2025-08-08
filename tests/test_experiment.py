import csv
import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules["whisperx"] = types.ModuleType("whisperx")
from experiment import SubtitleExperiment
sys.modules.pop("whisperx", None)
sys.modules.pop("transcribe", None)


def test_run_logging_and_aggregation(tmp_path, monkeypatch):
    cfg = {
        "run_id": "testrun",
        "inputs": ["audio.wav"],
        "output_root": str(tmp_path),
    }

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

    exp = SubtitleExperiment(cfg)
    run_dir = Path(cfg["output_root"]) / cfg["run_id"]
    cfg_path = run_dir / "config.json"
    assert not cfg_path.exists()

    exp.run()

    metrics_path = run_dir / "metrics.json"
    assert cfg_path.exists()
    assert metrics_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert metrics[0]["subtitle_count"] == 1
    assert metrics[0]["sync_offset"] == 0.2

    exp_csv = Path("experiments.csv")
    if exp_csv.exists():
        exp_csv.unlink()

    summary = exp.aggregate_results()
    assert summary["avg_subtitle_count"] == 1

    rows = list(csv.DictReader(exp_csv.open()))
    assert rows[0]["run_id"] == cfg["run_id"]
    loaded_cfg = json.loads(rows[0]["config"])
    assert loaded_cfg["run_id"] == cfg["run_id"]
    assert float(rows[0]["avg_subtitle_count"]) == 1

    exp_csv.unlink()
