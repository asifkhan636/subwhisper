import csv
import json
from pathlib import Path
import sys
import types
from typing import Any, Dict
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.modules["whisperx"] = types.ModuleType("whisperx")
from experiment import SubtitleExperiment
import experiment_runner
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

    called: Dict[str, Any] = {}

    def fake_enforce(subs, **kwargs):
        called.update(kwargs)

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
    cfg_path = run_dir / f"config_{cfg['run_id']}.json"
    commit_path = run_dir / f"commit_{cfg['run_id']}.txt"
    reqs_path = run_dir / "requirements.txt"
    assert commit_path.exists()
    assert reqs_path.exists()
    assert not cfg_path.exists()

    exp.run()

    assert called == {
        "max_chars": 45,
        "max_lines": 2,
        "max_duration": 6.0,
        "min_gap": 0.15,
    }

    metrics_path = run_dir / f"metrics_{cfg['run_id']}.json"
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
    commit_val = commit_path.read_text().strip()
    assert loaded_cfg["git_commit"] == commit_val
    assert float(rows[0]["avg_subtitle_count"]) == 1

    summary_md = run_dir / "summary.md"
    assert summary_md.exists()
    assert "Best Parameter Sets" in summary_md.read_text()

    exp_csv.unlink()


def test_failure_tracking_and_rerun(tmp_path, monkeypatch):
    cfg = {
        "run_id": "failrun",
        "inputs": ["good.wav", "bad.wav"],
        "output_root": str(tmp_path),
    }

    def fake_preprocess(src, workdir, **kwargs):
        return src, []

    def fake_transcribe(audio_path, out_dir, **kwargs):
        if Path(audio_path).name == "bad.wav":
            raise RuntimeError("boom")
        return str(tmp_path / "segments.json")

    class DummySubs:
        def __init__(self):
            self.events = []

    def fake_load_segments(path):
        return DummySubs()

    def fake_enforce(subs, **kwargs):
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
    exp.run()

    failed_path = Path("failed.csv")
    assert failed_path.exists()
    rows = list(csv.DictReader(failed_path.open()))
    assert rows[0]["file"] == "bad.wav"

    # Patch to succeed on rerun
    def success_transcribe(audio_path, out_dir, **kwargs):
        return str(tmp_path / "segments.json")

    monkeypatch.setattr("experiment.transcribe_and_align", success_transcribe)
    from rerun_failed import main as rerun_main

    rerun_main()

    assert not failed_path.exists()
    exp_csv = Path("experiments.csv")
    if exp_csv.exists():
        exp_csv.unlink()


def test_parameter_sweep_outputs_and_aggregation(tmp_path, monkeypatch):
    cfg = {
        "inputs": ["audio.wav"],
        "output_root": str(tmp_path),
        "grid": {
            "transcribe.dummy": [1, 2],
            "format.max_chars": [10, 20],
        },
    }
    cfg_path = tmp_path / "config.yml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def fake_preprocess(src, workdir, **kwargs):
        return src, []

    def fake_transcribe(audio_path, out_dir, **kwargs):
        return str(tmp_path / "segments.json")

    class DummySubs:
        def __init__(self):
            self.events = []

    def fake_load_segments(path):
        return DummySubs()

    def fake_enforce(subs, **kwargs):
        pass

    def fake_write_outputs(subs, srt_path, _):
        Path(srt_path).write_text("dummy", encoding="utf-8")

    def fake_collect_metrics(path):
        return {"subtitle_count": 1}

    def fake_validate_sync(path, audio):
        return {"offset": 0.1}

    monkeypatch.setattr("experiment.preprocess_pipeline", fake_preprocess)
    monkeypatch.setattr("experiment.transcribe_and_align", fake_transcribe)
    monkeypatch.setattr("experiment.load_segments", fake_load_segments)
    monkeypatch.setattr("experiment.enforce_limits", fake_enforce)
    monkeypatch.setattr("experiment.write_outputs", fake_write_outputs)
    monkeypatch.setattr("experiment.qc.collect_metrics", fake_collect_metrics)
    monkeypatch.setattr("experiment.qc.validate_sync", fake_validate_sync)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["experiment_runner.py", str(cfg_path), "--sweep"])
    experiment_runner.main()

    run_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("run")]
    assert len(run_dirs) == 4
    for run_dir in run_dirs:
        run_id = run_dir.name
        cfg_file = run_dir / f"config_{run_id}.json"
        metrics_file = run_dir / f"metrics_{run_id}.json"
        log_file = run_dir / "run.log"
        commit_file = run_dir / f"commit_{run_id}.txt"
        req_file = run_dir / "requirements.txt"
        assert cfg_file.exists()
        assert metrics_file.exists()
        assert log_file.exists()
        assert commit_file.exists()
        assert req_file.exists()
        metrics = json.loads(metrics_file.read_text())
        assert metrics[0]["subtitle_count"] == 1

    exp_csv = tmp_path / "experiments.csv"
    assert exp_csv.exists()
    rows = list(csv.DictReader(exp_csv.open()))
    assert len(rows) == 4
    run_ids = {d.name for d in run_dirs}
    for row in rows:
        assert row["run_id"] in run_ids
        assert float(row["avg_subtitle_count"]) == 1.0
