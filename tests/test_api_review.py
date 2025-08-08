from __future__ import annotations

import json
from pathlib import Path

import sys

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api import RUNS, app


def _setup_run(tmp_path: Path):
    run_id = "test_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    srt_path = run_dir / "sample.srt"
    srt_path.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nteh cat\n", encoding="utf-8"
    )
    (run_dir / "run.log").write_text("", encoding="utf-8")
    RUNS[run_id] = {
        "status": "completed",
        "run_dir": str(run_dir),
        "log_file": str(run_dir / "run.log"),
    }
    return run_id, srt_path


def test_review_roundtrip(tmp_path: Path):
    run_id, srt_path = _setup_run(tmp_path)
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-token"}

    resp = client.get(f"/review/{run_id}", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert "sample.srt" in data["subtitles"]
    assert "teh cat" in data["subtitles"]["sample.srt"]

    payload = {"corrections": {"teh": "the"}, "reviewer": {"name": "Bob"}}
    resp = client.post(f"/review/{run_id}", json=payload, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["applied"] == 1

    updated = srt_path.read_text(encoding="utf-8")
    assert "the cat" in updated

    corr_file = Path(RUNS[run_id]["run_dir"]) / "corrections.json"
    assert json.loads(corr_file.read_text(encoding="utf-8"))["teh"] == "the"

    log_lines = (
        Path(RUNS[run_id]["run_dir"]) / "review_log.jsonl"
    ).read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(log_lines[-1])["reviewer"]["name"] == "Bob"

    RUNS.clear()


def test_requires_auth(tmp_path: Path) -> None:
    run_id, _ = _setup_run(tmp_path)
    client = TestClient(app)
    resp = client.get(f"/review/{run_id}")
    assert resp.status_code == 401
    RUNS.clear()
