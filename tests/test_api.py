from pathlib import Path
import sys
import types

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from api import app, RUNS


def test_run_status_review_flow(tmp_path, monkeypatch):
    # Provide a lightweight experiment module used by the API
    class DummyExp:
        def __init__(self, config):
            self.run_id = config["run_id"]
            self.run_dir = Path(config["output_root"]) / self.run_id
            self.log_file = self.run_dir / "run.log"

    fake_module = types.ModuleType("experiment")
    fake_module.SubtitleExperiment = DummyExp
    monkeypatch.setitem(sys.modules, "experiment", fake_module)

    def fake_execute(exp, run_id):
        exp.run_dir.mkdir(parents=True, exist_ok=True)
        exp.log_file.write_text("done", encoding="utf-8")
        (exp.run_dir / "out.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nhello\n",
            encoding="utf-8",
        )
        RUNS[run_id]["status"] = "completed"

    monkeypatch.setattr("api._execute", fake_execute)

    client = TestClient(app)
    headers_admin = {"Authorization": "Bearer test-token"}
    headers_viewer = {"Authorization": "Bearer viewer-token"}

    config = {"run_id": "r1", "output_root": str(tmp_path)}
    resp = client.post("/run", json=config, headers=headers_admin)
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    resp = client.get(f"/status/{run_id}", headers=headers_viewer)
    assert resp.status_code == 200
    status = resp.json()
    assert status["status"] == "completed"
    assert "done" in status["log"]

    resp = client.get(f"/review/{run_id}", headers=headers_viewer)
    assert resp.status_code == 200
    review = resp.json()
    assert "out.srt" in review["subtitles"]
    assert "hello" in review["subtitles"]["out.srt"]

    RUNS.clear()
