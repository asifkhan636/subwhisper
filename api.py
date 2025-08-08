from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse

from experiment import SubtitleExperiment

app = FastAPI()

# Simple in-memory tracking of runs. In a production system this should be
# replaced with a persistent store or queue.
RUNS: Dict[str, Dict[str, Any]] = {}


def _execute(exp: SubtitleExperiment, run_id: str) -> None:
    """Helper to execute an experiment in the background."""
    info = RUNS[run_id]
    try:
        exp.run()
        exp.aggregate_results()
        exp.report()
        info["status"] = "completed"
    except Exception as exc:  # pragma: no cover - defensive
        info["status"] = "failed"
        info["error"] = str(exc)


@app.post("/run")
def run(config: Dict[str, Any], background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Start a new experiment run with the provided configuration."""
    exp = SubtitleExperiment(config)
    run_id = exp.run_id
    RUNS[run_id] = {
        "status": "running",
        "run_dir": str(exp.run_dir),
        "log_file": str(exp.log_file),
    }
    background_tasks.add_task(_execute, exp, run_id)
    return {"run_id": run_id}


@app.get("/status/{run_id}")
def status(run_id: str) -> Dict[str, Any]:
    """Return the status, log content and output directory for a run."""
    info = RUNS.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="run_id not found")
    log_text = None
    log_file = Path(info["log_file"])
    if log_file.exists():
        try:
            log_text = log_file.read_text(encoding="utf-8")
        except Exception:  # pragma: no cover - defensive
            log_text = None
    return {
        "run_id": run_id,
        "status": info["status"],
        "log": log_text,
        "output_dir": info["run_dir"],
    }


@app.get("/download/{run_id}")
def download(run_id: str) -> FileResponse:
    """Provide a zip archive of the run directory for download."""
    info = RUNS.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="run_id not found")
    run_dir = Path(info["run_dir"])
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="run directory missing")
    zip_path = run_dir.with_suffix(".zip")
    if not zip_path.exists():
        shutil.make_archive(str(run_dir), "zip", run_dir)
    return FileResponse(zip_path, filename=f"{run_dir.name}.zip")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run SubWhisper API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
