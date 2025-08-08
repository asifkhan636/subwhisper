from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pysubs2
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse

from corrections import apply_corrections, load_corrections
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


@app.get("/review/{run_id}")
def review(run_id: str) -> Dict[str, Any]:
    """Return current subtitle files for ``run_id``."""
    info = RUNS.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="run_id not found")
    run_dir = Path(info["run_dir"])
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="run directory missing")
    subtitles: Dict[str, str] = {}
    for path in run_dir.rglob("*.srt"):
        try:
            subtitles[str(path.relative_to(run_dir))] = path.read_text(
                encoding="utf-8"
            )
        except Exception:  # pragma: no cover - defensive
            continue
    return {"run_id": run_id, "subtitles": subtitles}


@app.post("/review/{run_id}")
def submit_review(run_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Accept corrections and reapply them to subtitle files.

    The request body should contain a ``corrections`` mapping and optional
    ``reviewer`` metadata. Corrections are merged into
    ``runs/<run_id>/corrections.json`` and immediately applied to all SRT
    files under the run directory. Each submission is appended to
    ``review_log.jsonl`` for auditing.
    """

    info = RUNS.get(run_id)
    if not info:
        raise HTTPException(status_code=404, detail="run_id not found")
    run_dir = Path(info["run_dir"])
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="run directory missing")

    corrections = payload.get("corrections")
    if not isinstance(corrections, dict) or not corrections:
        raise HTTPException(status_code=400, detail="corrections mapping required")
    reviewer = payload.get("reviewer", {})

    corr_path = run_dir / "corrections.json"
    if corr_path.exists():
        try:
            existing = json.loads(corr_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - defensive
            existing = {}
    else:
        existing = {}
    existing.update({str(k): str(v) for k, v in corrections.items()})
    corr_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "reviewer": reviewer,
        "changes": corrections,
    }
    with (run_dir / "review_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(log_entry) + "\n")

    try:
        rules = load_corrections(corr_path)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    for srt_path in run_dir.rglob("*.srt"):
        try:
            subs = pysubs2.load(str(srt_path))
            for ev in subs.events:
                fixed = apply_corrections(ev.plaintext, rules)
                ev.text = fixed.replace("\n", "\\N")
            subs.save(str(srt_path), format_="srt")
        except Exception:  # pragma: no cover - defensive
            continue

    logging.getLogger(__name__).info(
        "review submitted for %s by %s", run_id, reviewer or "unknown"
    )
    return {"run_id": run_id, "applied": len(corrections)}


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run SubWhisper API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
