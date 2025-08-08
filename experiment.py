"""High level experiment orchestration for SubWhisper."""
from __future__ import annotations

import csv
import json
import logging
import statistics
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import subprocess

from corrections import apply_corrections, load_corrections
from preproc import preprocess_pipeline
import qc
from subtitle_pipeline import enforce_limits, load_segments, write_outputs
from transcribe import transcribe_and_align

ROOT = Path(__file__).resolve().parent


def _get_git_commit() -> str:
    """Return current git commit hash or 'unknown' if unavailable."""
    try:
        result = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT)
        return result.decode().strip()
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _snapshot_requirements(path: Path) -> None:
    """Write ``pip freeze`` output to ``path``.

    Errors are swallowed to avoid disrupting experiment setup.
    """
    try:  # pragma: no cover - external command
        reqs = subprocess.check_output(["pip", "freeze"], text=True)
    except Exception:
        reqs = ""
    path.write_text(reqs, encoding="utf-8")


class SubtitleExperiment:
    """Run preprocessing, transcription, formatting and quality checks.

    Parameters
    ----------
    config:
        Dictionary of parameters controlling each pipeline stage. Expected keys
        include ``inputs`` (list of media files) and optional nested mappings
        for ``preprocess``, ``transcribe`` and ``format`` settings.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        if "run_id" in config:
            self.run_id = config["run_id"]
        else:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_id = f"{ts}-{uuid.uuid4().hex[:8]}"
            self.config["run_id"] = self.run_id

        # Allow experiments to be directed to an alternate root directory.
        output_root = Path(config.get("output_root", "runs"))
        self.run_dir = output_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # record git commit and environment snapshot for reproducibility
        self.git_commit = _get_git_commit()
        commit_path = self.run_dir / f"commit_{self.run_id}.txt"
        commit_path.write_text(self.git_commit, encoding="utf-8")
        self.config["git_commit"] = self.git_commit
        _snapshot_requirements(self.run_dir / "requirements.txt")

        # log everything to a file under the run directory
        self.log_file = self.run_dir / "run.log"

        class RunIDFilter(logging.Filter):
            def __init__(self, run_id: str) -> None:
                super().__init__()
                self.run_id = run_id

            def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple
                record.run_id = self.run_id
                return True

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(run_id)s %(name)s: %(message)s",
        )

        run_filter = RunIDFilter(self.run_id)

        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(run_filter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(run_filter)

        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.handlers.clear()
        root.addHandler(file_handler)
        root.addHandler(console_handler)

        self.logger = logging.getLogger(__name__)
        self.results: List[Dict[str, Any]] = []
        self.failures: List[Dict[str, Any]] = []

        # MLflow configuration (optional)
        self._mlflow = None
        self.mlflow_cfg = self.config.get("mlflow")

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute preprocessing, transcription, formatting and QC."""

        inputs: List[str] = self.config.get("inputs", [])
        pre_cfg = self.config.get("preprocess", {})
        tr_cfg = self.config.get("transcribe", {})
        fmt_cfg = self.config.get("format", {})
        corrections_path = self.config.get("corrections")
        references = self.config.get("references", {})

        # ------------------------------------------------------------------
        # Optional MLflow setup
        if self.mlflow_cfg:
            try:  # pragma: no cover - import guard tested separately
                import mlflow
            except ImportError as exc:  # pragma: no cover - tested
                raise RuntimeError(
                    "MLflow logging requested but mlflow is not installed. "
                    "Install mlflow or disable the 'mlflow' configuration."
                ) from exc

            self._mlflow = mlflow
            tracking_uri = self.mlflow_cfg.get("tracking_uri")
            experiment_name = self.mlflow_cfg.get("experiment_name")
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            if experiment_name:
                mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=self.run_id)

            def _flatten(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
                flat: Dict[str, Any] = {}
                for k, v in data.items():
                    key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        flat.update(_flatten(v, key))
                    else:
                        flat[key] = v
                return flat

            params = {k: str(v) for k, v in _flatten(self.config).items()}
            mlflow.log_params(params)

        # persist configuration for traceability
        (self.run_dir / f"config_{self.run_id}.json").write_text(
            json.dumps(self.config, indent=2), encoding="utf-8"
        )

        self.failures = []
        rules = None
        if corrections_path:
            try:
                rules = load_corrections(Path(corrections_path))
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(
                    "Failed to load corrections %s: %s", corrections_path, exc
                )

        for idx, src in enumerate(inputs, 1):
            try:
                src_path = Path(src)
                work_dir = self.run_dir / src_path.stem
                work_dir.mkdir(parents=True, exist_ok=True)

                audio_path, music_segments = preprocess_pipeline(
                    src, str(work_dir), **pre_cfg
                )

                trans_dir = work_dir / "transcript"
                trans_dir.mkdir(exist_ok=True)
                segments_path = transcribe_and_align(
                    audio_path,
                    str(trans_dir),
                    music_segments=music_segments,
                    **tr_cfg,
                )

                subs = load_segments(Path(segments_path))
                fmt_limits = {
                    "max_chars": 45,
                    "max_lines": 2,
                    "max_duration": 6.0,
                    "min_gap": 0.15,
                    **fmt_cfg,
                }
                enforce_limits(subs, **fmt_limits)

                if rules:
                    for ev in subs.events:
                        ev.text = apply_corrections(ev.text, rules)

                srt_path = work_dir / f"{src_path.stem}_{self.run_id}.srt"
                write_outputs(subs, srt_path, None)

                metrics = qc.collect_metrics(str(srt_path))
                ref_path = references.get(src_path.stem)
                if ref_path:
                    metrics["wer"] = qc.compute_wer(str(srt_path), ref_path)

                sync_metrics = qc.validate_sync(str(srt_path), audio_path)
                metrics.update({f"sync_{k}": v for k, v in sync_metrics.items()})
                metrics["file"] = str(src)
                metrics["status"] = "success"
                self.results.append(metrics)

                if self._mlflow:
                    numeric = {
                        k: v for k, v in metrics.items() if isinstance(v, (int, float))
                    }
                    if numeric:
                        self._mlflow.log_metrics(numeric, step=idx)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.error("Failed processing %s: %s", src, exc)
                err = {"file": str(src), "error": str(exc)}
                self.failures.append(err)
                self.results.append({"file": str(src), "status": "failed", **err})

        # log metrics for all processed files
        (self.run_dir / f"metrics_{self.run_id}.json").write_text(
            json.dumps(self.results, indent=2), encoding="utf-8"
        )

        if self.failures:
            failed_path = Path("failed.csv")
            write_header = not failed_path.exists()
            with failed_path.open("a", newline="", encoding="utf-8") as fh:
                fieldnames = ["run_id", "file", "config", "error"]
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                for item in self.failures:
                    writer.writerow(
                        {
                            "run_id": self.run_id,
                            "file": item["file"],
                            "config": json.dumps(self.config, sort_keys=True),
                            "error": item["error"],
                        }
                    )

        if self._mlflow:
            self._mlflow.end_run()

    # ------------------------------------------------------------------
    def aggregate_results(self) -> Dict[str, Any]:
        """Aggregate metrics collected from all processed files."""
        summary: Dict[str, Any] = {}

        if self.results:
            numeric_keys = set()
            for res in self.results:
                for k, v in res.items():
                    if k != "file" and isinstance(v, (int, float)):
                        numeric_keys.add(k)

            for key in numeric_keys:
                values = [
                    res[key]
                    for res in self.results
                    if isinstance(res.get(key), (int, float))
                ]
                if values:
                    summary[f"avg_{key}"] = statistics.mean(values)

        self.summary = summary

        experiments_path = Path("experiments.csv")
        row = {
            "run_id": self.run_id,
            "config": json.dumps(self.config, sort_keys=True),
        }
        row.update(summary)
        write_header = not experiments_path.exists()
        with experiments_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        self._write_summary_markdown()

        return summary

    # ------------------------------------------------------------------
    def report(self, format: str = "csv") -> None:
        """Persist metrics and summary tables to ``runs/<run_id>``."""
        if not self.results:
            return

        metrics_path = self.run_dir / f"metrics_{self.run_id}.{format}"
        summary_path = self.run_dir / f"summary_{self.run_id}.{format}"

        if format == "csv":
            with metrics_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.results[0].keys())
                writer.writeheader()
                writer.writerows(self.results)
            if hasattr(self, "summary"):
                with summary_path.open("w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=self.summary.keys())
                    writer.writeheader()
                    writer.writerow(self.summary)
        else:
            metrics_path.write_text(json.dumps(self.results, indent=2), encoding="utf-8")
            if hasattr(self, "summary"):
                summary_path.write_text(json.dumps(self.summary, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    def _best_params_table(self) -> List[Dict[str, Any]]:
        path = Path("experiments.csv")
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
        metrics = [f for f in fieldnames if f not in {"run_id", "config"}]
        table: List[Dict[str, Any]] = []
        for metric in metrics:
            vals = [r for r in rows if r.get(metric)]
            if not vals:
                continue
            try:
                best = min(vals, key=lambda r: float(r[metric]))
            except ValueError:  # pragma: no cover - non-numeric
                continue
            table.append(
                {
                    "metric": metric,
                    "run_id": best["run_id"],
                    "value": best[metric],
                    "config": best["config"],
                }
            )
        return table

    # ------------------------------------------------------------------
    @staticmethod
    def _make_md_table(rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return ""
        headers = list(rows[0].keys())
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _write_summary_markdown(self) -> None:
        summary_file = self.run_dir / "summary.md"
        lines = [f"# Summary for {self.run_id}", ""]
        if self.results:
            lines.append("## Metrics")
            lines.append(self._make_md_table(self.results))
            lines.append("")
        if hasattr(self, "summary") and self.summary:
            lines.append("## Aggregate Metrics")
            lines.append(self._make_md_table([self.summary]))
            lines.append("")
        best = self._best_params_table()
        if best:
            lines.append("## Best Parameter Sets")
            lines.append(self._make_md_table(best))
            lines.append("")
        summary_file.write_text("\n".join(lines), encoding="utf-8")
