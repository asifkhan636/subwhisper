"""High level experiment orchestration for SubWhisper."""
from __future__ import annotations

import csv
import json
import logging
import statistics
import uuid
from pathlib import Path
from typing import Any, Dict, List

from corrections import apply_corrections, load_corrections
from preproc import preprocess_pipeline
import qc
from subtitle_pipeline import enforce_limits, load_segments, write_outputs
from transcribe import transcribe_and_align


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
        self.run_id = config.get("run_id", uuid.uuid4().hex[:8])
        # Allow experiments to be directed to an alternate root directory.
        output_root = Path(config.get("output_root", "runs"))
        self.run_dir = output_root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # persist configuration for traceability
        (self.run_dir / "config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

        # log everything to a file under the run directory
        self.log_file = self.run_dir / "experiment.log"
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)

        self.logger = logging.getLogger(__name__)
        self.results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute preprocessing, transcription, formatting and QC."""

        inputs: List[str] = self.config.get("inputs", [])
        pre_cfg = self.config.get("preprocess", {})
        tr_cfg = self.config.get("transcribe", {})
        fmt_cfg = self.config.get("format", {})
        corrections_path = self.config.get("corrections")
        references = self.config.get("references", {})

        rules = None
        if corrections_path:
            try:
                rules = load_corrections(Path(corrections_path))
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.warning(
                    "Failed to load corrections %s: %s", corrections_path, exc
                )

        for src in inputs:
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
            enforce_limits(
                subs,
                fmt_cfg.get("max_chars", 42),
                fmt_cfg.get("max_lines", 2),
                fmt_cfg.get("max_duration", 6.0),
                fmt_cfg.get("min_gap", 0.1),
            )

            if rules:
                for ev in subs.events:
                    ev.text = apply_corrections(ev.text, rules)

            srt_path = work_dir / f"{src_path.stem}.srt"
            write_outputs(subs, srt_path, None)

            metrics = qc.collect_metrics(str(srt_path))
            ref_path = references.get(src_path.stem)
            if ref_path:
                metrics["wer"] = qc.compute_wer(str(srt_path), ref_path)

            sync_metrics = qc.validate_sync(str(srt_path), audio_path)
            metrics.update({f"sync_{k}": v for k, v in sync_metrics.items()})
            metrics["file"] = str(src)
            self.results.append(metrics)

    # ------------------------------------------------------------------
    def aggregate_results(self) -> Dict[str, Any]:
        """Aggregate metrics collected from all processed files."""
        summary: Dict[str, Any] = {}
        if not self.results:
            return summary

        numeric_keys = set()
        for res in self.results:
            for k, v in res.items():
                if k != "file" and isinstance(v, (int, float)):
                    numeric_keys.add(k)

        for key in numeric_keys:
            values = [res[key] for res in self.results if isinstance(res.get(key), (int, float))]
            if values:
                summary[f"avg_{key}"] = statistics.mean(values)

        self.summary = summary
        return summary

    # ------------------------------------------------------------------
    def report(self, format: str = "csv") -> None:
        """Persist metrics and summary tables to ``runs/<run_id>``."""
        if not self.results:
            return

        metrics_path = self.run_dir / f"metrics.{format}"
        summary_path = self.run_dir / f"summary.{format}"

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
