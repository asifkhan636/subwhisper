from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.email import send_email

from experiment import SubtitleExperiment

log = logging.getLogger(__name__)


def _run_experiment(**context: Dict[str, Any]) -> None:
    """Run ``SubtitleExperiment`` with parameters from the DAG run."""
    params: Dict[str, Any] = context.get("dag_run").conf or {}
    log.info("Starting SubtitleExperiment with params: %s", params)
    exp = SubtitleExperiment(params)
    exp.run()


def _notify_failure(context: Dict[str, Any]) -> None:  # pragma: no cover - Airflow callback
    """Send an email notification when the task or DAG fails."""
    subject = f"DAG {context['dag'].dag_id} failed"
    body = f"Run {context['run_id']} failed. See logs for details."
    send_email(to="alerts@example.com", subject=subject, html_content=body)


default_args = {
    "owner": "airflow",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="subtitle_pipeline",
    description="Run SubWhisper subtitle experiments",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    on_failure_callback=_notify_failure,
) as dag:
    run_experiment = PythonOperator(
        task_id="run_experiment",
        python_callable=_run_experiment,
    )
