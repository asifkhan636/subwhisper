# Deployment Guide

This guide outlines the main deployment options for `subwhisper`, covering the API server, Docker usage, Airflow integration, and authentication. The CLI workflow consists of `preproc.py`, `transcribe.py`, and `subtitle_pipeline.py`.

## FastAPI Service

Run the transcription API locally with Uvicorn:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The service exposes endpoints for starting runs, checking status, and submitting subtitle reviews.

## Authentication Setup

Endpoints are protected with bearer tokens defined in `auth.yaml` next to `api.py`:

```yaml
tokens:
  my-admin-token: admin
  read-only-token: viewer
```

Provide the token in requests using an `Authorization: Bearer <token>` header. Restart the server after editing the file to apply changes.

## Docker

Build and run the service in a container:

```bash
docker build -t subwhisper .
docker run -p 8000:8000 \
    -v "$(pwd)/input:/data/input" \
    -v "$(pwd)/output:/data/output" \
    subwhisper
```

For multi-container setups use `docker-compose up --build`. Place input media under the mounted `input` directory and results under `output`. API payloads or experiment configs should reference paths inside the container (e.g., `/data/input/video.mp4` and `output_root: /data/output`). The provided Dockerfile builds a CPU-only image and omits optional alignment tools like `aeneas`; install those packages or run `qc.py --no-sync` if you need sync validation.

## Airflow

The repository ships with an example DAG at `airflow/pipeline_dag.py`. Copy the file into your Airflow instance's `dags/` directory and ensure required dependencies from `requirements.txt` are installed. Trigger the DAG with configuration matching the `SubtitleExperiment` parameters to automate transcription jobs.
