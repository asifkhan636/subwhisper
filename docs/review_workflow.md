# Review Workflow

Human review can improve subtitle accuracy. This guide shows how to fetch generated subtitles, apply corrections, and submit the results back to the service.

If the API runs inside a container, mount input media under `/data/input` and
write results to `/data/output`. Refer to these paths in any payloads passed to
the service. The default Docker image is CPU-only and omits optional alignment
packages like `aeneas`; perform sync checks only if those dependencies are
installed (e.g., run `qc.py --no-sync` otherwise).

## Retrieve Subtitles

1. Identify the `run_id` of the transcription job.
2. Request the files and current metadata:

```bash
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/review/<run_id>
```

The response contains a mapping of subtitle filenames to their contents.

## Apply Corrections

Edit the subtitle text in your editor or load a rules file with `corrections.py`:

```python
from corrections import load_corrections, apply_corrections
rules = load_corrections(Path("rules.yml"))
new_text = apply_corrections(old_text, rules)
```

## Submit the Review

Send the corrected text back to the API with reviewer information:

```bash
curl -X POST -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"corrections": {"teh": "the"}, "reviewer": {"name": "Bob"}}' \
  http://localhost:8000/review/<run_id>
```

Corrections are merged into the existing subtitles and logged to `review_log.jsonl` for auditing.
