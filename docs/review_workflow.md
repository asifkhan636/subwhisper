# Review Workflow

Human review can improve subtitle accuracy. This guide shows how to fetch generated subtitles, apply corrections, and submit the results back to the service.

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
