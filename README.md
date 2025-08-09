# subwhisper

## Project Overview

`subwhisper` is a small utility for generating subtitle files from a
directory of videos.  It uses
[WhisperX](https://github.com/m-bain/whisperX) to perform speech-to-text
transcription.  WhisperX releases prior to `vad_filter` support require a
separate voice‑activity detection (VAD) step if you need to trim silence.
The goal is to provide an easily hackable starting point for automated
subtitle workflows.

## Smoke Test

Run a quick end-to-end check on a single video with:

```bash
bash scripts/smoke_test.sh path/to/video.mkv
```

Intermediate artifacts are written to `preproc/` and `transcript/`, the final
subtitles to `subtitles.srt`, and a QC summary to `qc/summary.json`.

## API Authentication

The accompanying FastAPI server secures its endpoints with simple
token-based authentication. Tokens and their associated roles are stored in
an `auth.yaml` file located next to `api.py`:

```yaml
tokens:
  my-admin-token: admin
  read-only-token: viewer
```

Clients must send the token in the `Authorization` header using the
`Bearer <token>` scheme. Two roles are recognized:

- `admin` – start runs and submit subtitle reviews.
- `viewer` – read-only access for status checks, downloads and viewing
  current subtitles.

Edit `auth.yaml` to add or revoke credentials and restart the server to
apply changes.

## Installation Prerequisites

The scripts rely on a few external tools and Python packages. They have been
tested with `whisperx>=3.4.2`, `torch==1.13.1`, and `pyannote.audio==2.1.1`.
If your environment is missing these minimum versions, the CLI utilities will
report the issue on startup. WhisperX 3.4.x does not support the `vad_filter`
argument; to apply VAD you must either upgrade to a release that implements it
or run VAD separately with `whisperx.load_vad_model` /
`whisperx.detect_voice_activity` before transcription.

### Version compatibility

Pretrained Pyannote VAD models are sensitive to dependency versions. Use the
exact pins from `requirements.txt` (or `environment.yml`) to avoid runtime
warnings or failures:

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
```

These files ensure that `torch==1.13.1` and `pyannote.audio==2.1.1` are
installed.

### Required Software

- **Python**: 3.9 or newer
- **Conda**: for managing the environment
- **FFmpeg**: used for audio extraction
- **Python packages**: `torch==1.13.1`, `pyannote.audio==2.1.1`, `speechbrain>=1.0`, `whisperx>=3.4.2,<4`, `librosa>=0.10`, `noisereduce>=3.0`

### Create a Conda Environment

Use the provided `environment.yml` to set up everything in one step:

```bash
conda env create -f environment.yml
conda activate subwhisper
```

This installs Python, `torch==1.13.1`, `pyannote.audio==2.1.1`,
`speechbrain>=1.0`, `whisperx>=3.4.2,<4`, and other dependencies. The `torch`
entry is CPU‑only by default; edit `environment.yml` to choose a CUDA‑enabled
build or add optional packages.

> **Note:** The pretrained Pyannote diarization model currently requires
> `torch==1.13.1` and `pyannote.audio==2.1.1`. Using mismatched versions may
> trigger runtime warnings or failures, so ensure your environment matches
> these versions when relying on the pretrained model.

#### Upgrading dependencies

If you need newer features from `pyannote.audio`, `torch`, or `speechbrain`,
check model compatibility before upgrading. Installing versions that differ
from the pinned ones above can lead to runtime warnings or failures unless you
also obtain pretrained models built for those releases. Refer to the respective
project documentation for migration notes.

If you prefer to configure things manually:

```bash
conda create -n subwhisper python=3.10
conda activate subwhisper

# Install dependencies
 pip install "torch==1.13.1" "pyannote.audio==2.1.1" "speechbrain>=1.0" "whisperx>=3.4.2,<4"
# Install ffmpeg (choose one of the following)
conda install -c conda-forge ffmpeg    # via conda
# or
sudo apt-get install ffmpeg            # on Debian/Ubuntu
```

#### Upgrading PyTorch Lightning checkpoints

Older models saved with previous versions of PyTorch Lightning may need to be
upgraded before use. If you see a warning about `pytorch_model.bin`, run:

```bash
python -m pytorch_lightning.utilities.upgrade_checkpoint /path/to/pytorch_model.bin
```

This converts the checkpoint to the latest format so it can be loaded without
warnings.

## Docker

The repository includes a simple Docker setup for running the FastAPI service.

### Build the image

```bash
docker build -t subwhisper .
```

### Run with Docker

```bash
docker run -p 8000:8000 \
    -v "$(pwd)/input:/data/input" \
    -v "$(pwd)/output:/data/output" \
    subwhisper
```

### Using docker-compose

```bash
docker-compose up --build
```

Input media should be placed under the mounted `input` directory and results
will be written under `output`. When submitting jobs to the API or running an
experiment, reference container paths (e.g. `/data/input/video.mp4` and set
`output_root` to `/data/output`).

### CPU and GPU variants

The provided `Dockerfile` builds a CPU-only image that uses the CPU build of
PyTorch. For GPU acceleration, replace the base image with a CUDA-enabled
PyTorch image, then rebuild and run the container with access to your GPU:

```bash
docker build -t subwhisper-cuda .
docker run --gpus all subwhisper-cuda
```

Using `--gpus all` enables Docker's GPU passthrough so the container can run
CUDA workloads. A similar flag can be added to your Docker Compose
configuration if preferred.

The image omits optional alignment dependencies such as `aeneas`. Install them
manually if you need sync validation (`qc.py --sync`) or invoke `qc.py --no-sync`
to skip those checks.

## Phase-1: Audio Preprocessing

`preproc.py` prepares audio for transcription by extracting a mono 16 kHz track
from the input video and optionally cleaning it up. The pipeline can:

1. Extract the selected audio stream into `audio.wav`.
2. Apply noise reduction to produce `denoised.wav` when `--denoise` is used;
   control the strength with `--denoise-aggressive`.
3. Apply loudness normalization to produce `normalized.wav` when `--normalize`
   is set.
4. Detect music and write `[start, end]` time ranges to `music_segments.json` in
   the output directory.

All files are written under the directory given by `--outdir` (default:
`preproc/`).

### CLI usage

Run the complete preprocessing pipeline:

```bash
python preproc.py --input video.mp4 --denoise --denoise-aggressive 0.9 \
    --normalize --outdir preproc
```

Additional options:

- `--track N` – process a specific audio stream.
- `--music-threshold T` – adjust the music detection threshold (default
  `0.5`).
- `--denoise-aggressive A` – set noise reduction aggressiveness from 0 to 1
  (default `0.85`); used with `--denoise`.

### Output files

- `audio.wav` – raw extracted audio.
- `denoised.wav` – noise‑reduced audio when `--denoise` is used.
- `normalized.wav` – loudness‑normalized audio when `--normalize` is used.
- `music_segments.json` – JSON array of detected music segments in seconds,
  saved under the `--outdir` directory.

## Phase-2: Transcription & Alignment

`transcribe.py` converts the cleaned audio from Phase 1 into time-aligned
segments using WhisperX. It can also ingest a `music_segments.json` file
from Phase 1 and, with `--skip-music`, exclude those ranges from the
transcript.

### CLI usage

```bash
python transcribe.py preproc/normalized.wav --outdir transcript \
    --music-segments preproc/music_segments.json --skip-music \
    --device cuda
```

#### Options

- `audio_path` – path to the mono 16 kHz WAV produced by Phase 1 (for example,
  `preproc/normalized.wav`).
- `--outdir DIR` – directory where `transcript.json` and `segments.json` are
  written.
- `--model NAME` – Whisper model to load (default `large-v3-turbo`).
- `--batch-size N` – batch size for both transcription and alignment
  (default `8`).
- `--beam-size N` – beam search width used during decoding (default `5`).
- `--compute-type TYPE` – precision for WhisperX such as `float16` or
  `float32` (default `float32`).
- `--device DEVICE` – torch device to run on; defaults to `cuda` when a GPU
  is available, otherwise `cpu`.
- `--music-segments FILE` – optional JSON file with `[start, end]` pairs from
  Phase 1 to flag music regions; combine with `--skip-music` to drop
  segments overlapping those ranges.
- `--skip-music` – drop segments overlapping the provided music ranges.

### Output files

- `transcript.json` – raw WhisperX segments with an additional `is_music`
  boolean.
- `segments.json` – simplified segments used by downstream tooling:

  ```json
  [
    {
      "start": 0.0,
      "end": 3.2,
      "text": "hello world",
      "words": [
        {"word": "hello", "start": 0.0, "end": 1.2},
        {"word": "world", "start": 1.3, "end": 3.2}
      ]
    }
  ]
  ```

## Phase-3: Subtitle Formatting

`subtitle_pipeline.py` converts the aligned segments from Phase 2 into subtitle
and transcript files.

### CLI usage

```bash
python subtitle_pipeline.py --segments transcript/segments.json --output subtitles.srt --transcript
```

`--segments` points to the `segments.json` file produced in Phase 2. Use
`--output` to choose the subtitle file path and add `--transcript` to write a
plain-text transcript alongside it.

### Full pipeline example

```bash
# Phase 1: extract, clean, and detect music
python preproc.py --input video.mp4 --denoise --normalize --outdir preproc

# Phase 2: transcribe and align using the cleaned audio and music ranges
python transcribe.py preproc/normalized.wav --outdir transcript \
    --music-segments preproc/music_segments.json --skip-music \
    --device cuda

# Phase 3: format subtitles
python subtitle_pipeline.py --segments transcript/segments.json --output subs/video.srt --transcript
```

## Phase-4: Validation & Benchmarking

`qc.py` provides tooling to assess subtitle quality once transcription is
complete. It can compute word error rate (WER) against a reference
transcript, verify subtitle timing against the original audio, and gather
summary statistics across many files.

### Computing WER

```bash
python qc.py subs/episode1.srt --reference refs/episode1.txt --wer
```

This prints the WER between `subs/episode1.srt` and the reference text. It
requires the `jiwer` and `pysubs2` packages and outputs a floating‑point
score in the range `0.0–1.0`.

### Sync checks

```bash
python qc.py subs/episode1.srt --audio audio/episode1.wav --sync
```

The command runs a forced‑alignment check (requires optional `aeneas` and
`pysubs2`) and reports offsets such as `mean_offset`, `median_offset`, and
`max_offset` in seconds. These packages are not installed in the CPU-only
Docker image; pass `--no-sync` to skip this step when they are missing.

### Batch validation

```bash
python qc.py subs/ -r refs/ -a audio/ --recursive --json qc/results.json --csv qc/results.csv
```

Processes all subtitle files under `subs/`, matching reference transcripts
and audio by filename. Per‑file metrics are printed to the console and the
aggregated results are written to JSON and CSV files.

## Maintenance

Old run outputs can accumulate over time. The `maintenance.py` helper removes
large intermediate files and archives completed runs to save space. By default
it performs a dry run and simply reports the actions:

```bash
python maintenance.py --output-root runs --days 30
```

Pass `--delete` to apply the deletions after creating `.tar.gz` archives:

```bash
python maintenance.py --output-root runs --days 30 --delete
```

### Scheduling

To run the cleanup periodically with cron:

```cron
0 2 * * * /usr/bin/python /path/to/maintenance.py --output-root /data/output --days 30 --delete
```

In Airflow, a `BashOperator` can invoke the script on a schedule:

```python
BashOperator(
    task_id="subwhisper_maintenance",
    bash_command="python /path/to/maintenance.py --output-root /data/output --days 30 --delete",
)
```

## Potential Enhancements

- Integrate speaker diarization for multi‑speaker videos
- Generate word‑level timestamps or translation
- Parallelise processing across multiple GPUs
- [Deployment guide](docs/deployment.md) for API, Docker, Airflow and authentication setup
- [Review workflow](docs/review_workflow.md) describing human-in-the-loop corrections

## Troubleshooting Tips

- Ensure `ffmpeg` is available on your `PATH`.
- If GPU memory is exhausted, try a smaller `--model-size`.
- For missing Python modules, check that the conda environment is
  activated before running the script.
- Review `failed_subtitles.log` for videos that could not be processed.

## Running Tests

The project includes a pytest suite with mocked dependencies. To run all tests
locally:

```bash
pytest
```

To focus on the API endpoints verified in `tests/test_api.py` you can run:

```bash
pytest tests/test_api.py
```

These tests rely on lightweight stubs, so they execute quickly without
installing heavy runtime dependencies like `ffmpeg` or `whisperx`.

