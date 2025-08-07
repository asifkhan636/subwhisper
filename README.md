# subwhisper

## Project Overview

`subwhisper` is a small utility for generating subtitle files from a
directory of videos.  It uses
[WhisperX](https://github.com/m-bain/whisperX) to perform speech-to-text
transcription.  WhisperX releases prior to `vad_filter` support require a
separate voice‑activity detection (VAD) step if you need to trim silence.
The goal is to provide an easily hackable starting point for automated
subtitle workflows.

## Installation Prerequisites

The script relies on a few external tools and Python packages. It has been
tested with `whisperx>=3.4.2`, `torch>=2.5`, and `pyannote.audio>=3.3`.
On startup, `generateSubtitles.py` checks that these minimum versions are
installed and provides guidance if outdated releases are detected. WhisperX
3.4.x does not support the `vad_filter` argument; to apply VAD you must either
upgrade to a release that implements it or run VAD separately with
`whisperx.load_vad_model` / `whisperx.detect_voice_activity` before
transcription.

### Required Software

- **Python**: 3.9 or newer
- **Conda**: for managing the environment
- **FFmpeg**: used for audio extraction
- **Python packages**: `torch>=2.5`, `pyannote.audio>=3.3`, `speechbrain>=1.0`, `whisperx>=3.4.2,<4`, `librosa>=0.10`, `noisereduce>=3.0`

### Create a Conda Environment

Use the provided `environment.yml` to set up everything in one step:

```bash
conda env create -f environment.yml
conda activate subwhisper
```

This installs Python, `torch>=2.5`, `pyannote.audio>=3.3`,
`speechbrain>=1.0`, `whisperx>=3.4.2,<4`, and other dependencies. The `torch`
entry is CPU‑only by default; edit `environment.yml` to choose a CUDA‑enabled
build or add optional packages.

#### Upgrading dependencies

If you need newer features from `pyannote.audio`, `torch`, or `speechbrain`,
run `pip install --upgrade torch pyannote.audio speechbrain` and ensure that
you download models compatible with the new versions. Refer to the respective
project documentation for migration notes.

If you prefer to configure things manually:

```bash
conda create -n subwhisper python=3.10
conda activate subwhisper

# Install dependencies
 pip install "torch>=2.5" "pyannote.audio>=3.3" "speechbrain>=1.0" "whisperx>=3.4.2,<4"
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

## Phase-1: Audio Preprocessing

`preproc.py` prepares audio for transcription by extracting a mono 16 kHz track
from the input video and optionally cleaning it up. The pipeline can:

1. Extract the selected audio stream into `audio.wav`.
2. Apply noise reduction to produce `denoised.wav` when `--denoise` is used.
3. Apply loudness normalization to produce `normalized.wav` when `--normalize`
   is set.
4. Detect music and write `[start, end]` time ranges to `music_segments.json`.

All files are written under the directory given by `--outdir` (default:
`preproc/`).

### CLI usage

Run the complete preprocessing pipeline:

```bash
python preproc.py --input video.mp4 --denoise --normalize --outdir preproc
```

Additional options:

- `--track N` – process a specific audio stream.
- `--music-threshold T` – adjust the music detection threshold (default
  `0.5`).

### Output files

- `audio.wav` – raw extracted audio.
- `denoised.wav` – noise‑reduced audio when `--denoise` is used.
- `normalized.wav` – loudness‑normalized audio when `--normalize` is used.
- `music_segments.json` – JSON array of detected music segments in seconds.

## Usage

Activate the `subwhisper` conda environment before running any commands.

1. Place the videos you want to process in a directory.
2. Run `generateSubtitles.py`, pointing it at the directory:

```bash
python generateSubtitles.py /path/to/videos
```

To inspect the available audio tracks for a video before processing, use:

```bash
python generateSubtitles.py --list-audio-tracks myvideo.mkv
```

This prints track indices, language codes and descriptions so you can choose the
appropriate `--audio-track` value.

Sample command with explicit options:

```bash
python generateSubtitles.py ./media \
    --model-size medium \
    --output-format vtt \
    --output-dir ./subs \
    --extensions .mp4 .mkv \
    --language en
```

Subtitle files (`.srt` or `.vtt`) will be written alongside the
corresponding videos by default.  Use `--output-dir` to place them under a
separate directory while preserving the videos' relative paths.

### Voice activity detection (VAD)

The bundled WhisperX version does not expose VAD through `model.transcribe`.
If you need to remove silence, either upgrade to a WhisperX release that adds
the `vad_filter` argument or run a separate VAD pass via
`whisperx.load_vad_model` and `whisperx.detect_voice_activity`, then
transcribe only the detected speech segments.

## Usage Examples

### Basic batch processing

```bash
python generateSubtitles.py ./videos
```

Processes every supported video in `./videos` with default settings.

### Selecting a specific audio track

```bash
python generateSubtitles.py ./videos --audio-track 1
```

Choose a different audio track when a file contains multiple tracks.

### Changing output format

```bash
python generateSubtitles.py ./videos --output-format vtt
```

Create WebVTT subtitles instead of the default SRT files.

### Overriding language

```bash
python generateSubtitles.py ./videos --language es
```

Force the transcription language (Spanish in this example) when
auto-detection is unreliable.

### Word timestamps and diarization

```bash
python generateSubtitles.py ./videos --word-timestamps --diarize
```

Embed word-level timing and speaker labels for detailed editing or analysis.

### Parallel workers

```bash
python generateSubtitles.py ./videos --workers 4
```

Process videos in parallel with four worker processes to reduce total time.

### Writing to a separate directory

```bash
python generateSubtitles.py ./videos --output-dir subs/
```

Store the generated subtitle files under the `subs/` directory while
preserving each video's relative path.


## Logging

After each video is processed a summary entry is appended to
`logs/subtitle_run.json`. The record includes start and end timestamps,
whether the operation succeeded, and any associated error message. The
`logs` directory is created automatically if it does not already exist.

## Configurable Options

The CLI exposes a number of switches for customising behaviour:

- `--extensions`: video file extensions to search for (default: `.mp4 .mkv .mov .avi`)
- `--audio-track`: audio track index to extract. By default the script attempts to
  auto-detect the spoken-language track; run `--list-audio-tracks` to discover
  indices or pass an explicit value
- `--list-audio-tracks VIDEO`: list audio tracks for a single video and exit
- `--model-size`: Whisper model size to load (e.g., `base`, `large-v2`; default: `large-v2`)
- `--output-format`: subtitle format (`srt` or `vtt`, default `srt`)
- `--output-dir`: directory where subtitle files are written; relative paths
  under the input directory are preserved
- `--max-line-width`: maximum characters per subtitle line (default: `42`)
- `--max-lines`: maximum lines per subtitle (default: `2`)
- `--case`: normalise subtitle text casing (`lower` or `upper`)
- `--strip-punctuation`: remove punctuation from subtitle text
- `--language`: override language detection with a code like `en` (default: auto)

## Potential Enhancements

- Integrate speaker diarization for multi‑speaker videos
- Generate word‑level timestamps or translation
- Parallelise processing across multiple GPUs
- Automatically clean up temporary audio files

## Troubleshooting Tips

- Ensure `ffmpeg` is available on your `PATH`.
- If GPU memory is exhausted, try a smaller `--model-size`.
- For missing Python modules, check that the conda environment is
  activated before running the script.
- Review `failed_subtitles.log` for videos that could not be processed.

## Running Tests

The project includes a small pytest suite with mocked dependencies. To run
the tests, execute:

```bash
pytest
```

The tests use mock objects so they do not require heavy dependencies such as
`ffmpeg` or `whisperx` to be installed.

