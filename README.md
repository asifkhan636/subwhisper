# subwhisper

## Project Overview

`subwhisper` is a small utility for generating subtitle files from a
directory of videos.  It uses
[WhisperX](https://github.com/m-bain/whisperX) to perform speech-to-text
transcription and optional voice‑activity detection (VAD) to trim
silence.  The goal is to provide an easily hackable starting point for
automated subtitle workflows.

## Installation Prerequisites

The script relies on a few external tools and Python packages.

### Required Software

- **Python**: 3.9 or newer
- **Conda**: for managing the environment
- **FFmpeg**: used for audio extraction
- **Python packages**: `torch`, `pyannote.audio`, `whisperx`

### Create a Conda Environment

```bash
conda create -n subwhisper python=3.10
conda activate subwhisper

# Install dependencies
pip install torch pyannote.audio whisperx
# Install ffmpeg (choose one of the following)
conda install -c conda-forge ffmpeg    # via conda
# or
sudo apt-get install ffmpeg            # on Debian/Ubuntu
```

## Usage

1. Place the videos you want to process in a directory.
2. Run `generateSubtitles.py`, pointing it at the directory:

```bash
python generateSubtitles.py /path/to/videos
```

Sample command with explicit options:

```bash
python generateSubtitles.py ./media \
    --model-size medium \
    --output-format vtt \
    --extensions .mp4 .mkv \
    --language en
```

Subtitle files (`.srt` or `.vtt`) will be written alongside the
corresponding videos.

## Logging

After each video is processed a summary entry is appended to
`logs/subtitle_run.json`. The record includes start and end timestamps,
whether the operation succeeded, and any associated error message. The
`logs` directory is created automatically if it does not already exist.

## Configurable Options

The CLI exposes a number of switches for customising behaviour:

- `--extensions`: video file extensions to search for (default: `.mp4 .mkv .mov .avi`)
- `--audio-track`: select which audio track to extract (default: `1`; use `0` for first track)
- `--model-size`: Whisper model size to load
- `--vad-model`: VAD backend (`silero_vad` by default)
- `--vad-threshold`: activation threshold for VAD (default: `0.35`)
- `--output-format`: subtitle format (`srt` or `vtt`, default `srt`)
- `--max-line-width`: maximum characters per subtitle line (default: `42`)
- `--max-lines`: maximum lines per subtitle (default: `2`)
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

