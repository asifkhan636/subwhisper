#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 path/to/video" >&2
  exit 1
fi

VIDEO="$1"

# Preprocess audio
python preproc.py --input "$VIDEO"

# Transcribe audio
python transcribe.py preproc/audio.wav --outdir transcript --music-segments preproc/music_segments.json

# Generate subtitles
python subtitle_pipeline.py --segments transcript/segments.json --output subtitles.srt

# Run quality control
python qc.py subtitles.srt --audio preproc/audio.wav --json qc/summary.json

echo "QC summary written to qc/summary.json"
