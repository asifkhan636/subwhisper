# subwhisper – Getting Started Guide

Subwhisper turns videos into subtitle files using the WhisperX speech‑to‑text engine.  
Follow these simple steps to make subtitles for your videos.

---

## 1. Install the Software

### Windows (PowerShell or Anaconda Prompt)

1. Install [Anaconda](https://www.anaconda.com/download), which includes Python and Conda.
2. Open **Anaconda Prompt**.
3. Run:

   ```powershell
   cd path\to\subwhisper
   conda env create -f environment.yml
   conda activate subwhisper
   ```

### macOS / Linux (Terminal)

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Conda.
2. Install FFmpeg (audio tools):

   ```bash
   conda install -c conda-forge ffmpeg
   # or (Linux)
   sudo apt-get install ffmpeg
   ```
3. Create and activate the project environment:

   ```bash
   cd /path/to/subwhisper
   conda env create -f environment.yml
   conda activate subwhisper
   ```

The environment installs Python, FFmpeg, WhisperX, and other required packages.
It also pins `torch` and `pyannote.audio` to versions compatible with the
pretrained Pyannote VAD model (`torch==1.13.1`, `pyannote.audio==2.1.1`). Using
different versions may lead to runtime warnings or failures.

---

## 2. Prepare Your Video

1. **Use the English audio track.** The tools try to pick the English track automatically
2. Place your video (e.g., `video.mp4`) in a folder without other large files.
3. Run the preprocessing script to extract and clean the audio:

   ```bash
   python preproc.py --input video.mp4 --denoise --denoise-aggressive 0.9 --normalize --outdir preproc
   ```

   This creates files like `preproc/audio.wav`, `preproc/denoised.wav`, `preproc/normalized.wav`, and `preproc/music_segments.json`
   *Tip: Removing background music or noise improves accuracy*

---

## 3. Transcribe the Audio

1. Feed the cleaned audio into WhisperX (use `--device cuda` for a GPU or
   `--device cpu` if no GPU is available):

   ```bash
   python transcribe.py preproc/normalized.wav --outdir transcript --music-segments preproc/music_segments.json --device cuda
   ```

2. Results:
   - `transcript/transcript.json`
   - `transcript/segments.json` (used for subtitle creation)

For reference, the complete Phase 1→2 pipeline looks like this:

```bash
# Phase 1: extract and clean
python preproc.py --input video.mp4 --denoise --normalize --outdir preproc

# Phase 2: transcribe and align
python transcribe.py preproc/normalized.wav --outdir transcript --music-segments preproc/music_segments.json --device cuda
```

---

## 4. Create Subtitle (SRT) Files

Run the subtitle formatter on the `segments.json` file:

```bash
python subtitle_pipeline.py --segments transcript/segments.json --output subtitles.srt --transcript
```

- The `.srt` subtitle file is saved as `subtitles.srt`.
- A plain-text transcript (`subtitles.txt`) is created when you add `--transcript`.
- Optional: provide a corrections file to fix common mistakes:

  ```bash
  python subtitle_pipeline.py --segments transcript/segments.json --output subtitles.srt --corrections rules.yml
  ```

---

## 5. Review and Fix Subtitles (Optional)

### Manual Editing
Open the generated `.srt` file in any subtitle editor or plain text editor and make changes.

### Automatic Corrections
Create a JSON or YAML file with “find → replace” rules and apply them:

```python
from corrections import load_corrections, apply_corrections
rules = load_corrections(Path("rules.yml"))
new_text = apply_corrections(old_text, rules)
```

For a more structured review workflow (fetching, editing, and submitting fixes through the API), see the project’s review guide

---

## 6. Troubleshooting

- Make sure `ffmpeg` is installed and on your path.
- If the program says you ran out of GPU memory, try a smaller model size.
- If Python packages are “missing,” ensure the Conda environment is activated.
- Check `failed_subtitles.log` for any videos that failed to process

---

## 7. Best Practices for Accuracy

- Always use the English dub or audio track
- Keep the audio as clean as possible—reduce noise and background music before transcribing
- Try the workflow on a short test clip before processing long videos.

---

## 8. FAQ & Tips

**Q: Where do the subtitles go?**  
A: Wherever you point the `--output` option. Use a full path if you want a specific folder.

**Q: Can I make WebVTT files instead of SRT?**  
A: Yes—run `subtitle_pipeline.py` and rename the output to `.vtt`, then open in a converter if needed.

**Q: The words are misspelled. What can I do?**  
A: Add `--spellcheck` to `subtitle_pipeline.py` or apply corrections with a rules file.

**Q: How can I speed things up?**  
A: Process one short video at a time until you’re comfortable. Run on a machine with a GPU for faster transcription.

**Q: Where can I ask for help?**  
A: Check the project’s issue tracker on GitHub or ask in the repository discussions.

---

Enjoy creating subtitles with subwhisper!

