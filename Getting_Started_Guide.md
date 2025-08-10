# Subwhisper – Super Simple Guide

Subwhisper turns a video into subtitles (SRT) with one command.  
No deep setup, no extra steps.

---

## 1) Install (once)

### Windows (Anaconda Prompt)  
1) Install Anaconda: https://www.anaconda.com/download  
2) Open **Anaconda Prompt**  
3) Run:
```powershell
cd path\to\subwhisper
conda env create -f environment.yml
conda activate subwhisper
```

### macOS / Linux (Terminal)  
1) Install Miniconda: https://docs.conda.io/en/latest/miniconda.html  
2) Run:
```bash
cd /path/to/subwhisper
conda env create -f environment.yml
conda activate subwhisper
```

Tip: If you have a GPU and CUDA set up, you can use `--device cuda` for faster runs.
Otherwise just use `--device cpu`.

## 2) One command (the easy way)

Put your video file in a folder. Then run:

```bash
python subwhisper_cli.py --input /path/to/your_video.mp4 --device cpu
```

That’s it. When it finishes:

- ✅ You get `/path/to/your_video.srt` next to your video
- ✅ A plain, readable subtitle file (SRT) you can open anywhere
- ✅ All temporary files are cleaned up automatically (on success)

If anything goes wrong, Subwhisper keeps the temporary files so you (or a friend) can help diagnose the problem.

## 3) Process a whole folder

```bash
python subwhisper_cli.py --input /path/to/folder --device cpu
```

Subtitles are created for each supported video inside that folder.

By default, each `.srt` is saved right next to its video.

## 4) Optional: choose a different output folder

```bash
python subwhisper_cli.py --input /path/to/your_video.mp4 --output-root /path/to/subtitles --device cpu
```

You’ll get `/path/to/subtitles/your_video.srt`.

## 5) Optional: also create a text transcript

```bash
python subwhisper_cli.py --input /path/to/your_video.mp4 --write-transcript --device cpu
```

This creates `your_video.txt` alongside the `.srt`.

## 6) Resume or recompute later

Subwhisper caches progress in `.subwhisper_cache/` so you can skip completed
work or clean up old runs.

- `--resume {auto,off}` controls whether cached results are reused (`auto`
  by default).
- `--force` recomputes everything, ignoring any cache.
- `--resume-clean DAYS` deletes cache manifests older than `DAYS` days before
  starting.

Examples:

First run

```bash
python subwhisper_cli.py --input "…/Season 9" --device cuda
```

Resume later

```bash
python subwhisper_cli.py --input "…/Season 9" --resume auto --device cuda
```

Recompute everything

```bash
python subwhisper_cli.py --input "…/Season 9" --force
```

Clean manifests older than 14 days

```bash
python subwhisper_cli.py --input "…/Season 9" --resume-clean 14
```

## 7) Quick tips

- If your computer has a supported NVIDIA GPU, try `--device cuda` for speed.
- Cleaner audio → better subtitles. If possible, use the English audio track and reduce loud background music.

## Troubleshooting (short)

- “Command not found” / “module not found”: Make sure you ran `conda activate subwhisper` first.
- “ffmpeg not found”: Re-run the install steps; ffmpeg comes from the environment.
- It failed mid-way: That’s okay—temporary files are kept. Re-run the same command, or share the error message.

## For power users (optional)

### Common flags

- `--device {cuda,cpu}` – pick GPU or CPU
- `--output-root <folder>` – put final files somewhere else
- `--write-transcript` – also write a .txt transcript
- `--skip-music` – ignore detected music segments
- `--resume {auto,off}` – reuse cached results (`auto`, default) or disable
  resuming
- `--force` – recompute even if cached outputs exist
- `--resume-clean DAYS` – remove manifests older than `DAYS` days

### Do it step by step

If you prefer the original 3-step flow:

1) Preprocess

```bash
python preproc.py --input video.mp4 --normalize --outdir preproc
```

2) Transcribe + align

```bash
python transcribe.py preproc/normalized.wav --outdir transcript --device cpu
```

3) Make subtitles

```bash
python subtitle_pipeline.py --segments transcript/segments.json --output subtitles.srt
```

Advanced checks (optional): `qc.py` can report timing/reading-speed metrics. The `--sync` option requires extra packages; if you see errors, just skip it.

### (Optional) Docker

If you use Docker, you can run the same one-command script inside a container by mounting your files and calling:

```bash
python subwhisper_cli.py --input /data/input/your_video.mp4 --device cpu
```

---

You’re done 🎉  
You now have a `.srt` file you can open in any video player or editor.

