import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from preproc import preprocess_pipeline
from transcribe import transcribe_and_align
from subtitle_pipeline import load_segments, enforce_limits, write_outputs
from corrections import load_corrections, apply_corrections


def _resolve_outputs(input_path: Path, output_root: Optional[Path]) -> Tuple[Path, Path, str]:
    stem = input_path.stem
    if output_root:
        out_dir = output_root
    else:
        out_dir = input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_srt = out_dir / f"{stem}.srt"
    out_txt = out_dir / f"{stem}.txt"
    return out_srt, out_txt, stem


def _work_dir_for(input_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return input_path.parent / ".subwhisper_runs" / ts / input_path.stem


def _process_one(
    media: Path,
    output_root: Optional[Path],
    device: str,
    skip_music: bool,
    no_sync: bool,
    clean_intermediates: bool,
    purge_all_on_success: bool,
    write_transcript_flag: bool,
    max_chars: int,
    max_lines: int,
    max_duration: float,
    min_gap: float,
    corrections_path: Optional[Path],
) -> int:
    work = _work_dir_for(media)
    work.mkdir(parents=True, exist_ok=True)

    # 1) Preprocess → returns path to final audio and music segments
    # We enable normalize by default; denoise optional (off by default here)
    audio_path, music_segments = preprocess_pipeline(
        input_path=str(media),
        outdir=str(work),
        track_index=None,
        denoise=False,
        denoise_aggressiveness=0.85,
        normalize=True,
        music_threshold=0.5,
        stem=media.stem,
    )

    # 2) Transcribe+align → write <stem>.transcript.json & <stem>.segments.json in work
    stem = media.stem
    segments_path = transcribe_and_align(
        audio_path=audio_path,
        outdir=str(work),
        model="large-v3-turbo",
        compute_type="float32",
        device=device,
        batch_size=8,
        beam_size=5,
        music_segments=music_segments,
        skip_music=skip_music,
        spellcheck=False,
        stem=stem,  # added parameter supported by our transcribe.py change
    )

    # 3) Format → enforce limits and write SRT/TXT to output folder (default: media.parent)
    out_srt, out_txt, stem = _resolve_outputs(media, output_root)
    subs = load_segments(Path(segments_path))

    # Apply corrections first if provided
    if corrections_path and corrections_path.exists():
        rules = load_corrections(corrections_path)
        for ev in subs.events:
            fixed = apply_corrections(ev.plaintext, rules)
            ev.text = fixed.replace("\n", "\\N")

    # Enforce limits and write outputs
    enforce_limits(
        subs,
        max_chars=max_chars,
        max_lines=max_lines,
        max_duration=max_duration,
        min_gap=min_gap,
    )
    out_txt_path = out_txt if write_transcript_flag else None
    write_outputs(subs, out_srt, out_txt_path)

    # Mark success
    success_flag = work / "SUCCESS"
    success_flag.write_text("ok", encoding="utf-8")

    # Cleanup policy: only run when success flag exists
    if success_flag.exists():
        if purge_all_on_success:
            try:
                out_srt.unlink(missing_ok=True)
                if out_txt_path:
                    out_txt_path.unlink(missing_ok=True)
            except Exception:
                pass
        if clean_intermediates:
            # delete the work dir tree
            import shutil
            try:
                shutil.rmtree(work)
            except Exception:
                pass

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="subwhisper - single command pipeline")
    p.add_argument("--input", required=True, help="Input media file or directory")
    p.add_argument("--output-root", help="Root directory for outputs (default: same as input file parent for single-file; for folder, mirrors per file parent)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device for WhisperX")
    p.add_argument("--skip-music", action="store_true", help="Skip segments overlapping detected music")
    p.add_argument("--no-sync", action="store_true", help="Skip QC sync checks (handled in experiment flows)")
    p.add_argument("--clean-intermediates", action="store_true", default=True, help="Delete working dir after success")
    p.add_argument("--purge-all-on-success", action="store_true", help="Delete final outputs as well (not recommended)")
    p.add_argument("--write-transcript", action="store_true", help="Also write a plain text transcript")
    p.add_argument("--max-chars", type=int, default=45)
    p.add_argument("--max-lines", type=int, default=2)
    p.add_argument("--max-duration", type=float, default=6.0)
    p.add_argument("--min-gap", type=float, default=0.15)
    p.add_argument("--corrections", help="Path to JSON/YAML corrections file")
    args = p.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output_root).resolve() if args.output_root else None
    corrections_path = Path(args.corrections).resolve() if args.corrections else None

    if input_path.is_file():
        try:
            return _process_one(
                media=input_path.resolve(),
                output_root=output_root,
                device=args.device,
                skip_music=args.skip_music,
                no_sync=args.no_sync,
                clean_intermediates=args.clean_intermediates,
                purge_all_on_success=args.purge_all_on_success,
                write_transcript_flag=args.write_transcript,
                max_chars=args.max_chars,
                max_lines=args.max_lines,
                max_duration=args.max_duration,
                min_gap=args.min_gap,
                corrections_path=corrections_path,
            )
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
            return 1

    if input_path.is_dir():
        rc = 0
        for media in sorted(input_path.rglob("*")):
            if media.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".m4a", ".mp3", ".flac"}:
                try:
                    code = _process_one(
                        media=media.resolve(),
                        output_root=output_root,  # if None, defaults to media.parent
                        device=args.device,
                        skip_music=args.skip_music,
                        no_sync=args.no_sync,
                        clean_intermediates=args.clean_intermediates,
                        purge_all_on_success=args.purge_all_on_success,
                        write_transcript_flag=args.write_transcript,
                        max_chars=args.max_chars,
                        max_lines=args.max_lines,
                        max_duration=args.max_duration,
                        min_gap=args.min_gap,
                        corrections_path=corrections_path,
                    )
                    if code != 0:
                        rc = code
                except Exception:
                    print(f"[ERROR] {media}", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    rc = 1
        return rc

    print("Input path does not exist.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
