import argparse
import sys
import traceback
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Set

from audio_backend import setup_torchaudio_backend

setup_torchaudio_backend()

from preproc import preprocess_pipeline
from transcribe import transcribe_and_align
from subtitle_pipeline import load_segments, enforce_limits, write_outputs
from corrections import load_corrections, apply_corrections
from run_state import (
    compute_source_fingerprint,
    load_manifest,
    stage_complete,
    is_stage_reusable,
    acquire_lock,
    release_lock,
    clean_old_manifests,
    CACHE_DIR,
    MANIFEST_SUFFIX,
)
from vad import warn_if_incompatible_pyannote

logger = logging.getLogger(__name__)


def _resolve_outputs(input_path: Path, output_root: Optional[Path]) -> Tuple[Path, Path, str]:
    stem = input_path.stem
    if output_root:
        out_dir = output_root / stem
    else:
        out_dir = input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_srt = out_dir / f"{stem}.srt"
    out_txt = out_dir / f"{stem}.txt"
    return out_srt, out_txt, stem


def _work_dir_for(input_path: Path) -> Path:
    cache_root = input_path.parent / CACHE_DIR
    return cache_root / input_path.stem


def _process_one(
    media: Path,
    output_root: Optional[Path],
    device: str,
    skip_music: bool,
    beam_size: Optional[int],
    clean_intermediates: bool,
    purge_all_on_success: bool,
    write_transcript_flag: bool,
    max_chars: int,
    max_lines: int,
    max_duration: float,
    min_gap: float,
    corrections_path: Optional[Path],
    resume: str,
    force: bool,
    resume_clean_days: Optional[int],
    cleaned_roots: Set[Path],
) -> int:
    work = _work_dir_for(media)
    work.mkdir(parents=True, exist_ok=True)
    cache_root = work.parent
    manifest_path = cache_root / f"{media.stem}{MANIFEST_SUFFIX}"

    if resume_clean_days and cache_root not in cleaned_roots:
        removed = clean_old_manifests(cache_root, resume_clean_days)
        if removed:
            logger.info("resume: cleaned %s old manifests", removed)
        cleaned_roots.add(cache_root)

    acquire_lock(manifest_path)
    try:
        # Compute source fingerprint and load manifest
        _ = compute_source_fingerprint(str(media))
        manifest = load_manifest(media.parent, media.stem)
        if force or resume == "off":
            manifest = {}

        allow_resume = resume != "off" and not force
        downstream_valid = True

        audio_path: Optional[str] = None
        music_segments = None

        # Stage: preprocess
        preproc_params = {
            "denoise": False,
            "denoise_aggressiveness": 0.85,
            "normalize": True,
            "music_threshold": 0.5,
        }
        reusable, outputs = (
            is_stage_reusable(media.parent, media.stem, "preproc", str(media), preproc_params)
            if allow_resume and downstream_valid
            else (False, [])
        )
        if reusable:
            logger.info("resume: preproc valid; skipping")
            audio_path = outputs[0]
            if len(outputs) > 1:
                try:
                    with open(outputs[1], "r", encoding="utf-8") as fh:
                        music_segments = json.load(fh)
                except Exception:
                    music_segments = None
        else:
            logger.info("resume: preproc invalid; running")
            pre_out = preprocess_pipeline(
                input_path=str(media),
                outdir=str(work),
                track_index=None,
                denoise=False,
                denoise_aggressiveness=0.85,
                normalize=True,
                music_threshold=0.5,
                stem=media.stem,
            )
            audio_path = pre_out.get("normalized_wav") or pre_out.get("audio_wav")
            outputs = [audio_path]
            if pre_out.get("music_segments"):
                with open(pre_out["music_segments"], "r", encoding="utf-8") as fh:
                    music_segments = json.load(fh)
                outputs.append(pre_out["music_segments"])
            stage_complete(media.parent, media.stem, "preproc", str(media), preproc_params, outputs)
        downstream_valid = downstream_valid and reusable

        # Stage: transcribe
        transcribe_params = {
            "model": "large-v3-turbo",
            "compute_type": "float32",
            "device": device,
            "batch_size": 8,
            "skip_music": skip_music,
        }
        if beam_size is not None:
            transcribe_params["beam_size"] = beam_size
        reusable, outputs = (
            is_stage_reusable(media.parent, media.stem, "transcribe", str(media), transcribe_params)
            if allow_resume and downstream_valid
            else (False, [])
        )
        if reusable:
            logger.info("resume: transcribe valid; skipping")
            trans_out = {
                "transcript_json": outputs[0],
                "segments_json": outputs[1] if len(outputs) > 1 else None,
            }
        else:
            logger.info("resume: transcribe invalid; running")
            trans_out = transcribe_and_align(
                audio_path=audio_path,
                outdir=str(work),
                model="large-v3-turbo",
                compute_type="float32",
                device=device,
                batch_size=8,
                beam_size=beam_size,
                music_segments=music_segments,
                skip_music=skip_music,
                spellcheck=False,
                stem=media.stem,
            )
            outputs = [trans_out["transcript_json"], trans_out["segments_json"]]
            stage_complete(media.parent, media.stem, "transcribe", str(media), transcribe_params, outputs)
        downstream_valid = downstream_valid and reusable

        # Stage: format
        out_srt, out_txt, stem = _resolve_outputs(media, output_root)
        out_txt_path = out_txt if write_transcript_flag else None
        corrections_fp = (
            compute_source_fingerprint(str(corrections_path))
            if corrections_path and corrections_path.exists()
            else None
        )
        format_params = {
            "max_chars": max_chars,
            "max_lines": max_lines,
            "max_duration": max_duration,
            "min_gap": min_gap,
            "write_transcript": write_transcript_flag,
            "output_root": str(output_root) if output_root else None,
            "corrections": corrections_fp,
        }
        reusable, outputs = (
            is_stage_reusable(media.parent, media.stem, "format", str(media), format_params)
            if allow_resume and downstream_valid
            else (False, [])
        )
        if reusable:
            logger.info("resume: format valid; skipping")
        else:
            logger.info("resume: format invalid; running")
            subs = load_segments(Path(trans_out["segments_json"]))
            if corrections_path and corrections_path.exists():
                rules = load_corrections(corrections_path)
                for ev in subs.events:
                    fixed = apply_corrections(ev.plaintext, rules)
                    ev.text = fixed.replace("\n", "\\N")
            enforce_limits(
                subs,
                max_chars=max_chars,
                max_lines=max_lines,
                max_duration=max_duration,
                min_gap=min_gap,
            )
            write_outputs(subs, out_srt, out_txt_path)
            outputs = [str(out_srt)]
            if out_txt_path:
                outputs.append(str(out_txt_path))
            stage_complete(media.parent, media.stem, "format", str(media), format_params, outputs)

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
                import shutil
                try:
                    shutil.rmtree(work)
                except Exception:
                    pass

        return 0
    finally:
        release_lock(manifest_path)


def main() -> int:
    warn_if_incompatible_pyannote()
    p = argparse.ArgumentParser(description="subwhisper - single command pipeline")
    p.add_argument("--input", required=True, help="Input media file or directory")
    p.add_argument("--output-root", help="Root directory for outputs (default: same as input file parent for single-file; for folder, mirrors per file parent)")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device for WhisperX")
    p.add_argument("--skip-music", action="store_true", help="Skip segments overlapping detected music")
    p.add_argument(
        "--beam-size",
        type=int,
        default=None,
        help="Beam size for Whisper decoder (default: model default)",
    )
    p.add_argument(
        "--clean-intermediates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete working dir after success (use --no-clean-intermediates to keep)",
    )
    p.add_argument("--purge-all-on-success", action="store_true", help="Delete final outputs as well (not recommended)")
    p.add_argument("--write-transcript", action="store_true", help="Also write a plain text transcript")
    p.add_argument("--max-chars", type=int, default=45)
    p.add_argument("--max-lines", type=int, default=2)
    p.add_argument("--max-duration", type=float, default=6.0)
    p.add_argument("--min-gap", type=float, default=0.15)
    p.add_argument("--corrections", help="Path to JSON/YAML corrections file")
    p.add_argument("--resume", choices=["auto", "off"], default="auto", help="Resume from previous runs")
    p.add_argument("--force", action="store_true", help="Force recomputation, ignore manifests")
    p.add_argument("--resume-clean", type=int, metavar="DAYS", help="Remove manifests older than DAYS")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)

    clean_intermediates = args.clean_intermediates

    input_path = Path(args.input)
    output_root = Path(args.output_root).resolve() if args.output_root else None
    corrections_path = Path(args.corrections).resolve() if args.corrections else None

    cleaned_roots: Set[Path] = set()

    if input_path.is_file():
        try:
            return _process_one(
                media=input_path.resolve(),
                output_root=output_root,
                device=args.device,
                skip_music=args.skip_music,
                beam_size=args.beam_size,
                clean_intermediates=clean_intermediates,
                purge_all_on_success=args.purge_all_on_success,
                write_transcript_flag=args.write_transcript,
                max_chars=args.max_chars,
                max_lines=args.max_lines,
                max_duration=args.max_duration,
                min_gap=args.min_gap,
                corrections_path=corrections_path,
                resume=args.resume,
                force=args.force,
                resume_clean_days=args.resume_clean,
                cleaned_roots=cleaned_roots,
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
                        beam_size=args.beam_size,
                        clean_intermediates=clean_intermediates,
                        purge_all_on_success=args.purge_all_on_success,
                        write_transcript_flag=args.write_transcript,
                        max_chars=args.max_chars,
                        max_lines=args.max_lines,
                        max_duration=args.max_duration,
                        min_gap=args.min_gap,
                        corrections_path=corrections_path,
                        resume=args.resume,
                        force=args.force,
                        resume_clean_days=args.resume_clean,
                        cleaned_roots=cleaned_roots,
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
