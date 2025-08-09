import json
import logging
import os
import subprocess
from typing import List, Optional, Tuple
import shutil

import librosa
import noisereduce as nr
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_english_track(video_path: str) -> int:
    """Return index of English audio track in a media file.

    Parameters
    ----------
    video_path: str
        Path to the input media file.

    Returns
    -------
    int
        Index of the audio track marked as English. Defaults to ``0`` when an
        English track cannot be determined.
    """
    if shutil.which("ffprobe") is None:  # pragma: no cover - environment check
        logger.error("ffprobe not found; please install FFmpeg")
        raise RuntimeError("ffprobe missing")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index:stream_tags=language",
        "-of",
        "json",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams", [])
        for idx, stream in enumerate(streams):
            language = stream.get("tags", {}).get("language", "").lower()
            if language == "eng":
                logger.info("Selected English audio stream %s", idx)
                return idx
        if streams:
            logger.info(
                "No English stream found; defaulting to first audio stream %s",
                0,
            )
        else:
            logger.warning("ffprobe reported no audio streams; defaulting to 0")
    except subprocess.CalledProcessError as exc:
        logger.error("ffprobe failed: %s", exc.stderr.strip())
    except FileNotFoundError:
        logger.error("ffprobe not found; ensure FFmpeg is installed")
    except json.JSONDecodeError:
        logger.error("Could not parse ffprobe output")
    return 0


def extract_audio(
    video_path: str, output_path: str, track_index: Optional[int]
) -> str:
    """Extract a mono 16 kHz WAV from a media file.

    Parameters
    ----------
    video_path: str
        Path to the source media file.
    output_path: str
        Destination path of the extracted audio WAV.
    track_index: int | None
        Index of the audio stream to extract. ``None`` defaults to ``0``.

    Returns
    -------
    str
        Path to the resulting WAV file.
    """
    stream_index = track_index if track_index is not None else 0
    logger.info(
        "Extracting audio from %s (track %s) to %s",
        video_path,
        stream_index,
        output_path,
    )
    if shutil.which("ffmpeg") is None:  # pragma: no cover - environment check
        logger.error("ffmpeg not found; please install FFmpeg")
        raise RuntimeError("ffmpeg missing")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-map",
        f"0:a:{stream_index}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Audio successfully extracted to %s", output_path)
    except subprocess.CalledProcessError as exc:
        err = exc.stderr.decode() if hasattr(exc.stderr, "decode") else str(exc)
        logger.error("ffmpeg failed: %s", err.strip())
        raise RuntimeError("ffmpeg audio extraction failed") from exc
    except FileNotFoundError:
        logger.error("ffmpeg not found; ensure it is installed and on PATH")
        raise RuntimeError("ffmpeg missing")
    return output_path


def denoise_audio(audio_path: str, output_path: str, aggressiveness: float = 0.85) -> str:
    """Denoise an audio file using spectral gating.

    Parameters
    ----------
    audio_path: str
        Path to the input audio file.
    output_path: str
        Destination path for the denoised audio WAV.
    aggressiveness: float, optional
        Value between 0 and 1 controlling how strongly noise is reduced; higher
        values remove more noise at the risk of artifacts. Defaults to ``0.85``.

    Returns
    -------
    str
        Path to the denoised WAV file.
    """
    logger.info("Loading audio from %s", audio_path)
    try:
        data, rate = sf.read(audio_path)
        logger.info("Applying noise reduction (aggressiveness=%s)", aggressiveness)
        reduced = nr.reduce_noise(y=data, sr=rate, prop_decrease=aggressiveness)
        logger.info("Writing denoised audio to %s", output_path)
        sf.write(output_path, reduced, rate)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Noise reduction failed: %s", exc)
        raise RuntimeError("noise reduction failed") from exc
    return output_path


def normalize_audio(input_wav: str, output_wav: str, enabled: bool = True) -> str:
    """Normalize an audio file using ``ffmpeg`` or copy when disabled.

    Parameters
    ----------
    input_wav: str
        Path to the source WAV file.
    output_wav: str
        Destination path for the normalized (or copied) WAV.
    enabled: bool, optional
        Whether to perform loudness normalization. Defaults to ``True``.

    Returns
    -------
    str
        Path to the resulting WAV file.
    """
    if not enabled:
        logger.info(
            "Normalization disabled; copying %s to %s", input_wav, output_wav
        )
        try:
            shutil.copyfile(input_wav, output_wav)
        except OSError as exc:  # pragma: no cover - defensive
            logger.error("File copy failed: %s", exc)
            raise RuntimeError("audio copy failed") from exc
        return output_wav

    if shutil.which("ffmpeg") is None:  # pragma: no cover - environment check
        logger.error("ffmpeg not found; please install FFmpeg")
        raise RuntimeError("ffmpeg missing")
    logger.info("Normalizing audio %s to %s", input_wav, output_wav)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_wav,
        "-af",
        "loudnorm",
        output_wav,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Audio normalized to %s", output_wav)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - defensive
        err = exc.stderr.decode() if hasattr(exc.stderr, "decode") else str(exc)
        logger.error("ffmpeg normalization failed: %s", err.strip())
        raise RuntimeError("ffmpeg audio normalization failed") from exc
    except FileNotFoundError:
        logger.error("ffmpeg not found; ensure it is installed and on PATH")
        raise RuntimeError("ffmpeg missing")
    return output_wav


def detect_music_segments(
    audio_path: str, segments_file: str, threshold: float = 0.5
) -> List[Tuple[float, float]]:
    """Detect likely music segments within an audio file.

    The function separates the harmonic and percussive components of the
    waveform using :func:`librosa.effects.hpss`. It then computes the
    percussive-to-harmonic energy ratio over short windows and returns the
    start and end times of intervals whose ratio exceeds ``threshold``. All
    detected segments are also written to ``segments_file``.

    Parameters
    ----------
    audio_path: str
        Path to the input audio file.
    segments_file: str
        Path where ``music_segments.json`` will be written.
    threshold: float, optional
        Minimum percussive-to-harmonic energy ratio to qualify as a music
        segment. Defaults to ``0.5``.

    Returns
    -------
    list of tuple(float, float)
        A list of ``(start, end)`` pairs in seconds marking detected music
        regions.
    """

    try:
        logger.info("Loading audio from %s", audio_path)
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        logger.info("Separating harmonic and percussive components")
        y_harm, y_perc = librosa.effects.hpss(y)

        frame_length = 2048
        hop_length = 512
        logger.info("Computing RMS energies")
        harm_rms = librosa.feature.rms(
            y=y_harm, frame_length=frame_length, hop_length=hop_length
        )[0]
        perc_rms = librosa.feature.rms(
            y=y_perc, frame_length=frame_length, hop_length=hop_length
        )[0]

        ratio = perc_rms / (harm_rms + 1e-10)
        mask = ratio > threshold

        segments: List[Tuple[float, float]] = []
        start_time: Optional[float] = None
        for idx, is_music in enumerate(mask):
            time = librosa.frames_to_time(idx, sr=sr, hop_length=hop_length)
            if is_music and start_time is None:
                start_time = time
            elif not is_music and start_time is not None:
                segments.append((start_time, time))
                start_time = None

        if start_time is not None:
            end_time = librosa.frames_to_time(
                len(mask), sr=sr, hop_length=hop_length
            )
            segments.append((start_time, end_time))

        logger.info("Detected %s music segments", len(segments))
        with open(segments_file, "w", encoding="utf-8") as fh:
            json.dump(segments, fh)
        return segments
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Music segment detection failed: %s", exc)
        raise RuntimeError("music segment detection failed") from exc


def preprocess_pipeline(
    input_path: str,
    outdir: str,
    track_index: Optional[int] = None,
    denoise: bool = False,
    denoise_aggressiveness: float = 0.85,
    normalize: bool = False,
    music_threshold: float = 0.5,
    stem: Optional[str] = None,
) -> Tuple[str, List[Tuple[float, float]]]:
    """Run the full preprocessing pipeline.
    
    All intermediate files and the resulting ``music_segments.json`` are written
    to ``outdir``. When ``stem`` is provided, filenames are prefixed with
    ``"<stem>."``.

    Parameters
    ----------
    input_path: str
        Source media file to process.
    outdir: str
        Directory where intermediate and output files are written.
    track_index: int | None, optional
        Specific audio track to use. When ``None``, the English track is
        auto-detected.
    denoise: bool, optional
        Apply noise reduction when ``True``.
    denoise_aggressiveness: float, optional
        Strength for noise reduction passed to :func:`denoise_audio`. This value
        ranges from 0 to 1 and is used only when ``denoise`` is ``True``.
    normalize: bool, optional
        Apply loudness normalization when ``True``.
    music_threshold: float, optional
        Threshold passed to :func:`detect_music_segments`. Detected segments are
        saved to ``music_segments.json`` inside ``outdir``.
    stem: str | None, optional
        Base name for generated files. When provided, intermediate and output
        files are prefixed with ``"<stem>."``.

    Returns
    -------
    tuple
        A ``(audio_path, segments)`` pair with the final audio file and list of
        music segments.
    """

    if not os.path.isfile(input_path):
        logger.error("Input file does not exist: %s", input_path)
        raise FileNotFoundError(input_path)

    os.makedirs(outdir, exist_ok=True)

    track = track_index if track_index is not None else find_english_track(input_path)
    raw_audio = os.path.join(outdir, f"{stem}.audio.wav" if stem else "audio.wav")
    audio_path = extract_audio(input_path, raw_audio, track)

    if denoise:
        denoised = os.path.join(outdir, f"{stem}.denoised.wav" if stem else "denoised.wav")
        audio_path = denoise_audio(
            audio_path, denoised, aggressiveness=denoise_aggressiveness
        )

    if normalize:
        normalized = os.path.join(
            outdir, f"{stem}.normalized.wav" if stem else "normalized.wav"
        )
        audio_path = normalize_audio(audio_path, normalized, enabled=True)

    segments_file = os.path.join(
        outdir, f"{stem}.music_segments.json" if stem else "music_segments.json"
    )
    segments = detect_music_segments(audio_path, segments_file, threshold=music_threshold)

    return audio_path, segments


def main() -> None:
    """Entry point for command-line execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess audio for SubWhisper")
    parser.add_argument("--input", required=True, help="Input media file")
    parser.add_argument(
        "--track",
        type=int,
        help="Audio track index to extract; auto-detect English when omitted",
    )
    parser.add_argument(
        "--denoise", action="store_true", help="Apply noise reduction"
    )
    parser.add_argument(
        "--denoise-aggressive",
        type=float,
        default=0.85,
        help=(
            "Noise reduction aggressiveness between 0 and 1 (higher removes "
            "more noise); used with --denoise"
        ),
    )
    parser.add_argument(
        "--normalize", action="store_true", help="Apply loudness normalization"
    )
    parser.add_argument(
        "--music-threshold",
        type=float,
        default=0.5,
        help=(
            "Threshold for music detection; detected segments are saved to "
            "music_segments.json in --outdir"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="preproc",
        help=(
            "Directory for processed outputs; music_segments.json will be "
            "written here"
        ),
    )
    parser.add_argument(
        "--stem",
        help="Base name for generated files",
    )

    args = parser.parse_args()

    try:
        preprocess_pipeline(
            input_path=args.input,
            outdir=args.outdir,
            track_index=args.track,
            denoise=args.denoise,
            denoise_aggressiveness=args.denoise_aggressive,
            normalize=args.normalize,
            music_threshold=args.music_threshold,
            stem=args.stem,
        )
    except Exception as exc:  # pragma: no cover - CLI
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
