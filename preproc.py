import json
import logging
import subprocess
from typing import List, Optional, Tuple
import shutil

import librosa
import noisereduce as nr
import soundfile as sf

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
        Value between 0 and 1 controlling noise reduction strength.

    Returns
    -------
    str
        Path to the denoised WAV file.
    """
    logger.info("Loading audio from %s", audio_path)
    data, rate = sf.read(audio_path)
    logger.info("Applying noise reduction (aggressiveness=%s)", aggressiveness)
    reduced = nr.reduce_noise(y=data, sr=rate, prop_decrease=aggressiveness)
    logger.info("Writing denoised audio to %s", output_path)
    sf.write(output_path, reduced, rate)
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
    return output_wav


def detect_music_segments(audio_path: str, threshold: float = 0.5) -> List[Tuple[float, float]]:
    """Detect likely music segments within an audio file.

    The function separates the harmonic and percussive components of the
    waveform using :func:`librosa.effects.hpss`. It then computes the
    percussive-to-harmonic energy ratio over short windows and returns the
    start and end times of intervals whose ratio exceeds ``threshold``. All
    detected segments are also written to ``music_segments.json`` in the
    current working directory.

    Parameters
    ----------
    audio_path: str
        Path to the input audio file.
    threshold: float, optional
        Minimum percussive-to-harmonic energy ratio to qualify as a music
        segment. Defaults to ``0.5``.

    Returns
    -------
    list of tuple(float, float)
        A list of ``(start, end)`` pairs in seconds marking detected music
        regions.
    """

    logger.info("Loading audio from %s", audio_path)
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    logger.info("Separating harmonic and percussive components")
    y_harm, y_perc = librosa.effects.hpss(y)

    frame_length = 2048
    hop_length = 512
    logger.info("Computing RMS energies")
    harm_rms = librosa.feature.rms(y=y_harm, frame_length=frame_length, hop_length=hop_length)[0]
    perc_rms = librosa.feature.rms(y=y_perc, frame_length=frame_length, hop_length=hop_length)[0]

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
        end_time = librosa.frames_to_time(len(mask), sr=sr, hop_length=hop_length)
        segments.append((start_time, end_time))

    logger.info("Detected %s music segments", len(segments))
    with open("music_segments.json", "w", encoding="utf-8") as fh:
        json.dump(segments, fh)
    return segments


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Audio preprocessing utilities")
    subparsers = parser.add_subparsers(dest="command")

    denoise_parser = subparsers.add_parser("denoise", help="Apply noise reduction to an audio file")
    denoise_parser.add_argument("audio_path", help="Path to the input audio file")
    denoise_parser.add_argument(
        "--output",
        default="denoised.wav",
        help="Destination path for the denoised WAV",
    )
    denoise_parser.add_argument(
        "--aggressiveness",
        type=float,
        default=0.85,
        help="Noise reduction aggressiveness between 0 and 1",
    )

    music_parser = subparsers.add_parser(
        "music", help="Detect music segments in an audio file"
    )
    music_parser.add_argument("audio_path", help="Path to the input audio file")
    music_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Percussive to harmonic ratio threshold",
    )

    args = parser.parse_args()
    if args.command == "denoise":
        denoise_audio(args.audio_path, args.output, aggressiveness=args.aggressiveness)
    elif args.command == "music":
        detect_music_segments(args.audio_path, threshold=args.threshold)
