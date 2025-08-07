import json
import logging
import subprocess
from typing import Optional

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
