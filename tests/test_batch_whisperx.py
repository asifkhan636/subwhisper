from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import batch_whisperx


def test_translate_whisperx_command():
    steps = batch_whisperx._translate_whisperx_command(
        "whisperx audio.wav --model medium.en --device cpu --compute_type int8 "
        "--batch_size 1 --beam_size 3 --output_dir out --output_format srt "
        "--max_line_width 45 --max_line_count 2 --language en --task transcribe"
    )

    assert len(steps) == 2
    assert steps[0][0] == sys.executable
    assert steps[0][1].endswith("transcribe.py")
    assert steps[0][2:] == [
        "audio.wav",
        "--outdir",
        "out",
        "--model",
        "medium.en",
        "--device",
        "cpu",
        "--language",
        "en",
        "--compute-type",
        "int8",
        "--batch-size",
        "1",
        "--beam-size",
        "3",
    ]
    assert steps[1][0] == sys.executable
    assert steps[1][1].endswith("subtitle_pipeline.py")
    assert "--max-chars" in steps[1]
    assert "--max-lines" in steps[1]


def test_translate_rejects_unsupported_whisperx_flag():
    with pytest.raises(ValueError, match="Unsupported WhisperX-only flag"):
        batch_whisperx._translate_whisperx_command(
            "whisperx audio.wav --output_dir out --output_format srt --align_model foo"
        )


def test_main_translates_commands_file(monkeypatch, tmp_path):
    commands_file = tmp_path / "commands.txt"
    commands_file.write_text(
        "whisperx {audio} --model medium.en --device cpu --compute_type int8 "
        "--batch_size 1 --output_dir {outdir}/cpu --output_format srt "
        "--max_line_width 45 --max_line_count 2 --language en --task transcribe\n",
        encoding="utf-8",
    )

    calls = []

    def fake_run_command(cmd, description):
        calls.append((list(cmd), description))

    monkeypatch.setattr(batch_whisperx, "run_command", fake_run_command)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batch_whisperx.py",
            "video.mkv",
            "--audio",
            "audio.wav",
            "--outdir",
            str(tmp_path / "out"),
            "--commands-file",
            str(commands_file),
        ],
    )

    batch_whisperx.main()

    assert calls[0][1] == "ffmpeg extraction"
    assert calls[1][0][1].endswith("transcribe.py")
    assert calls[1][0][2:] == [
        "audio.wav",
        "--outdir",
        str(tmp_path / "out" / "cpu"),
        "--model",
        "medium.en",
        "--device",
        "cpu",
        "--language",
        "en",
        "--compute-type",
        "int8",
        "--batch-size",
        "1",
    ]
    assert calls[2][0][1].endswith("subtitle_pipeline.py")
