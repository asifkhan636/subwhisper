import sys
import pathlib
from types import SimpleNamespace
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from format_subtitles import format_subtitles


def test_merge_and_line_split():
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 1.0},
            ],
        },
        {
            "start": 1.05,
            "end": 2.0,
            "text": "how are you",
            "words": [
                {"word": "how", "start": 1.05, "end": 1.3},
                {"word": "are", "start": 1.3, "end": 1.6},
                {"word": "you", "start": 1.6, "end": 2.0},
            ],
        },
    ]
    cfg = SimpleNamespace(
        max_line_length=11,
        max_line_count=2,
        max_duration=10.0,
        skip_music=False,
        gap=0.3,
    )
    blocks = format_subtitles(segments, cfg)
    assert blocks == [
        {
            "start": 0.0,
            "end": 2.0,
            "lines": ["hello world", "how are you"],
        }
    ]


def test_max_duration_and_gap():
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "one two three four",
            "words": [
                {"word": "one", "start": 0.0, "end": 0.5},
                {"word": "two", "start": 0.5, "end": 1.0},
                {"word": "three", "start": 1.0, "end": 1.5},
                {"word": "four", "start": 1.5, "end": 2.0},
            ],
        },
        {
            "start": 2.05,
            "end": 4.5,
            "text": "five six seven eight",
            "words": [
                {"word": "five", "start": 2.05, "end": 2.5},
                {"word": "six", "start": 2.5, "end": 3.0},
                {"word": "seven", "start": 3.0, "end": 3.5},
                {"word": "eight", "start": 3.5, "end": 4.5},
            ],
        },
    ]
    cfg = SimpleNamespace(
        max_line_length=20,
        max_line_count=2,
        max_duration=2.0,
        skip_music=False,
        gap=0.1,
    )
    blocks = format_subtitles(segments, cfg)
    assert [b["lines"] for b in blocks] == [
        ["one two three four"],
        ["five six seven"],
        ["eight"],
    ]
    assert blocks[0]["end"] == pytest.approx(2.0)
    # second block starts at least 0.1s after first ends
    assert blocks[1]["start"] == pytest.approx(2.1)
    # third block also respects gap
    assert blocks[2]["start"] == pytest.approx(3.6)


def test_skip_music_segments():
    segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "text": "speech",
            "words": [{"word": "speech", "start": 0.0, "end": 1.0}],
        },
        {
            "start": 1.5,
            "end": 2.5,
            "text": "music",
            "is_music": True,
            "words": [],
        },
    ]
    cfg = SimpleNamespace(
        max_line_length=20,
        max_line_count=2,
        max_duration=5.0,
        skip_music=True,
        gap=0.1,
    )
    blocks = format_subtitles(segments, cfg)
    assert blocks == [{"start": 0.0, "end": 1.0, "lines": ["speech"]}]
