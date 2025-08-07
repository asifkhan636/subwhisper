import sys
import pathlib
import pysubs2
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subtitle_pipeline import enforce_limits


def make_event(start, end, text):
    return pysubs2.SSAEvent(start=start, end=end, text=text)


def test_wrap_and_truncate_lines():
    subs = pysubs2.SSAFile()
    subs.events.append(make_event(0, 1000, "hello world it's me"))
    enforce_limits(subs, max_chars=5, max_lines=2, max_duration=10.0, min_gap=0.0)
    assert subs.events[0].text == "hello\\Nworld"


def test_clip_duration():
    subs = pysubs2.SSAFile()
    subs.events.append(make_event(0, 7000, "long"))
    enforce_limits(subs, max_chars=20, max_lines=1, max_duration=2.0, min_gap=0.0)
    assert subs.events[0].end == 2000


def test_insert_gap():
    subs = pysubs2.SSAFile()
    subs.events.append(make_event(0, 1000, "first"))
    subs.events.append(make_event(1100, 2000, "second"))
    enforce_limits(subs, max_chars=20, max_lines=1, max_duration=5.0, min_gap=0.5)
    assert subs.events[1].start == 1500
