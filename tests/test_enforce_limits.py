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
    subs.events.append(make_event(0, 2000, "hello world it's me"))
    enforce_limits(subs, max_chars=5, max_lines=2, max_duration=10.0, min_gap=0.0)
    assert len(subs.events) == 1
    assert subs.events[0].text == "hello\\Nworld"
    assert hasattr(subs.events[0], "event_cps")


def test_split_long_duration():
    subs = pysubs2.SSAFile()
    subs.events.append(make_event(0, 7000, "hi"))
    enforce_limits(subs, max_chars=20, max_lines=1, max_duration=2.0, min_gap=0.0)
    assert all((ev.end - ev.start) <= 2000 for ev in subs.events)


def test_gap_shift_preserves_duration():
    subs = pysubs2.SSAFile()
    subs.events.append(make_event(0, 1000, "first"))
    subs.events.append(make_event(1100, 1600, "second"))
    enforce_limits(subs, max_chars=20, max_lines=1, max_duration=5.0, min_gap=1.0)
    ev = subs.events[1]
    assert ev.start == 2000
    assert ev.end - ev.start == 500


def test_split_high_cps():
    subs = pysubs2.SSAFile()
    subs.events.append(make_event(0, 1000, "a" * 40))
    enforce_limits(subs, max_chars=40, max_lines=1, max_duration=5.0, min_gap=0.0)
    assert len(subs.events) > 1
    assert all(ev.event_cps <= 17 for ev in subs.events)
