import sys
import pathlib
import json
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corrections import apply_corrections, load_corrections


def test_apply_corrections_basic(caplog):
    rules = {"hello world": "hi earth", "cat": "dog"}
    caplog.set_level("DEBUG")
    result = apply_corrections("hello world cat", rules)
    assert result == "hi earth dog"
    # original line is logged when debug enabled
    assert "hello world cat" in caplog.text


def test_apply_corrections_regex(caplog):
    rules = {r"c[ao]t": "dog"}
    caplog.set_level("DEBUG")
    result = apply_corrections("the cat sat on the cot", rules, use_regex=True)
    assert result == "the dog sat on the dog"
    assert "the cat sat on the cot" in caplog.text


def test_apply_corrections_multiline(caplog):
    rules = {"foo": "bar"}
    caplog.set_level("DEBUG")
    result = apply_corrections("foo\nfoo", rules)
    assert result == "bar\nbar"
    assert caplog.text.count("foo") >= 2


def test_load_corrections_json_yaml(tmp_path):
    json_path = tmp_path / "rules.json"
    json_path.write_text(json.dumps({"foo": "bar"}))
    yaml_path = tmp_path / "rules.yaml"
    yaml_path.write_text("baz: qux\n")

    assert load_corrections(json_path) == {"foo": "bar"}
    assert load_corrections(yaml_path) == {"baz": "qux"}
