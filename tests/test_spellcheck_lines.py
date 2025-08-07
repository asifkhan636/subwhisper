import sys
import types
import pysubs2
import subtitle_pipeline


def test_spellcheck_lines(monkeypatch):
    class DummyTool:
        instances = 0

        def __init__(self, lang):
            DummyTool.instances += 1

        def correct(self, text):
            return text.replace("teh", "the")

    dummy_module = types.SimpleNamespace(LanguageTool=DummyTool)
    monkeypatch.setitem(sys.modules, "language_tool_python", dummy_module)

    subs = pysubs2.SSAFile()
    subs.events.append(pysubs2.SSAEvent(start=0, end=1000, text="teh cat"))
    subs.events.append(pysubs2.SSAEvent(start=1000, end=2000, text="teh dog"))

    subtitle_pipeline.spellcheck_lines(subs)

    assert [e.plaintext for e in subs.events] == ["the cat", "the dog"]
    assert DummyTool.instances == 1
