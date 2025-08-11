import sys
import types
import pytest


_stub = types.SimpleNamespace(set_audio_backend=lambda *a, **k: None)
sys.modules.setdefault("torchaudio", _stub)


@pytest.fixture(autouse=True)
def _mock_torchaudio(monkeypatch):
    """Provide a dummy torchaudio module for tests.

    The real torchaudio backend selection can emit warnings during import on
    systems without the optional dependencies installed.  Tests stub out the
    module to keep the output clean and deterministic.
    """

    monkeypatch.setitem(sys.modules, "torchaudio", _stub)
