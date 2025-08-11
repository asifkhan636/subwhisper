from unittest.mock import MagicMock
import platform
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_backend import setup_torchaudio_backend


def test_setup_backend_windows(monkeypatch):
    mock = MagicMock()
    stub = types.SimpleNamespace(set_audio_backend=mock)
    monkeypatch.setitem(sys.modules, "torchaudio", stub)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    setup_torchaudio_backend()
    mock.assert_called_once_with("soundfile")


def test_setup_backend_non_windows(monkeypatch):
    mock = MagicMock()
    stub = types.SimpleNamespace(set_audio_backend=mock)
    monkeypatch.setitem(sys.modules, "torchaudio", stub)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    setup_torchaudio_backend()
    mock.assert_not_called()
