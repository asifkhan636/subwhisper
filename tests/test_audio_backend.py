from unittest.mock import MagicMock
import logging
import os
import platform
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio_backend import setup_torchaudio_backend


def test_setup_backend_windows(monkeypatch):
    mock = MagicMock()
    stub = types.SimpleNamespace(set_audio_backend=mock)
    monkeypatch.delenv("TORCHAUDIO_ENABLE_SOX_IO_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "torchaudio", stub)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    setup_torchaudio_backend()
    mock.assert_called_once_with("soundfile")
    assert os.environ["TORCHAUDIO_ENABLE_SOX_IO_BACKEND"] == "0"


def test_setup_backend_non_windows(monkeypatch):
    mock = MagicMock()
    stub = types.SimpleNamespace(set_audio_backend=mock)
    monkeypatch.delenv("TORCHAUDIO_ENABLE_SOX_IO_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "torchaudio", stub)
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    setup_torchaudio_backend()
    mock.assert_not_called()
    assert "TORCHAUDIO_ENABLE_SOX_IO_BACKEND" not in os.environ


def test_setup_backend_no_warning(monkeypatch, caplog):
    stub = types.SimpleNamespace(set_audio_backend=lambda *a, **k: None)
    monkeypatch.delenv("TORCHAUDIO_ENABLE_SOX_IO_BACKEND", raising=False)
    monkeypatch.setitem(sys.modules, "torchaudio", stub)
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    with caplog.at_level(logging.WARNING):
        setup_torchaudio_backend()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
