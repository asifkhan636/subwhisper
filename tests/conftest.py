import sys
import types
import pytest


_torchaudio_stub = types.SimpleNamespace(set_audio_backend=lambda *a, **k: None)
sys.modules.setdefault("torchaudio", _torchaudio_stub)

# Provide a minimal ``torch`` stub required by tests and dependencies like
# ``noisereduce``.
_torch_stub = types.ModuleType("torch")
_torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
class _NoGrad:
    def __enter__(self, *a, **k):
        return None
    def __exit__(self, *a, **k):
        return False
    def __call__(self, func):
        return func
_torch_stub.no_grad = lambda: _NoGrad()

_torch_nn = types.ModuleType("torch.nn")
_torch_nn.functional = types.SimpleNamespace(
    conv1d=lambda *a, **k: None, conv2d=lambda *a, **k: None
)

_torch_types = types.ModuleType("torch.types")
_torch_types.Number = float

sys.modules.setdefault("torch", _torch_stub)
sys.modules.setdefault("torch.nn", _torch_nn)
sys.modules.setdefault("torch.nn.functional", _torch_nn.functional)
sys.modules.setdefault("torch.types", _torch_types)
_torch_stub.nn = _torch_nn


@pytest.fixture(autouse=True)
def _mock_torchaudio(monkeypatch):
    """Provide a dummy torchaudio module for tests.

    The real torchaudio backend selection can emit warnings during import on
    systems without the optional dependencies installed.  Tests stub out the
    module to keep the output clean and deterministic.
    """

    monkeypatch.setitem(sys.modules, "torchaudio", _torchaudio_stub)
    monkeypatch.setitem(sys.modules, "torch", _torch_stub)
    monkeypatch.setitem(sys.modules, "torch.nn", _torch_nn)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", _torch_nn.functional)
    monkeypatch.setitem(sys.modules, "torch.types", _torch_types)
