import sys
from types import ModuleType
from unittest.mock import Mock

import pytest

from api_backend import Generator


@pytest.fixture
def startup(monkeypatch):
    device = ModuleType("spar3d.device")
    device.select_device = Mock(return_value="cpu")
    system = ModuleType("spar3d.system")
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    system.SPAR3D = Mock()
    system.SPAR3D.from_pretrained.return_value = model
    remover = ModuleType("transparent_background")
    remover.Remover = Mock()
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "transparent_background", remover)
    monkeypatch.setitem(sys.modules, "spar3d.device", device)
    monkeypatch.setitem(sys.modules, "spar3d.system", system)
    for name in ["SPAR3D_DEVICE", "SPAR3D_MODEL", "SPAR3D_LOW_VRAM"]:
        monkeypatch.delenv(name, raising=False)
    return model, system, device, remover


def test_cpu_startup(startup, monkeypatch):
    model, system, device, remover = startup
    monkeypatch.setenv("SPAR3D_LOW_VRAM", "1")
    engine = Generator()
    assert engine.device == "cpu"
    device.select_device.assert_called_once_with()
    model.to.assert_called_once_with("cpu")
    model.eval.assert_called_once_with()
    assert system.SPAR3D.from_pretrained.call_args.kwargs == {
        "config_name": "config.yaml",
        "weight_name": "model.safetensors",
        "low_vram_mode": False,
    }
    remover.Remover.assert_called_once_with(device="cpu")


def test_cuda_low_vram_and_local_model(startup, monkeypatch):
    _, system, device, _ = startup
    device.select_device.return_value = "cuda"
    monkeypatch.setenv("SPAR3D_LOW_VRAM", "1")
    monkeypatch.setenv("SPAR3D_MODEL", "/models/spar3d")
    assert Generator().device == "cuda"
    call = system.SPAR3D.from_pretrained.call_args
    assert call.args == ("/models/spar3d",)
    assert call.kwargs["low_vram_mode"] is True


def test_startup_failure_is_not_retried(startup):
    _, system, _, remover = startup
    system.SPAR3D.from_pretrained.side_effect = RuntimeError("weights unavailable")
    with pytest.raises(RuntimeError, match="weights unavailable"):
        Generator()
    system.SPAR3D.from_pretrained.assert_called_once()
    remover.Remover.assert_not_called()
