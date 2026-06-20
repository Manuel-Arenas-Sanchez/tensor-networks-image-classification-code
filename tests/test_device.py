from __future__ import annotations

import pytest
import torch

from tn_dl.utils.device import resolve_device


def test_resolve_device_respects_cpu_override() -> None:
    device = resolve_device("cpu")
    assert device.type == "cpu"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = resolve_device("auto")
    assert device.type == "cpu"


def test_resolve_device_accepts_cuda_string_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device = resolve_device("cuda")
    assert device.type == "cuda"
