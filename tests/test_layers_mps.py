from __future__ import annotations

import torch

from tn_dl.layers import MPSClassifier


def test_mps_classifier_runs_forward_and_backward() -> None:
    layer = MPSClassifier(
        sequence_length=4,
        local_dim=2,
        n_classes=3,
        bond_dim=2,
    )
    inputs = torch.rand(5, 4, 2, requires_grad=True)

    logits = layer(inputs)
    loss = logits.sum()
    loss.backward()

    assert logits.shape == (5, 3)
    assert inputs.grad is not None
    assert any(parameter.grad is not None for parameter in layer.parameters())
