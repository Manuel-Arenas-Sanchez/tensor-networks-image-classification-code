from __future__ import annotations

import torch

from tn_dl.config import ModelConfig
from tn_dl.models import ImageClassifier, build_model


def test_build_model_returns_image_classifier() -> None:
    config = ModelConfig(
        embedding_name="trig",
        layer_name="mps",
        input_shape=(1, 2, 2),
        local_dim=2,
        bond_dim=3,
        n_classes=4,
    )

    model = build_model(config)
    logits = model(torch.rand(3, 1, 2, 2))

    assert isinstance(model, ImageClassifier)
    assert logits.shape == (3, 4)
