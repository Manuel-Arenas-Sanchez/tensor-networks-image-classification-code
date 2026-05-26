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


def test_build_model_supports_fixed_patch_trig_custom_ttn() -> None:
    config = ModelConfig(
        embedding_name="fixed_patch_trig",
        layer_name="custom_ttn",
        input_shape=(3, 32, 32),
        local_dim=2,
        bond_dim=3,
        n_classes=5,
        patch_size=16,
    )

    model = build_model(config)
    logits = model(torch.rand(2, 3, 32, 32))

    assert isinstance(model, ImageClassifier)
    assert logits.shape == (2, 5)


def test_build_model_supports_aim_patch_mean_pool_linear() -> None:
    config = ModelConfig(
        embedding_name="aim_patch",
        layer_name="mean_pool_linear",
        input_shape=(3, 32, 32),
        local_dim=4,
        bond_dim=3,
        n_classes=5,
        patch_size=16,
        aim_channels=4,
        num_aim_blocks=1,
    )

    model = build_model(config)
    logits = model(torch.rand(2, 3, 32, 32))

    assert isinstance(model, ImageClassifier)
    assert logits.shape == (2, 5)
