from __future__ import annotations

import torch

from tn_dl.embeddings import FixedPatchTrigEmbedding, TrigPixelEmbedding


def test_trig_embedding_flattens_images_into_local_feature_vectors() -> None:
    embedding = TrigPixelEmbedding()
    images = torch.rand(2, 1, 3, 4)

    embedded = embedding(images)

    assert embedded.shape == (2, 12, 2)
    norms = torch.linalg.norm(embedded, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_fixed_patch_trig_embedding_builds_non_trainable_patch_features() -> None:
    embedding = FixedPatchTrigEmbedding(
        patch_size=16,
        local_dim=2,
        in_channels=3,
        order="zigzag",
        grid_h=14,
        grid_w=14,
    )
    images = torch.rand(2, 3, 224, 224)

    embedded = embedding(images)

    assert embedded.shape == (2, 196, 2)
    assert sum(parameter.numel() for parameter in embedding.parameters()) == 0
    norms = torch.linalg.norm(embedded, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
