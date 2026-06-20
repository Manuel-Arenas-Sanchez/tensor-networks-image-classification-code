from __future__ import annotations

import torch

from tn_dl.embeddings import TrigPixelEmbedding


def test_trig_embedding_flattens_images_into_local_feature_vectors() -> None:
    embedding = TrigPixelEmbedding()
    images = torch.rand(2, 1, 3, 4)

    embedded = embedding(images)

    assert embedded.shape == (2, 12, 2)
    norms = torch.linalg.norm(embedded, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
