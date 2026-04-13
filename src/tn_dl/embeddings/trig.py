from __future__ import annotations

import torch
from tn_dl.embeddings.base import BaseImageEmbedding


class TrigPixelEmbedding(BaseImageEmbedding):
    def __init__(self, local_dim: int = 2) -> None:
        super().__init__()
        if local_dim != 2:
            raise ValueError("TrigPixelEmbedding currently supports local_dim=2.")
        self.local_dim = local_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flattened = x.float().reshape(x.shape[0], -1).clamp(0.0, 1.0)
        angles = flattened * (torch.pi / 2.0)
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

