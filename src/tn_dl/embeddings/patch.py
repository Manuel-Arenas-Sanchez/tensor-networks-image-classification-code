from __future__ import annotations

import torch
from torch import nn

from tn_dl.embeddings.base import BaseImageEmbedding


class PatchEmbedding(BaseImageEmbedding):
    def __init__(self, patch_size: int, local_dim: int, in_channels: int=3) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.local_dim = local_dim
        self.in_channels = in_channels

        patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, local_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Embedding expects inputs shaped [batch, channels, height, width].")
        
        batch_size, channels, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, received {channels}")
        
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Image height and width must be divisible by patch_size")
        
        n_patches_h = height // self.patch_size
        n_patches_w = width // self.patch_size

        # [B, C, H, W] -> [B, C, n_patches_h, patch_size, n_patches_w, patch_size]
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size)
        
        # [B, C, n_patches_h, n_patches_w, patch_size, patch_size]
        x = x.contiguous()

        # [B, n_patches_h, n_patches_w, C, patch_size, patch_size]
        x = x.permute(0, 2, 3, 1, 4, 5)

        # [B, n_patches, C, patch_size, patch_size]
        x = x.reshape(batch_size, n_patches_h * n_patches_w, -1)

        # Finalmente obtenemos lo que queremos -> [B, n_patches, local_dim]
        x = self.proj(x)

        return x

