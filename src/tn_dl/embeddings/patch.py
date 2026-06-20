from __future__ import annotations

import torch
from torch import nn

from tn_dl.embeddings.base import BaseImageEmbedding


class PatchEmbedding(BaseImageEmbedding):
    def __init__(
            self, 
            patch_size: int, 
            local_dim: int, 
            in_channels: int=3, 
            order: str = "row",
            grid_w: int | None = None,
            grid_h: int | None = None,) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.local_dim = local_dim
        self.in_channels = in_channels
        self.order = order

        patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(patch_dim, local_dim)

        if order not in ("row", "zigzag"):
            raise ValueError(f"Unsupported order '{order}'. Use 'row' or 'zigzag'.")
        
        if order == "zigzag":
            if grid_h is None or grid_w is None:
                raise ValueError("order= 'zigzag' requires grid_h and grid_w.")
            perm = self._build_zigzag_perm(grid_h, grid_w)
            # persistent=False -> no se guarda en checkpoints; es determinista
            self.register_buffer("perm", perm, persistent=False)
        else:
            self.perm = None
    
    @staticmethod
    def _build_zigzag_perm(h: int, w: int) -> torch.Tensor:
        """
        Índices de permutación row-major -> zigzag para una rejilla h x w.
        Fila par: izquierda -> derecha. Fila impar: derecha -> izquierda.
        """
        indices = []
        for r in range(h):
            row = list(range(r * w, (r + 1) * w))
            if r % 2 == 1:
                row.reverse()
            indices.extend(row)
        return torch.tensor(indices, dtype = torch.long)

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

        if self.order == "zigzag":
            x = x[:, self.perm]

        return x

