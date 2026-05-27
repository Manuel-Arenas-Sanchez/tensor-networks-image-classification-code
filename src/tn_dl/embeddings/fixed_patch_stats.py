from __future__ import annotations

import torch

from tn_dl.embeddings.base import BaseImageEmbedding


class FixedPatchStatsEmbedding(BaseImageEmbedding):
    def __init__(
        self,
        patch_size: int,
        local_dim: int,
        in_channels: int = 3,
        order: str = "row",
        grid_h: int | None = None,
        grid_w: int | None = None,
    ) -> None:
        super().__init__()

        expected_local_dim = 4 * in_channels

        if patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")
        if in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if local_dim != expected_local_dim:
            raise ValueError(
                f"FixedPatchStatsEmbedding requires local_dim = {expected_local_dim}"
                f"for {in_channels} input channels."
            )
        if order not in ("row", "zigzag"):
            raise ValueError(f"Unsupported order '{order}'. Use 'row' or 'zigzag'.")
    
        self.patch_size = patch_size
        self.local_dim = local_dim
        self.in_channels = in_channels
        self.order = order

        if order == "zigzag":
            if grid_h is None or grid_w is None:
                raise ValueError("order = 'zigzag' requires grid_h and grid_w")
            perm = self._build_zigzag_perm(grid_h, grid_w)
            self.register_buffer("perm", perm, persistent = False)
        else:
            self.perm = None

    @staticmethod
    def _build_zigzag_perm(h: int, w:int) -> torch.Tensor:
        indices: list[int] = []
        for r in range(h):
            row = list(range(r*w, (r+1) * w))
            if r % 2 == 1:
                row.reverse()
            indices.extend(row)
        return torch.tensor(indices, dtype = torch.long)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "FixedPatchStatsEmbedding expects input shaped [batch, channels, height, width]."
            )
        
        batch_size, channels, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, received {channels}.")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Image height and width must be divisible by patch_size.")
        
        n_patches_h = height // self.patch_size
        n_patches_w = width // self.patch_size

        x = x.float().clamp(0.0, 1.0)

        # [B, C, H, W] -> [B, C, grid_h, grid_w, patch_size, patch_size]
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3,
            self.patch_size,
            self.patch_size,
        )

        means = patches.mean(dim=(-1, -2))
        stds = patches.std(dim=(-1, -2), unbiased = False)
        mins = patches.amin(dim=(-1, -2))
        maxs = patches.amax(dim=(-1, -2))

        # [B, 12, grid_h, grid_w]
        features = torch.cat([means, stds, mins, maxs], dim = 1)

        #[B, grid_h, grid_w, 12] -> [B, grid_h * grid_w, 12]
        features = features.permute(0, 2, 3, 1)
        features = features.reshape(batch_size, n_patches_h * n_patches_w, self.local_dim)

        if self.order == "zigzag":
            if self.perm is None:
                raise RuntimeError("order = 'zigzag' but perm was not initialized.")
            features = features[:, self.perm]
        
        return features