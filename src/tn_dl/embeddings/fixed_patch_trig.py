from __future__ import annotations

import torch

from tn_dl.embeddings.base import BaseImageEmbedding


class FixedPatchTrigEmbedding(BaseImageEmbedding):
    def __init__(
        self,
        patch_size: int,
        local_dim: int = 2,
        in_channels: int = 3,
        order: str = "row",
        grid_h: int | None = None,
        grid_w: int | None = None,
    ) -> None:
        super().__init__()

        if patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")
        if local_dim != 2:
            raise ValueError("FixedPatchTrigEmbedding currently supports local_dim=2.")
        if in_channels <= 0:
            raise ValueError("in_channels must be a positive integer.")
        if order not in ("row", "zigzag"):
            raise ValueError(f"Unsupported order '{order}'. Use 'row' or 'zigzag'.")

        self.patch_size = patch_size
        self.local_dim = local_dim
        self.in_channels = in_channels
        self.order = order

        if order == "zigzag":
            if grid_h is None or grid_w is None:
                raise ValueError("order='zigzag' requires grid_h and grid_w.")
            perm = self._build_zigzag_perm(grid_h, grid_w)
            self.register_buffer("perm", perm, persistent=False)
        else:
            self.perm = None

    @staticmethod
    def _build_zigzag_perm(h: int, w: int) -> torch.Tensor:
        indices: list[int] = []
        for r in range(h):
            row = list(range(r * w, (r + 1) * w))
            if r % 2 == 1:
                row.reverse()
            indices.extend(row)
        return torch.tensor(indices, dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "FixedPatchTrigEmbedding expects inputs shaped [batch, channels, height, width]."
            )

        batch_size, channels, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, received {channels}.")

        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Image height and width must be divisible by patch_size.")

        n_patches_h = height // self.patch_size
        n_patches_w = width // self.patch_size

        x = x.float().clamp(0.0, 1.0)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3,
            self.patch_size,
            self.patch_size,
        )
        patch_means = x.mean(dim=(1, 4, 5)).reshape(batch_size, n_patches_h * n_patches_w)

        if self.order == "zigzag":
            if self.perm is None:
                raise RuntimeError("order='zigzag' but perm was not initialized.")
            if self.perm.numel() != patch_means.shape[1]:
                raise ValueError(
                    "The 'zigzag' permutation size does not match the produced grid. "
                    f"Expected {self.perm.numel()} sites, got {patch_means.shape[1]}."
                )
            patch_means = patch_means[:, self.perm]

        angles = patch_means * (torch.pi / 2.0)
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
