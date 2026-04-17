from __future__ import annotations

import torch

from tn_dl.embeddings.base import BaseImageEmbedding


class TrigPixelEmbedding(BaseImageEmbedding):
    def __init__(
        self,
        local_dim: int = 2,
        order: str = 'row',
        channels: int | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> None:
        super().__init__()
        if local_dim != 2:
            raise ValueError("TrigPixelEmbedding currently supports local_dim=2.")
        self.local_dim = local_dim
        self.order = order

        if order not in ("row", "zigzag"):
            raise ValueError(f"Unsupported order '{order}'. Use 'row' or 'zigzag'.")
        
        if order == "zigzag":
            if channels is None or height is None or width is None:
                raise ValueError(
                    "order = 'zigzag' requires channels, height and width."
                )
            perm = self._build_per_channel_zigzag_perm(channels, height, width)
            self.register_buffer("perm", perm, persistent=False)
        else:
            self.perm = None
    
    @staticmethod
    def _build_per_channel_zigzag_perm(c: int, h: int, w: int) -> torch.Tensor:
        """
        Zigzag espacial (h, w) aplicado por canal; canales concatenados en el orden 
        natural. La permutación opera sobre el tensor plano [C*H*W].
        """
        zigzag_hw: list[int] = []
        for r in range(h):
            row = list(range(r * w, (r + 1) * w))
            if r % 2 == 1:
                row.reverse()
            zigzag_hw.extend(row)
        
        perm: list[int] = []
        for ch in range(c):
            offset = ch * h* w
            perm.extend(offset + i for i in zigzag_hw)

        return torch.tensor(perm, dtype = torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flattened = x.float().reshape(x.shape[0], -1).clamp(0.0, 1.0)
        if self.order == "zigzag":
            flattened = flattened[:, self.perm]
        angles = flattened * (torch.pi / 2.0)
        return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

