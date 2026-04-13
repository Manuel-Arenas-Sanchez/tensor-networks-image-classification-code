from __future__ import annotations

import torch
from tensorkrowch.models import MPSLayer

from tn_dl.layers.base import BaseTensorNetworkLayer


class MPSClassifier(BaseTensorNetworkLayer):
    def __init__(
        self,
        sequence_length: int,
        local_dim: int,
        n_classes: int,
        bond_dim: int,
        out_position: int | None = None,
        boundary: str = "obc",
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.local_dim = local_dim
        self.layer = MPSLayer(
            n_features=sequence_length + 1,
            in_dim=local_dim,
            out_dim=n_classes,
            bond_dim=bond_dim,
            out_position=out_position,
            boundary=boundary,
            init_method="randn",
            std=0.02,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("MPSClassifier expects inputs shaped [batch, sequence, local_dim].")
        if x.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, received {x.shape[1]}."
            )
        if x.shape[2] != self.local_dim:
            raise ValueError(f"Expected local dimension {self.local_dim}, received {x.shape[2]}.")
        return self.layer(x)
