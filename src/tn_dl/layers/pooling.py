from __future__ import annotations

import torch
from torch import nn

from tn_dl.layers.base import BaseTensorNetworkLayer


class MeanPoolLinearClassifier(BaseTensorNetworkLayer):
    def __init__(
        self,
        sequence_length: int,
        local_dim: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.local_dim = local_dim
        self.n_classes = n_classes
        self.classifier = nn.Linear(local_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "MeanPoolLinearClassifier expects inputs shaped [batch, sequence, local_dim]."
            )
        if x.shape[1] != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, received {x.shape[1]}."
            )
        if x.shape[2] != self.local_dim:
            raise ValueError(f"Expected local dimension {self.local_dim}, received {x.shape[2]}.")

        pooled = x.mean(dim=1)
        return self.classifier(pooled)
