from __future__ import annotations

import torch
from torch import nn

from tn_dl.embeddings import BaseImageEmbedding
from tn_dl.layers import BaseTensorNetworkLayer


class ImageClassifier(nn.Module):
    def __init__(
        self,
        embedding: BaseImageEmbedding,
        tensor_layer: BaseTensorNetworkLayer,
    ) -> None:
        super().__init__()
        self.embedding = embedding
        self.tensor_layer = tensor_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tensor_layer(self.embedding(x))
