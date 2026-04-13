from __future__ import annotations

from math import prod

from torch import nn

from tn_dl.config import ModelConfig
from tn_dl.embeddings import TrigPixelEmbedding
from tn_dl.layers import MPSClassifier
from tn_dl.models.classifier import ImageClassifier


def build_model(config: ModelConfig) -> nn.Module:
    embedding_name = config.embedding_name.lower()
    layer_name = config.layer_name.lower()
    if embedding_name != "trig":
        raise ValueError(f"Unsupported embedding '{config.embedding_name}'.")
    if layer_name != "mps":
        raise ValueError(f"Unsupported tensor layer '{config.layer_name}'.")

    return ImageClassifier(
        embedding=TrigPixelEmbedding(local_dim=config.local_dim),
        tensor_layer=MPSClassifier(
            sequence_length=prod(config.input_shape),
            local_dim=config.local_dim,
            n_classes=config.n_classes,
            bond_dim=config.bond_dim,
            out_position=config.out_position,
            boundary=config.boundary,
        ),
    )
