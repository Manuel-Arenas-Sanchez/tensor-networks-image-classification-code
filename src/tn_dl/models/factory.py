from __future__ import annotations

from math import prod

from torch import nn

from tn_dl.config import ModelConfig
from tn_dl.embeddings import PatchEmbedding, TrigPixelEmbedding
from tn_dl.layers import MPSClassifier, MPSClassifier_1
from tn_dl.models.classifier import ImageClassifier


def build_model(config: ModelConfig) -> nn.Module:
    embedding_name = config.embedding_name.lower()
    layer_name = config.layer_name.lower()
    order = config.linearization_order.lower()

    in_channels, height, width = config.input_shape

    if embedding_name == "trig":
        embedding = TrigPixelEmbedding(
            local_dim = config.local_dim,
            order = order,
            channels = in_channels,
            height = height,
            width = width,
        )
        sequence_length = prod(config.input_shape)
    elif embedding_name == "patch":
        if config.patch_size is None:
            raise ValueError("PatchEmbedding requires 'patch_size' in ModelConfig")
        if height % config.patch_size != 0 or width % config.patch_size != 0:
            raise ValueError("input_shape H and W must be divisible by patch_size.")
        grid_h = height // config.patch_size
        grid_w = width // config.patch_size  
        embedding = PatchEmbedding(
            patch_size = config.patch_size,
            local_dim = config.local_dim,
            in_channels = in_channels,
            order = order,
            grid_h = grid_h,
            grid_w = grid_w
        )
        sequence_length = grid_h * grid_w
    else:
        raise ValueError(f"Unsupported embedding '{config.embedding_name}'.")
    
    if layer_name == "mps":
        tensor_layer = MPSClassifier(
            sequence_length = sequence_length,
            local_dim = config.local_dim,
            n_classes = config.n_classes,
            bond_dim=  config.bond_dim,
            out_position = config.out_position,
            boundary = config.boundary,
        )
    elif layer_name == "custom_ttn":
        tensor_layer = MPSClassifier_1(
            sequence_length = sequence_length,
            local_dim = config.local_dim,
            n_classes = config.n_classes,
            bond_dim = config.bond_dim,
            out_position = config.out_position,
            boundary = config.boundary,
        )
    else:
         raise ValueError(f"Unsupported tensor layer '{config.layer_name}'.")

    return ImageClassifier(embedding=embedding, tensor_layer=tensor_layer)