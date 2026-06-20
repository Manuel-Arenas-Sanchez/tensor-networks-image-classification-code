from __future__ import annotations

import logging

import torchvision
from torch import nn
from torchvision.models import ResNet50_Weights

from tn_dl.config import ModelConfig

logger = logging.getLogger(__name__)


def build_resnet50(config: ModelConfig) -> nn.Module:
    """
    Construye un baseline ResNet50 de torchvision para clasificar los mismos crops
    que la red de tensores.

    Dos variantes segun config.pretrained:
        - True:  fine-tuning desde pesos ImageNet (baseline fuerte estandar de vision).
        - False: pesos aleatorios, entrenado desde cero solo con los crops.

    El modelo consume directamente imagenes [B, 3, H, W] (los crops normalizados con
    media/std de ImageNet que ya produce el data pipeline), de modo que es
    directamente comparable con la TN: mismos crops, mismas particiones, mismas clases.
    """
    in_channels, _, _ = config.input_shape
    if in_channels != 3:
        raise ValueError(
            f"ResNet50 espera 3 canales de entrada, recibido {in_channels}."
        )

    weights = ResNet50_Weights.IMAGENET1K_V2 if config.pretrained else None
    model = torchvision.models.resnet50(weights=weights)

    logger.info(
        "ResNet50 construido: pretrained = %s, freeze_backbone = %s, n_classes = %s",
        config.pretrained,
        config.freeze_backbone,
        config.n_classes,
    )

    if config.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    if config.head_dropout > 0.0:
        model.fc = nn.Sequential(
            nn.Dropout(p=config.head_dropout),
            nn.Linear(in_features, config.n_classes),
        )
    else:
        model.fc = nn.Linear(in_features, config.n_classes)

    return model
