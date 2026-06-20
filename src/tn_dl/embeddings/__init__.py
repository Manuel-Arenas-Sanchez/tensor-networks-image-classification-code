from __future__ import annotations

from tn_dl.embeddings.aim_patch import AIMPatchEmbedding
from tn_dl.embeddings.base import BaseImageEmbedding
from tn_dl.embeddings.patch import PatchEmbedding
from tn_dl.embeddings.trig import TrigPixelEmbedding

__all__ = [
    "BaseImageEmbedding",
    "TrigPixelEmbedding",
    "PatchEmbedding",
    "AIMPatchEmbedding",
]

