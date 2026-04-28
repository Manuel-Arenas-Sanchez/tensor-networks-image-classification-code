from __future__ import annotations

import math

import torch
import torch.nn as nn

from tn_dl.embeddings.base import BaseImageEmbedding
from tn_dl.layers.aim import AIM2D


class AIMPatchEmbedding(BaseImageEmbedding):
    """
    Embedding de imágen basado en un front-end AIM 2D antes de la TTN.

    Entrada esperada:
        x: [B, C, H, W]
    
    Salida:
        y: [B, num_sites, local_dim]
    
    Para Flowers102 con imágenes 224x224 y patch_size=16:

        [B, 3, 224, 224]
        1. stem CNN con 4 reducciones stride=2
        2. [B, aim_channels, 14, 14]
        3. AIM2D x num_aim_blocks
        4. Conv2D 1x1
        5. [B, local_dim, 14, 14]
        6. flatten
        7. [B, 196, local_dim]
    
    Diferencia con PatchEmbedding:
        PatchEmbedding corta la imágen en patches y proyecta cada patch con un Linear independiente.
        Eso destruye la estructura 2D en cuento se forma la secuencia.

        AIMPatchEmbedding matiene un mapa espacial [H', W'] antes del flatten. Por eso puede aplicar
        convoluciones depthwise dentro de AIM2D y conservar localidad espacial antes de entregar la
        secuencia a la HybridCustomTTN
    
    Interpretación:
        - patch_size ya no significa "tamaño del patch que se aplana" sino factor total de reducción
          espacial del stem.
        - Con patch_size = 16, el stem reduce 224 -> 112 -> 56 -> 28 -> 14.
        - Cada posición final del mapa 14x14 funciona como un sitio de la TTN.
    """

    def __init__(
            self,
            patch_size: int,
            local_dim: int,
            in_channels: int = 3,
            order: str = "row",
            grid_w: int | None = None,
            grid_h: int | None = None,
            aim_channels: int = 32,
            num_aim_blocks: int = 2,
            aim_expansion_ratio: int = 3,
            aim_kernel_size: int = 3,
            aim_use_ln: bool = True,
    ) -> None:
        super().__init__()

        if patch_size <= 0:
            raise ValueError("patch_size debe ser un entero positivo.")
        if local_dim <= 0:
            raise ValueError("local_dim debe ser un entero positivo.")
        if in_channels <= 0:
            raise ValueError("in_channels debe ser un entero positivo.")
        if aim_channels <= 0:
            raise ValueError("aim_channels debe ser un entero positivo.")
        if num_aim_blocks < 0:
            raise ValueError("num_aim_blocksno puede ser negativo.")
        if order not in ("row", "zigzag"):
            raise ValueError(f"Unsupported order '{order}'. Use 'row' or 'zigzag'.")

        # El stem esta definido como una pila de Conv2d kernel = 2, stride = 2. Por tanto patch_size
        # debe ser una potencia de 2 para que log2(patch_size) sea el número entero de reducciones
        # espaciales.
        if patch_size & (patch_size - 1) != 0:
            raise ValueError(
                "AIMPatchEmbedding requiere que patch_size sea potencia de 2, porque el stem reduce"
                "con stride = 2 repetidamente."
            )

        self.patch_size = patch_size
        self.local_dim = local_dim
        self.in_channels = in_channels
        self.order = order
        self.aim_channels = aim_channels
        self.num_aim_blocks = num_aim_blocks
        self.aim_expansion_ratio = aim_expansion_ratio
        self.aim_kernel_size = aim_kernel_size
        self.aim_use_ln = aim_use_ln

        self.num_downsampling_steps = int(math.log2(patch_size))

        self.stem = self._build_stem(
            in_channels = in_channels,
            out_channels = aim_channels,
            num_steps = self.num_downsampling_steps,
        )

        self.aim_blocks = nn.Sequential(
            *[
                AIM2D(
                    channels = aim_channels,
                    expansion_ratio = aim_expansion_ratio,
                    kernel_size = aim_kernel_size,
                    use_ln = aim_use_ln,
                )
                for _ in range(num_aim_blocks)
            ]
        )

        # Conv 1x1 final: convierte el número de canales del front-end AIM en la dimensión local que
        # espera la capa tensorial.
        self.to_local_dim = nn.Conv2d(
            in_channels = aim_channels,
            out_channels = local_dim,
            kernel_size = 1,
        )

        if order == "zigzag":
            if grid_h is None or grid_w is None:
                raise ValueError("order = 'zigzag' requires grid_h and grid_w.")
            perm = self._build_zigzag_perm(grid_h, grid_w)
            self.register_buffer("perm", perm, persistent = False)
        else:
            self.perm = None
    
    @staticmethod
    def _build_stem(
        in_channels: int,
        out_channels: int,
        num_steps: int,
    ) -> nn.Sequential:
        """
        Contruye el stem convolucional que reemplaza al corte duro en patches.

        Cada bloque hace:

            Conv2d(kernel = 2, stride = 2)
            BatchNorm2d
            ReLU
        
        Con patch_size = 16 se crean 4 bloques, asi que la imágen baja de 224x224 a 14x14.

        Si patch_size = 1, no hay downsampling. En ese caso usamos una Conv2d 1x1 para llevar la
        imágen directamente a aim_channels.
        """
        layers: list[nn.Module] = []

        if num_steps == 0:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels = in_channels,
                        out_channels = out_channels,
                        kernel_size = 1,
                        bias = False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace = True),
                ]
            )
            return nn.Sequential(*layers)
        
        current_channels = in_channels

        for _ in range(num_steps):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels = current_channels,
                        out_channels = out_channels,
                        kernel_size = 2,
                        stride = 2,
                        padding = 0,
                        bias = False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace = True),
                ]
            )
            current_channels = out_channels
        
        return nn.Sequential(*layers)
    
    @staticmethod
    def _build_zigzag_perm(h: int, w: int) -> torch.Tensor:
        """
        Indices de permutación row-major -> zigzag para una rejilla h x w.

        Fila par:
            izquierda -> derecha

        Fila impar:
            derecha -> izquierda
        
        Es la misma convención que PatchEmbedding, pero esta vez aplicada al mapa espacial ya
        procesado por el stem y por AIM2D.
        """
        indices: list[int] = []

        for r in range(h):
            row = list(range(r * w, (r+1) * w))
            if r % 2 == 1:
                row.reverse()
            indices.extend(row)
        
        return torch.tensor(indices, dtype = torch.long)
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Convierte una imagen [B, C, H, W] en una secuencia [B, num_sites, local_dim].

        Pasos:
            1. Comprobar que la forma y los canales son correctos.
            2. Aplica el stem CNN para reducir resolución.
            3. Aplica los bloques AIM2D manteniendo estructura [H', W'].
            4. Proyecta canales a local_dim.
            5. Aplana el mapa 2d como secuencia.
            6. Aplica zigzag si se pidio en la configuracion.
        """
        if x.ndim != 4:
            raise ValueError(
                "AIMPatchEmbedding expects inputs shaped [batch, channels, height, width]."
            )
        
        batch_size, channels, height, width = x.shape

        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, received {channels}.")
        
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError("Image height and width must be divisible by patch_size.")
        
        x = self.stem(x)
        x = self.aim_blocks(x)
        x = self.to_local_dim(x)

        _, _, grid_h, grid_w = x.shape

        # [B, local_dim, H', W'] -> [B, H', W', local_dim]
        x = x.permute(0, 2, 3, 1)

        # [B, H'. W', local_dim] -> [B, H' * W', local_dim]
        x = x.reshape(batch_size, grid_h * grid_w, self.local_dim)

        if self.order == "zigzag":
            if self.perm is None:
                raise RuntimeError("order = 'zigzag' but perm was not initialized.")
            if self.perm.numel() != grid_h * grid_w:
                raise ValueError(
                    "The 'zigzag' permutation size does not match the produced grid."
                    f"Expected {self.perm.numel()} sites, got {grid_h * grid_w}."
                )
            x = x[:, self.perm]
        
        return x