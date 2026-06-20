from __future__ import annotations

import torch
import torch.nn as nn


class AIM2D(nn.Module):
    """
    Bloque AIM 2D: Antisymmetric Interaction Module adaptado a imágenes.
    https://arxiv.org/html/2502.09928v1#S3

    Entrada esperada:
        x: [B, C, H, W]

    Salida:
        out: [B, C, H, W]
    
    Idea Matemática:
        El bloque construye dos ramas que aplican las mismas dos familias de operaciones, pero en
        orden inverso:

            rama 1: Linear por canales -> DWConv espacial
            rama 2: GroupedConv espacial -> Linear por canales
        
        Después se multiplican elemento a elemento:

            interaction = f1(x) * f2(x)
        
        Este es un producto de Hadamard, que introduce una interacción multiplicativa. Si se usan
        operaciones sin bias y sin normalización, f1 y f2 son mapas lineales de x, y por tanto
        interaction es bilineal/cuadrática en la entrada.
    
    Porque tiene sentido ponerlo antes de la TTN:
        Porque conserva la estructura espacial 2D antes de aplanar la secuencia. Usa convoluciones
        con pesos compartidos, lo que actua como sesgo inductivo y regularización.
        Introduce no linealidad multiplicativa sin recurrir a un ReLU/Sigmoid, y el residual permite
        que el bloque empiece cerca de la identidad.
    
    Nota:
        nn.Linear opera sobre la última dimension. Por eso, para aplicarlo sobre canales, se permuta
        de forma temporal:

            [B, C, H, W] -> [B, H, W, C] -> Linear -> [B, C_out, H, W]
        
        Esto corrige el pseudo-codigo ambiguo del paper, donde una doble permutación consecutiva
        puede parecer una identidad.
    """

    def __init__(
            self,
            channels: int,
            expansion_ratio: int = 3,
            kernel_size: int = 3,
            use_ln: bool = True,
            init_scale: float = 0.1,
            use_bias: bool = False,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError("channels debe ser un entero positivo.")
        if expansion_ratio <= 0:
            raise ValueError("expansion_ratio debe ser un entero positivo.")
        if kernel_size <= 0:
            raise ValueError("kernel_size debe ser un entero positivo.")
        if kernel_size % 2 == 0:
            raise ValueError(
                "kernel_size debe ser impar para preservar HxW con padding = kernel_size/2."
                )
        
        self.channels = channels
        self.expansion_ratio = expansion_ratio
        self.hidden_channels = channels * expansion_ratio
        self.kernel_size = kernel_size
        self.use_ln = use_ln

        padding = kernel_size // 2

        # Rama 1:
        #     mezcla primero canales en cada pixel y después aplica una convolución depthwise
        #     sobre el mapa expandido.
        self.f1_linear = nn.Linear(
            in_features = channels,
            out_features = self.hidden_channels,
            bias = use_bias,
        )
        self.f1_depthwise = nn.Conv2d(
            in_channels = self.hidden_channels,
            out_channels = self.hidden_channels,
            kernel_size = kernel_size,
            padding = padding,
            groups = self.hidden_channels,
            bias = use_bias,
        )

        # Rama 2:
        #     aplica primero una convolucion agrupada. groups = channels hace que cada canal
        #     original genere expansion_ratio canales, sin mezclar aun información entre 
        #     canales. La mezcla de canales ocurre después con el Linear.
        self.f2_grouped = nn.Conv2d(
            in_channels = channels,
            out_channels = self.hidden_channels,
            kernel_size = kernel_size,
            padding = padding,
            groups = channels,
            bias = use_bias,
        )
        self.f2_linear = nn.Linear(
            in_features = self.hidden_channels,
            out_features = self.hidden_channels,
            bias = use_bias,
        )

        # La normalización se aplica después del Hadamard. Así la interacción multiplicativa
        # f1(x) * f2(x) ya se ha formado antes de estabilizar escala y distribución.
        self.norm = (
            nn.LayerNorm(self.hidden_channels)
            if use_ln
            else nn.Identity()
        )

        # Proyecta de vuelta desde C * expansion_ratio hasta C para poder sumar el residual
        # con la entrada original.
        self.projection = nn.Conv2d(
            in_channels = self.hidden_channels,
            out_channels = channels,
            kernel_size = 1,
            bias = use_bias,
        )

        # Escala residual aprendible. Inicializarla pequeña hace que el bloque empiece cerca
        # de la identidad y evita cambios bruscos de escala al colocar AIM delante de la TTN.
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))
    
    @staticmethod
    def _apply_linear_on_channels(
        x: torch.Tensor,
        linear: nn.Linear,
    ) -> torch.Tensor:
        """
        Aplica un nn.Linear sobre la dimension de canales de un tensor [B, C, H, W].

        nn.Linear espera que la dimensión a tranformar sea la última, así que se pasa temporalmente
        a formato channels-last:

            [B, C, H, W] -> [B, H, W, C]
        
        y luego se vuelve al formato habitual de PyTorch para convoluciones:

            [B, H, W, C_out] -> [B, C_out, H, W]
        """
        x = x.permute(0, 2, 3, 1)
        x = linear(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()

    def _apply_norm_on_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica LayerNorm sobre canales en un tensor [B, C, H, W].

        LayerNorm normaliza la última dimensión, así que se usa la misma idea que con nn.Linear:
        permutar a [B, H, W, C], normalizar canales y volver.
        """
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta el bloque AIM completo.

        Flujo:
            1. Comprueba que la entrada sea [B, C, H, W].
            2. Calcula las dos ramas en orden inverso.
            3. Multiplica ambas ramas elemento a elemento.
            4. Normaliza opcionalmente.
            5. Proyecta de vuelta a C canales.
            6. Suma el residual escalado.
        """
        if x.ndim != 4:
            raise ValueError(
                f"AIM2D espera un tensor [B, C, H, W], pero recibió ndim = {x.ndim}."
            )
        if x.shape[1] != self.channels:
            raise ValueError(
                f"AIM2D fue construido con channels = {self.channels},"
                f"pero recibio una entrada con C={x.shape[1]}."
            )
        
        f1 = self._apply_linear_on_channels(x, self.f1_linear)
        f1 = self.f1_depthwise(f1)

        f2 = self.f2_grouped(x)
        f2 = self._apply_linear_on_channels(f2, self.f2_linear)

        interaction = f1 * f2
        interaction = self._apply_norm_on_channels(interaction)

        correction = self.projection(interaction)

        return x + self.residual_scale * correction
