from __future__ import annotations

from typing import Any

import torch


class IdentityTensorNetworkOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor) -> torch.Tensor:
        del ctx
        return inputs

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        del ctx
        return (grad_output,)
