"""Leaky-integrator readout for EGRU classifier head (wearsnn AllConvPLIFEGRULISNN style)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import Module, Parameter


@torch.jit.script
def LIScript(u_tm1: Tensor, current: Tensor, decay_u: Tensor) -> Tensor:
    return decay_u * u_tm1 + (1.0 - decay_u) * current


class LIReadout(Module):
    """Multi-step LI over (T, B, F) — matches wearsnn LI wrapper."""

    def __init__(self, out_features: int, device: torch.device | None = None) -> None:
        super().__init__()
        self.out_features = out_features
        self.decay = Parameter(
            torch.exp(-torch.tensor(1.0 / 15.0)), requires_grad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        u = torch.zeros_like(x[0])
        outs = []
        for t in range(x.shape[0]):
            u = LIScript(u, x[t], self.decay)
            outs.append(u)
        return torch.stack(outs, dim=0)
