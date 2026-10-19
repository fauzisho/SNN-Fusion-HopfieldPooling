"""NeuroBenchModel wrapper for IMU EGRUClassifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from neurobench.hooks.layer import LayerHook
from neurobench.models import NeuroBenchModel

from neurobench_integration.egru_activation_hook import EGRUActivationHook


class EGRUIMUNeuroBenchModel(NeuroBenchModel):
    """
    Wrap EGRUClassifier for NeuroBench workload metrics.

    Input: (B, T, F) normalized IMU features.
    Output: (B, C, T) logits.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.net.eval()

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            return self.net(batch)

    def __net__(self) -> nn.Module:
        return self.net

    def register_hooks(self) -> None:
        for layer in self.connection_layers():
            self.connection_hooks.append(LayerHook(layer))

        self.activation_hooks.append(EGRUActivationHook(self.net.egru, "egru"))
