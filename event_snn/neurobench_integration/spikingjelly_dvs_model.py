"""NeuroBenchModel wrapper for AllConvAdLIFEGRU_DVS128 (SpikingJelly)."""

from __future__ import annotations

import torch
import torch.nn as nn

from neurobench.hooks.layer import LayerHook
from neurobench.models import NeuroBenchModel

from event_snn.models.adlif import AdLIF
from event_snn.neurobench_integration.adlif_neuron_hook import AdLIFNeuronHook


class SpikingJellyDVSModel(NeuroBenchModel):
    """
    Wrap experimentv2e DVS SNN for NeuroBench workload metrics.

    Input: (B, C, H, W, T) myCount tensor from V2E dataloader.
    Output: (B, classes, T) LI readout logits.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.net.eval()
        self.add_activation_module(AdLIF)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        with torch.no_grad():
            return self.net(batch, reset=True)

    def __net__(self) -> nn.Module:
        return self.net

    def register_hooks(self) -> None:
        for layer in self.connection_layers():
            self.connection_hooks.append(LayerHook(layer))

        for layer_info in self.activation_layers():
            self.activation_hooks.append(
                AdLIFNeuronHook(layer_info["layer"], layer_info["layer_name"])
            )
