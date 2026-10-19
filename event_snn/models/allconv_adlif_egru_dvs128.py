"""AllConv + AdLIF + EGRU + LI for native 128×128 DVS128 v2e.

Single-path variant inspired by emmanuela ``DualPathAdLIFEGRULISNN`` (AdLIF conv
blocks + EGRU temporal head), without the ventral/dorsal dual stream.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, layer

from event_snn.models.adlif import AdLIF
from event_snn.models.egru import EGRU
from event_snn.models.li_readout import LIReadout


class AllConvAdLIFEGRU_DVS128(torch.nn.Module):
    """
    AllConv backbone with SE-AdLIF neurons, EGRU temporal pooling, LI readout.

    Expected input: (B, 2, 128, 128, T) with T=100 for 20 ms bins over 2 s.
    Output: (B, classes, T).
    """

    def __init__(
        self,
        inp_features: int = 2,
        channels: int = 8,
        feat_neurons: int = 512,
        classes: int = 18,
        dropout: float = 0.05,
        input_height: int = 128,
        input_width: int = 128,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.bias = False
        self.device = torch.device(device) if device is not None else None

        def adlif_block(ch: int) -> AdLIF:
            return AdLIF(ch, ch, threshold=1.0, rnn=False)

        self.feature_extractor = nn.ModuleList(
            [
                layer.Conv2d(
                    inp_features,
                    channels,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=self.bias,
                ),
                layer.BatchNorm2d(channels, momentum=0.01, eps=1e-3),
                adlif_block(channels),
                layer.Dropout(p=dropout),
                layer.Conv2d(
                    channels,
                    2 * channels,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=self.bias,
                ),
                layer.BatchNorm2d(2 * channels, momentum=0.01, eps=1e-3),
                adlif_block(2 * channels),
                layer.Dropout(p=dropout),
                layer.Conv2d(
                    2 * channels,
                    4 * channels,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    bias=self.bias,
                ),
                layer.BatchNorm2d(4 * channels, momentum=0.01, eps=1e-3),
                adlif_block(4 * channels),
                layer.Dropout(p=dropout),
                layer.Conv2d(
                    4 * channels,
                    8 * channels,
                    kernel_size=3,
                    padding=0,
                    stride=2,
                    bias=self.bias,
                ),
                layer.BatchNorm2d(8 * channels, momentum=0.01, eps=1e-3),
                adlif_block(8 * channels),
                layer.Dropout(p=dropout),
                layer.Conv2d(
                    8 * channels,
                    8 * channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=self.bias,
                ),
                layer.BatchNorm2d(8 * channels, momentum=0.01, eps=1e-3),
                adlif_block(8 * channels),
                layer.Dropout(p=dropout),
            ]
        )

        self.step_mode = "m"
        functional.set_step_mode(self.feature_extractor, step_mode=self.step_mode)
        functional.set_backend(self.feature_extractor, backend="torch")

        flat_dim = self._infer_flat_dim(input_height, input_width)

        self.egru = EGRU(
            flat_dim,
            feat_neurons,
            dropout=dropout,
            zoneout=0.0,
            return_state_sequence=True,
        )
        self.classifier = nn.ModuleList(
            [
                layer.Linear(feat_neurons, classes, bias=self.bias),
                LIReadout(classes, device=self.device),
            ]
        )
        functional.set_step_mode(self.classifier, step_mode=self.step_mode)
        functional.set_backend(self.classifier, backend="torch")

        for module in self.modules():
            if isinstance(module, layer.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, layer.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _infer_flat_dim(self, input_height: int, input_width: int) -> int:
        with torch.no_grad():
            x = torch.zeros(2, 2, input_height, input_width, 4)
            x = x.permute(4, 0, 1, 2, 3).contiguous()
            for block in self.feature_extractor:
                x = block(x)
            return int(x.flatten(2).shape[-1])

    def forward(self, spike: torch.Tensor, reset: bool = True) -> torch.Tensor:
        if reset:
            functional.reset_net(self)
        x = spike.permute(4, 0, 1, 2, 3).contiguous()
        if self.device is not None:
            x = x.to(self.device)

        for block in self.feature_extractor:
            x = block(x)

        x = x.flatten(2)
        x = self.egru(x)

        for block in self.classifier:
            x = block(x)

        return x.permute(1, 2, 0).contiguous()
