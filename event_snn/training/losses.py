"""Loss functions for rate-coded SNN readout."""

from __future__ import annotations

import torch
import torch.nn as nn


def calc_ce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy for (B, C, T) mean-over-time or (B, C) pooled logits."""
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    if logits.dim() == 3:
        return loss_fn(logits.mean(dim=-1), targets)
    return loss_fn(logits, targets)


def logits_to_pred(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() == 3:
        return logits.mean(dim=-1).argmax(dim=1)
    return logits.argmax(dim=1)
