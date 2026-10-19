"""Patch AdLIF + NeuroBench NeuronOperations table for SE-AdLIF dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from event_snn.models.adlif import AdLIF

# Matches profile_dvs_compute_spikes.py (between Leaky and Synaptic in NeuroBench).
ADLIF_OPS_PER_UPDATE = 8
ADLIF_RESET_MECHANISM = "zero"


def extend_neuron_ops_reset_operations() -> None:
    from neurobench.metrics.workload.neuron_operations import neuron_ops_reset_operations

    neuron_ops_reset_operations.setdefault(
        "AdLIF",
        {
            "subtract": ADLIF_OPS_PER_UPDATE,
            "zero": ADLIF_OPS_PER_UPDATE,
        },
    )


def patch_adlif_forward() -> None:
    from event_snn.models.adlif import AdLIF

    if getattr(AdLIF, "_neurobench_patched", False):
        return

    import torch

    def forward(self: "AdLIF", x: torch.Tensor) -> torch.Tensor:
        self.neuron.apply_parameter_constraints()
        u, z, w = self.neuron.initial_state(x[0], x.device)
        states = (u, z, w)
        outs = []
        pre_mem = []
        post_mem = []
        for t in range(x.shape[0]):
            pre_mem.append(states[0].detach().clone())
            out, states = self.neuron(x[t], states)
            post_mem.append(states[0].detach().clone())
            outs.append(out)
        self._nb_pre_mem = pre_mem
        self._nb_post_mem = post_mem
        self.mem = states[0]
        self._reset_mechanism = ADLIF_RESET_MECHANISM
        return torch.stack(outs, dim=0)

    AdLIF.forward = forward  # type: ignore[method-assign]
    AdLIF._neurobench_patched = True


def apply_adlif_neurobench_patches() -> None:
    """Idempotent setup required before NeuroBench Benchmark.run()."""
    extend_neuron_ops_reset_operations()
    patch_adlif_forward()
