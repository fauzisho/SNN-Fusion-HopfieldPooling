"""NeuronHook for custom AdLIF (not snnTorch / SpikingJelly BaseNode)."""

from __future__ import annotations


class AdLIFNeuronHook:
    """Collect AdLIF spike outputs and membrane traces for NeuroBench metrics."""

    def __init__(self, layer, name: str | None = None) -> None:
        self.activation_outputs: list = []
        self.activation_inputs: list = []
        self.pre_fire_mem_potential: list = []
        self.post_fire_mem_potential: list = []
        self.name = name
        self.layer = layer
        self.spiking = True
        self.hook = layer.register_forward_hook(self.hook_fn)
        self.hook_pre = layer.register_forward_pre_hook(self.pre_hook_fn)

    def pre_hook_fn(self, _layer, input) -> None:
        self.activation_inputs.append(input)

    def hook_fn(self, _layer, _input, output) -> None:
        self.activation_outputs.append(output)
        if hasattr(self.layer, "_nb_pre_mem"):
            self.pre_fire_mem_potential.extend(self.layer._nb_pre_mem)
        if hasattr(self.layer, "_nb_post_mem"):
            self.post_fire_mem_potential.extend(self.layer._nb_post_mem)

    def reset(self) -> None:
        self.activation_outputs.clear()
        self.activation_inputs.clear()
        self.pre_fire_mem_potential.clear()
        self.post_fire_mem_potential.clear()

    def close(self) -> None:
        if self.hook:
            self.hook.remove()
        if self.hook_pre:
            self.hook_pre.remove()
