"""SE-AdLIF neurons (emmanuela / IGITUGraz SE-adlif, arXiv:2408.07517)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter


def get_tau_trainer_class(name: str):
    if name == "interpolation":
        return InterpolationTrainer
    if name == "fixed":
        return FixedTau
    raise ValueError(f"Invalid tau trainer name: {name}")


class TauTrainer(Module):
    __constants__ = ["in_features"]
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        dt: float,
        tau_min: float,
        tau_max: float,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__(**kwargs)
        self.dt = dt
        self.weight = Parameter(torch.empty(in_features, **factory_kwargs))
        self.register_buffer("tau_max", torch.tensor(tau_max, **factory_kwargs))
        self.register_buffer("tau_min", torch.tensor(tau_min, **factory_kwargs))

    def reset_parameters(self) -> None:
        raise NotImplementedError

    def apply_parameter_constraints(self) -> None:
        raise NotImplementedError

    def forward(self) -> torch.Tensor:
        raise NotImplementedError

    def get_tau(self) -> torch.Tensor:
        raise NotImplementedError

    def get_decay(self):
        return self.forward()


class FixedTau(TauTrainer):
    def apply_parameter_constraints(self):
        pass

    def forward(self):
        return torch.exp(-self.dt / self.get_tau())

    def get_tau(self):
        return self.weight

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight, a=self.tau_min, b=self.tau_max)
        self.weight.requires_grad = False


class InterpolationTrainer(TauTrainer):
    def apply_parameter_constraints(self):
        with torch.no_grad():
            self.weight.clamp_(0.0, 1.0)

    def forward(self):
        return torch.exp(-self.dt / self.get_tau())

    def get_tau(self):
        return self.weight * self.tau_max + (1.0 - self.weight) * self.tau_min

    def reset_parameters(self):
        torch.nn.init.uniform_(self.weight)
        self.weight.requires_grad = True


def SLAYER(x, alpha, c):
    return c * alpha / (2 * torch.exp(x.abs() * alpha))


@torch.jit.script
def SEAdLIFScript(
    u_tm1: Tensor,
    z_tm1: Tensor,
    w_tm1: Tensor,
    soma_current: Tensor,
    decay_u: Tensor,
    decay_w: Tensor,
    thr: Tensor,
    alpha: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    q: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    u_t = decay_u * u_tm1 + (1.0 - decay_u) * (soma_current - w_tm1)
    u_thr = u_t - thr
    z_t = (
        torch.heaviside(u_thr, torch.as_tensor(0.0).type(u_thr.dtype)).detach()
        + (u_thr - u_thr.detach()) * SLAYER(u_thr, alpha, c).detach()
    )
    u_t = u_t * (1 - z_t.detach())
    w_t = decay_w * w_tm1 + (1.0 - decay_w) * (a * u_t + b * z_t) * q
    return u_t, z_t, w_t


class EFAdLIF(Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int

    def __init__(
        self,
        cfg,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__(**kwargs)
        self.in_features = cfg["input_size"]
        self.out_features = cfg["n_neurons"]
        self.dt = 1.0
        self.register_buffer("thr", torch.tensor(cfg.get("thr", 1.0), **factory_kwargs))
        self.register_buffer("alpha", torch.tensor(cfg.get("alpha", 5.0), **factory_kwargs))
        self.register_buffer("c", torch.tensor(cfg.get("c", 0.4), **factory_kwargs))
        self.tau_u_range = cfg["tau_u_range"]
        self.train_tau_u_method = "interpolation"
        self.tau_w_range = cfg["tau_w_range"]
        self.train_tau_w_method = "interpolation"
        self.use_recurrent = cfg.get("use_recurrent", True)
        self.with_linear = cfg.get("with_linear", True)
        self.a_range = [0.0, 1.0]
        self.b_range = [0.0, 2.0]
        self.register_buffer("q", torch.tensor(cfg["q"], **factory_kwargs))

        self.tau_u_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_u_method)(
            self.out_features,
            self.dt,
            self.tau_u_range[0],
            self.tau_u_range[1],
            **factory_kwargs,
        )
        self.tau_w_trainer: TauTrainer = get_tau_trainer_class(self.train_tau_w_method)(
            self.out_features,
            self.dt,
            self.tau_w_range[0],
            self.tau_w_range[1],
            **factory_kwargs,
        )

        if self.with_linear:
            self.weight = Parameter(
                torch.empty((self.out_features, self.in_features), **factory_kwargs)
            )
            self.bias = Parameter(torch.empty(self.out_features, **factory_kwargs))

        if self.use_recurrent:
            self.recurrent = Parameter(
                torch.empty((self.out_features, self.out_features), **factory_kwargs)
            )
        else:
            self.register_buffer("recurrent", None)

        self.a = Parameter(torch.empty(self.out_features, **factory_kwargs))
        self.b = Parameter(torch.empty(self.out_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.tau_u_trainer.reset_parameters()
        self.tau_w_trainer.reset_parameters()
        if self.with_linear:
            torch.nn.init.uniform_(
                self.weight,
                -1.0 * torch.sqrt(1 / torch.tensor(self.in_features)),
                torch.sqrt(1 / torch.tensor(self.in_features)),
            )
            torch.nn.init.zeros_(self.bias)
        if self.use_recurrent:
            torch.nn.init.orthogonal_(self.recurrent, gain=1.0)
        torch.nn.init.uniform_(self.a, self.a_range[0], self.a_range[1])
        torch.nn.init.uniform_(self.b, self.b_range[0], self.b_range[1])

    def initial_state(self, x, device) -> Tensor:
        u = torch.zeros_like(x, device=device, dtype=torch.float, requires_grad=True)
        z = torch.zeros_like(x, device=device, dtype=torch.float, requires_grad=True)
        w = torch.zeros_like(x, device=device, dtype=torch.float, requires_grad=True)
        return u, z, w

    def apply_parameter_constraints(self):
        self.tau_u_trainer.apply_parameter_constraints()
        self.tau_w_trainer.apply_parameter_constraints()
        self.a.data = torch.clamp(self.a, min=self.a_range[0], max=self.a_range[1])
        self.b.data = torch.clamp(self.b, min=self.b_range[0], max=self.b_range[1])

    def forward(
        self, input_tensor: Tensor, states: Tuple[Tensor, Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        u_tm1, z_tm1, w_tm1 = states
        decay_u = self.tau_u_trainer.get_decay()
        decay_w = self.tau_w_trainer.get_decay()
        if self.with_linear:
            soma_current = F.linear(input_tensor, self.weight, self.bias)
        else:
            soma_current = input_tensor
        if self.use_recurrent:
            soma_current = soma_current + F.linear(z_tm1, self.recurrent, None)

        if len(soma_current.shape) == 4:
            soma_current = soma_current.permute(0, 2, 3, 1)
        if len(u_tm1.shape) == 4:
            u_tm1 = u_tm1.permute(0, 2, 3, 1)
            z_tm1 = z_tm1.permute(0, 2, 3, 1)
            w_tm1 = w_tm1.permute(0, 2, 3, 1)

        u_t, z_t, w_t = SEAdLIFScript(
            u_tm1,
            z_tm1,
            w_tm1,
            soma_current,
            decay_u,
            decay_w,
            self.thr,
            self.alpha,
            self.a,
            self.b,
            self.c,
            self.q,
        )

        if len(u_t.shape) == 4:
            u_t = u_t.permute(0, 3, 1, 2)
            z_t = z_t.permute(0, 3, 1, 2)
            w_t = w_t.permute(0, 3, 1, 2)

        return z_t.clone(), (u_t, z_t, w_t)


class SEAdLIF(EFAdLIF):
    pass


class AdLIF(Module):
    """Multi-step SE-AdLIF wrapper for (T, B, C, H, W) tensors."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        threshold: float,
        rnn: bool = False,
        dtype=None,
        device: torch.device | str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = {
            "input_size": in_features,
            "n_neurons": out_features,
            "thr": threshold,
            "alpha": 5.0,
            "c": 0.4,
            "tau_u_range": [5, 25],
            "tau_w_range": [60, 300],
            "use_recurrent": rnn,
            "with_linear": False,
            "q": 120,
        }
        self.neuron = SEAdLIF(self.cfg, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        self.neuron.apply_parameter_constraints()
        u, z, w = self.neuron.initial_state(x[0], x.device)
        states = (u, z, w)
        outs = []
        for t in range(x.shape[0]):
            out, states = self.neuron(x[t], states)
            outs.append(out)
        return torch.stack(outs, dim=0)
