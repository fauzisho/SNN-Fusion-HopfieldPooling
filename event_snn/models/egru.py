# Copyright (c) 2023  Khaleelulla Khan Nazeer
# This file incorporates work covered by the following copyright:
# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Event-based Gated Recurrent Unit (EGRU) — PyTorch reference (wearsnn/emmanuela)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseRNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, batch_first, zoneout, return_state_sequence
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.zoneout = zoneout
        self.return_state_sequence = return_state_sequence

    def _permute(self, x):
        if self.batch_first:
            return x.permute(1, 0, 2)
        return x

    def _get_state(self, input, state, state_shape):
        if state is None:
            state = _zero_state(input, state_shape)
        else:
            _validate_state(state, state_shape)
        return state

    def _get_zoneout_mask(self, input):
        if self.zoneout:
            zoneout_mask = input.new_empty(
                input.shape[0], input.shape[1], self.hidden_size
            )
            zoneout_mask.bernoulli_(1.0 - self.zoneout)
        else:
            zoneout_mask = input.new_empty(0, 0, 0)
        return zoneout_mask


def _validate_state(state, state_shape):
    if isinstance(state, (tuple, list)):
        if not isinstance(state_shape, (tuple, list)):
            raise ValueError(
                "RNN state has invalid structure; expected {}".format(state_shape)
            )
        for s, ss in zip(state, state_shape):
            _validate_state(s, ss)
    else:
        shape = list(state.size())
        if len(shape) != len(state_shape):
            raise ValueError(
                "RNN state dimension mismatch; expected {} got {}".format(
                    len(state_shape), len(shape)
                )
            )
        for i, (d1, d2) in enumerate(zip(list(state.size()), state_shape)):
            if d2 is not None and d1 != d2:
                raise ValueError(
                    "RNN state size mismatch on dim {}; expected {} got {}".format(
                        i, d2, d1
                    )
                )


def _zero_state(input, state_shape):
    if isinstance(state_shape, (tuple, list)) and isinstance(state_shape[0], int):
        state = input.new_zeros(*state_shape)
    elif isinstance(state_shape, tuple):
        state = tuple(_zero_state(input, s) for s in state_shape)
    elif isinstance(state_shape, list):
        state = [_zero_state(input, s) for s in state_shape]
    else:
        raise ValueError("RNN state_shape is invalid")
    return state


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, dampening_factor, pseudo_derivative_support):
        ctx.save_for_backward(inp, dampening_factor, pseudo_derivative_support)
        return torch.heaviside(inp, torch.zeros_like(inp))

    @staticmethod
    def backward(ctx, grad_output):
        inp, dampening_factor, pseudo_derivative_support = ctx.saved_tensors
        dE_dz = grad_output
        zero = torch.zeros((), device=grad_output.device, dtype=grad_output.dtype)
        dz_du = dampening_factor * torch.maximum(
            1 - pseudo_derivative_support * torch.abs(inp),
            zero,
        )
        dE_dv = dE_dz * dz_du
        return dE_dv, None, None


def EGRUScript(
    training: bool,
    zoneout_prob: float,
    dampening_factor: float,
    pseudo_derivative_support: float,
    input,
    h0,
    kernel,
    recurrent_kernel,
    bias,
    recurrent_bias,
    thr,
    zoneout_mask,
):
    time_steps = input.shape[0]

    h = [torch.zeros_like(h0)]
    o = [torch.zeros_like(h0)]
    y = [h0]
    Wx = input @ kernel + bias

    for t in range(time_steps):
        Rh = y[t] @ recurrent_kernel + recurrent_bias
        vx = torch.chunk(Wx[t], 3, 1)
        vh = torch.chunk(Rh, 3, 1)

        z = torch.sigmoid(vx[0] + vh[0])
        r = torch.sigmoid(vx[1] + vh[1])
        g = torch.tanh(vx[2] + r * vh[2])

        cur_h = z * h[t] + (1 - z) * g
        if zoneout_prob:
            if training:
                cur_h = (cur_h - h[t]) * zoneout_mask[t] + h[t]
            else:
                cur_h = zoneout_prob * h[t] + (1 - zoneout_prob) * h[t]

        event = SpikeFunction.apply(
            cur_h - thr, dampening_factor, pseudo_derivative_support
        )
        o.append(event)
        h.append(cur_h - event * thr)
        y.append(event * cur_h)

    y = torch.stack(y)
    h = torch.stack(h)
    o = torch.stack(o)

    tr_vals = torch.zeros_like(y)
    alpha = 0.9
    for t in range(1, time_steps + 1):
        tr_vals[t] = alpha * tr_vals[t - 1] + (1 - alpha) * y[t]

    return y, h, o, tr_vals


class EGRU(BaseRNN):
    """Event-based Gated Recurrent Unit layer."""

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_first=False,
        dropout=0.0,
        zoneout=0.0,
        dampening_factor=0.7,
        pseudo_derivative_support=1.0,
        thr_mean=0.3,
        weight_initialization_gain=1.0,
        return_state_sequence=False,
        grad_clip=None,
    ):
        super().__init__(
            input_size, hidden_size, batch_first, zoneout, return_state_sequence
        )

        if grad_clip:
            self.grad_clip_norm(enable=True, norm=grad_clip)
        else:
            self.grad_clip_norm(False)

        self.dropout = dropout
        self.weight_initialization_gain = weight_initialization_gain

        self.kernel = nn.Parameter(torch.empty(input_size, hidden_size * 3))
        self.recurrent_kernel = nn.Parameter(torch.empty(hidden_size, hidden_size * 3))
        self.bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.recurrent_bias = nn.Parameter(torch.empty(hidden_size * 3))
        self.reset_parameters()

        self.dampening_factor = nn.Parameter(
            torch.Tensor([dampening_factor]), requires_grad=False
        )
        self.pseudo_derivative_support = nn.Parameter(
            torch.Tensor([pseudo_derivative_support]), requires_grad=False
        )

        assert 0 < thr_mean < 1, f"thr_mean must be between 0 and 1, got {thr_mean}"
        beta = 3
        alpha = beta * thr_mean / (1 - thr_mean)
        distribution = torch.distributions.beta.Beta(alpha, beta)
        self.thr = nn.Parameter(distribution.sample(torch.Size([self.hidden_size])))

    def reset_parameters(self):
        for k, v in self.named_parameters():
            if k in ["kernel", "recurrent_kernel", "bias", "recurrent_bias"]:
                if v.data.ndimension() >= 2:
                    nn.init.xavier_normal_(v, gain=self.weight_initialization_gain)
                else:
                    nn.init.zeros_(v)

    def grad_clip_norm(self, enable=True, norm=2.0):
        self._enable_grad_clip = enable
        self._max_norm = nn.Parameter(
            torch.Tensor([norm if enable else -1.0]), requires_grad=False
        )

    def forward(self, input, state=None, lengths=None):
        input = self._permute(input)
        state_shape = [1, input.shape[1], self.hidden_size]
        h0 = self._get_state(input, state, state_shape)

        self.thr.data.clamp_(min=0.0, max=1.0)

        y, h, o, trace = EGRUScript(
            self.training,
            self.zoneout,
            self.dampening_factor,
            self.pseudo_derivative_support,
            input.contiguous(),
            h0[0].contiguous(),
            self.kernel.contiguous(),
            F.dropout(
                self.recurrent_kernel, self.dropout, self.training
            ).contiguous(),
            self.bias.contiguous(),
            self.recurrent_bias.contiguous(),
            self.thr.contiguous(),
            self._get_zoneout_mask(input).contiguous(),
        )

        output = self._permute(y[1:])
        return output
