"""Standalone compatibility copy for projects that copy this file directly.

The installable API is ``from mitsu_torch import MITSU, MitsuGLU, mitsu``.
This snapshot uses the same stable formula and is checked against that API.
Float16/bfloat16 intermediate arithmetic is promoted to float32, with output
cast back to input dtype. This is an activation fix, not a guarantee that the
surrounding GLU or network can never overflow.
"""
import torch
from torch import nn
from torch.nn import functional as F


def mitsu(u: torch.Tensor) -> torch.Tensor:
    """Elementwise s(s-1)/(s+2), s=softplus(u), without an s**2 intermediate.

    Divide before multiplying and promote low-precision intermediates. No
    input or output clipping is used. Autograd follows these exact operations.
    Floating tensors are required; output dtype and device match the input.
    """
    if not u.is_floating_point():
        raise TypeError("MITSU expects a floating-point tensor")
    work = u.float() if u.dtype in (torch.float16, torch.bfloat16) else u
    s = F.softplus(work)
    return (s * ((s - 1) / (s + 2))).to(dtype=u.dtype)


class MitsuGLU(nn.Module):
    """W2(gate_mult * MITSU(W1 x) * W3 x).

    gate_mult=1.0 evaluates the raw gate. Approximately 3.04 matches the gate
    standard deviation to SiLU for standard-normal inputs, as in the pilot.
    The multiplier scales gate derivatives too; neither the product nor the
    full block has the raw scalar activation's derivative bound.
    """

    def __init__(self, dim: int, hidden_dim: int, gate_mult: float = 1.0, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.gate_mult = gate_mult

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.gate_mult * mitsu(self.w1(x)) * self.w3(x))
