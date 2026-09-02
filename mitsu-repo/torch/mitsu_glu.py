"""MitsuGLU — SwiGLU with the SiLU gate replaced by MITSU (centered, a=b=1).
Drop-in for TorchTitan / Llama-style FeedForward. Only the nonlinearity differs
from Llama 3; W1/W3/W2 shapes and init are untouched.

gate_mult: fixed post-activation multiplier to match SiLU's signal variance at
init (MITSU is flatter at the origin). Tune once at 50M with muP coord checks
(typical value 2-4), then it transfers. Set 1.0 to test the raw formula.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mitsu(u: torch.Tensor) -> torch.Tensor:
    s = F.softplus(u)
    return s * (s - 1) / (s + 2)


class MitsuGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, gate_mult: float = 1.0, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)   # gate proj
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)   # up proj
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)   # down proj
        self.gate_mult = gate_mult

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.gate_mult * mitsu(self.w1(x)) * self.w3(x))


# TorchTitan patch: in the model's FeedForward construction, replace the SwiGLU
# module with MitsuGLU(dim, hidden_dim, gate_mult=GATE_MULT). Nothing else changes.
