"""MitsuGLU — SwiGLU with the SiLU gate transplanted for MITSU.

Llama-family feed-forward blocks compute  W2( act(W1 x) * W3 x ).  This module
keeps that exact structure — same three projections, same shapes, same init
conventions as the host model — and changes only `act`, so a MitsuGLU network
is a controlled experiment on the nonlinearity alone.

gate_mult
---------
MITSU is much quieter than SiLU under a standard-normal input diet
(std ratio ≈ 3.04 measured over z ~ N(0,1); see experiments/03_scaling_ladder,
where the value is computed and frozen at setup). A fixed post-activation
multiplier restores the gate's signal variance at init. Tune it ONCE at the
smallest muP width via coordinate checks, then it transfers up the ladder.
Set gate_mult = 1.0 to study the raw, uncorrected formula.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mitsu(u: torch.Tensor) -> torch.Tensor:
    """Centered MITSU: s(s - 1)/(s + 2), s = softplus(u). Elementwise."""
    s = F.softplus(u)
    return s * (s - 1) / (s + 2)


class MitsuGLU(nn.Module):
    """Gated feed-forward: W2( gate_mult * MITSU(W1 x) ⊙ W3 x ).

    Args:
        dim:        model width (input and output features).
        hidden_dim: FFN width (Llama convention: ~8/3 * dim, rounded).
        gate_mult:  fixed scalar applied after MITSU on the gate path only.
        bias:       include biases on the three projections (Llama uses False).
    """

    def __init__(self, dim: int, hidden_dim: int, gate_mult: float = 1.0, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)   # gate projection (through MITSU)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)   # up projection (linear path)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)   # down projection
        self.gate_mult = gate_mult

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.gate_mult * mitsu(self.w1(x)) * self.w3(x))


# TorchTitan / Llama patch: construct MitsuGLU(dim, hidden_dim, gate_mult=G)
# wherever the model builds its SwiGLU FeedForward. Nothing else changes.
