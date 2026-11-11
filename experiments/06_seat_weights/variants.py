"""MITSU seat weights — the c family.

Weighted mother function (weights i, j, k on the three seats):

    f(x, y, z) = (i x^(y+z) + j y^(x+z) + k z^(x+y)) / (i x + j y + k z)

With the signal in seat x as s = softplus(u) and seats y = z = 1, f rests at 1
for a silent input, and subtracting that resting level leaves

    MITSU_c(u) = s (s - 1) / (s + c),        c = (j + k) / i

so the three weights act through one number. c = 2 is current MITSU.
Small c: deeper dip, steeper near zero. Large c: shallower, quieter, and the
knee where the curve turns linear moves outward (tail -> u - (1 + c)).

    fixed c     c = 0.25, 0.5, 1, 2 (= mitsu), 4, 8
    learnc      one learned c per layer, c = exp(theta), starts at exactly 2,
                kept inside [0.05, 50]

`specs()` and `Layer` have the same interface as experiments/04_dial_family,
so the same two bench scripts run unchanged. Plain autograd, no custom backward.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

C_VALUES = [0.25, 0.5, 1.0, 4.0, 8.0]      # c = 2 is the baseline, named "mitsu"
C_MIN, C_MAX = 0.05, 50.0                  # range allowed to the learned c


def mitsu_c(u, c):
    """s(s-1)/(s+c), s = softplus(u). c: positive float or tensor."""
    s = F.softplus(u)
    return s * (s - 1) / (s + c)


def weighted_mother(u, i, j, k):
    """Centred weighted mother function with seats y = z = 1 (used by the gate)."""
    s = F.softplus(u)
    return (i * s ** 2 + j + k) / (i * s + j + k) - 1.0


def specs():
    out = {"mitsu": dict(kind="std", fn=lambda u: mitsu_c(u, 2.0))}
    for c in C_VALUES:
        out[f"c={c:g}"] = dict(kind="std", fn=(lambda u, c=c: mitsu_c(u, c)))
    out["learnc"] = dict(kind="learn", init=2.0)
    return out


class Layer(nn.Module):
    """input (.., d_in) -> activations (.., width). Same call signature as the
    Layer of experiment 04; `init_w(out, in)` supplies the weight init so every
    variant starts from identical weights for a given seed."""

    def __init__(self, spec, d_in, width, init_w, bias=True, init_aux=None):
        super().__init__()
        self.spec, self.kind, self.width = spec, spec["kind"], width
        self.record, self.last_ab = False, None
        self.fc = nn.Linear(d_in, width, bias=bias)
        with torch.no_grad():
            self.fc.weight.copy_(init_w(width, d_in))
            if bias:
                self.fc.bias.zero_()
        if self.kind == "learn":
            self.theta = nn.Parameter(torch.tensor(math.log(spec["init"])))

    def c(self):
        return torch.exp(self.theta.clamp(math.log(C_MIN), math.log(C_MAX)))

    def dials(self):
        """Current learned c (learnc only), as a one-element list."""
        return [float(self.c().detach())]

    def forward(self, x):
        if self.kind == "std":
            return self.spec["fn"](self.fc(x))
        return mitsu_c(self.fc(x), self.c())
