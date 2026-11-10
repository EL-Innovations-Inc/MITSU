"""MITSU (三つ巴) — canonical reference implementation (numpy).

Provenance
----------
Mother function (three symmetric variables trading exponents):

    f(x, y, z) = (x^(y+z) + y^(x+z) + z^(x+y)) / (x + y + z)

Fixing two variables at the dial values a = b = 1, substituting the neuron's
pre-activation through x = s = softplus(u) (exponentiation needs a positive
base), and subtracting the resting level f(0+, 1, 1) = 1 collapses the whole
expression to a single rational form in s:

    MITSU(u) = s(s - 1) / (s + 2),        s = softplus(u)

Properties (all verified in experiments/):
  * smooth and non-monotonic: minimum ≈ -0.101 at u ≈ -0.57 (s* = sqrt(6) - 2)
  * asymptotically linear: MITSU(u) -> u - 3 as u -> +inf, -> 0 as u -> -inf
  * derivative bounded in (-0.5, 1) — flatter-tailed than GELU/SiLU, the basis
    of the (open) loss-spike stability hypothesis "S2" in docs/PLAN.md

This module is intentionally dependency-light (numpy only) and is the single
source of truth for the math. PyTorch drop-ins live in torch/.
"""
import numpy as np


def softplus(u):
    """Numerically safe softplus: log(1 + e^u).

    For u > 30 the identity softplus(u) ~= u is exact to double precision and
    avoids overflow in exp; the inner minimum() guards the same overflow for
    the values that do go through the exact formula.
    """
    return np.where(u > 30, u, np.log1p(np.exp(np.minimum(u, 30))))


def mitsu(u):
    """MITSU activation, centered a = b = 1 form. Elementwise, any shape."""
    s = softplus(u)
    return s * (s - 1) / (s + 2)


def mitsu_d(u):
    """Exact derivative dMITSU/du.

    Chain rule through s = softplus(u):
        dF/ds = (s^2 + 4s - 2) / (s + 2)^2      (bounded in (-0.5, 1))
        ds/du = sigmoid(u)
    The clip inside the sigmoid only prevents overflow warnings; it cannot
    change any value at double precision.
    """
    s = softplus(u)
    return (s * s + 4 * s - 2) / ((s + 2) ** 2) / (1 + np.exp(-np.clip(u, -60, 60)))
