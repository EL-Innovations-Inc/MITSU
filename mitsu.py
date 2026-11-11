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
source of truth for the math. The installable PyTorch package is mitsu_torch; standalone drop-ins live in torch/.
"""
import numpy as np


def _working_array(u):
    """Promote half precision for intermediate arithmetic; accept numeric scalars."""
    a = np.asarray(u)
    if a.dtype.kind not in "fiu":
        raise TypeError("MITSU expects real numeric inputs")
    if a.dtype.kind in "iu":
        return a.astype(np.float64), None
    return a.astype(np.float32) if a.dtype == np.float16 else a, a.dtype


def _restore(y, dtype):
    return y.astype(dtype, copy=False) if dtype is not None else y


def softplus(u):
    """Stable log(1 + exp(u)); float16 intermediates use float32.

    The result retains floating input dtype (integer inputs produce float64).
    Very negative inputs can underflow to zero at the destination precision.
    """
    a, dtype = _working_array(u)
    with np.errstate(under="ignore"):
        return _restore(np.logaddexp(0, a), dtype)


def mitsu(u):
    """Elementwise MITSU with overflow-safe rational evaluation.

    Dividing before multiplying avoids the s**2 intermediate in the original
    formula. For finite floating inputs the activation is finite, including
    large positive values near the dtype limit. No clipping is applied.
    """
    a, dtype = _working_array(u)
    with np.errstate(under="ignore"):
        s = np.logaddexp(0, a)
        return _restore(s * ((s - 1) / (s + 2)), dtype)


def mitsu_d(u):
    """Analytic derivative, stable without squaring a large softplus value.

    dF/ds = 1 - 6/(s + 2)**2; dsoftplus/du = sigmoid(u).
    Sigmoid is evaluated by sign, retaining the true negative tail rather
    than clipping the input to an artificial minimum derivative magnitude.
    """
    a, dtype = _working_array(u)
    with np.errstate(under="ignore"):
        s = np.logaddexp(0, a)
        inv = 1 / (s + 2)
        e = np.exp(-np.abs(a))
        sigmoid = np.where(a >= 0, 1 / (1 + e), e / (1 + e))
        return _restore((1 - 6 * inv * inv) * sigmoid, dtype)
