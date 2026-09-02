"""MITSU (三つ巴) — reference implementation, numpy.
Mother function: f(x,y,z) = (x^(y+z) + y^(x+z) + z^(x+y)) / (x+y+z)
Activation (centered, a=b=1):  MITSU(u) = s(s-1)/(s+2),  s = softplus(u)
Derivative bounded in (-0.5, 1). See torch/ for PyTorch drop-ins.
"""
import numpy as np

def softplus(u):
    return np.where(u > 30, u, np.log1p(np.exp(np.minimum(u, 30))))

def mitsu(u):
    s = softplus(u)
    return s * (s - 1) / (s + 2)

def mitsu_d(u):
    s = softplus(u)
    return (s * s + 4 * s - 2) / ((s + 2) ** 2) / (1 + np.exp(-np.clip(u, -60, 60)))
