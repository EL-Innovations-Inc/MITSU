"""MITSU for PyTorch — drop-in replacement for nn.GELU / nn.SiLU.

Centered a = b = 1 form:  MITSU(u) = s(s - 1) / (s + 2),  s = softplus(u).
Derivative bounded in (-0.5, 1). Autograd differentiates the closed form, so
no custom backward is needed; the math is validated against hand-derived
gradients (finite-difference checked to ~1e-6) in the numpy twin,
experiments/02_char_gpt/mitsu_gpt.py.

Usage — nanoGPT (model.py, class MLP):
    self.gelu = nn.GELU()          # before
    self.gelu = MITSU()            # after — nothing else changes

The module is stateless and shape-agnostic; softplus is evaluated in the
tensor's own dtype and is bf16/fp16-safe (F.softplus handles the large-input
linear regime internally).
"""
import torch.nn as nn
import torch.nn.functional as F


class MITSU(nn.Module):
    """Elementwise MITSU activation. No parameters, no buffers."""

    def forward(self, u):
        s = F.softplus(u)
        return s * (s - 1) / (s + 2)
