"""MITSU for PyTorch — drop-in replacement for nn.GELU in any transformer MLP.
Centered a=b=1 form: s(s-1)/(s+2), s = softplus(u). Derivative bounded in (-0.5, 1).
Math validated by finite-difference gradcheck in the numpy twin (mitsu_gpt.py).

nanoGPT patch (model.py, class MLP): replace `self.gelu = nn.GELU()` with `MITSU()`.
"""
import torch.nn as nn
import torch.nn.functional as F

class MITSU(nn.Module):
    def forward(self, u):
        s = F.softplus(u)
        return s * (s - 1) / (s + 2)
