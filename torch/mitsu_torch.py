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


class MITSU(nn.Module):
    """Stateless MITSU activation; no parameters or buffers."""

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return mitsu(u)

