"""Unit tests for the reference implementation (mitsu.py) and the PyTorch
drop-ins (torch/). Run from the repository root:  pytest -q
"""
import importlib.util
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mitsu import mitsu, mitsu_d, softplus  # noqa: E402

U = np.linspace(-20.0, 20.0, 4001)


def _load(name):
    """Import a file from torch/ by path (the folder is not a package)."""
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "torch", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_is_the_mother_function_with_two_seats_at_one():
    s = softplus(U)
    x, y, z = s, 1.0, 1.0
    f = (x ** (y + z) + y ** (x + z) + z ** (x + y)) / (x + y + z)
    assert np.allclose(f - 1.0, mitsu(U), atol=1e-12)


def test_dip_location_and_depth():
    s_star = math.sqrt(6) - 2                     # where dF/ds = 0
    u_star = math.log(math.expm1(s_star))         # softplus inverse
    depth = s_star * (s_star - 1) / (s_star + 2)
    assert abs(u_star - (-0.5665)) < 1e-3
    assert abs(depth - (-0.10102)) < 1e-5
    fine = np.linspace(-2, 1, 300001)
    assert abs(mitsu(fine).min() - depth) < 1e-9
    assert abs(fine[mitsu(fine).argmin()] - u_star) < 1e-4


def test_limits():
    assert abs(mitsu(np.array(-40.0))) < 1e-12              # silence -> 0
    big = np.array(1e6)
    assert abs(mitsu(big) - (big - 3)) < 1e-5                # tail -> u - 3
    extreme = np.array([-1e4, -745.0, 0.0, 745.0, 1e4])
    with np.errstate(over="raise", invalid="raise", divide="raise"):   # no overflow, no NaN
        assert np.isfinite(mitsu(extreme)).all()
        assert np.isfinite(mitsu_d(extreme)).all()


def test_derivative_matches_finite_differences():
    h = 1e-5
    fd = (mitsu(U + h) - mitsu(U - h)) / (2 * h)
    assert np.max(np.abs(fd - mitsu_d(U))) < 1e-8


def test_derivative_is_bounded():
    wide = np.linspace(-60.0, 60.0, 240001)
    d = mitsu_d(wide)
    assert d.min() > -0.5 and d.max() < 1.0                  # the stated bound
    assert d.min() > -0.05                                   # what it actually reaches
    assert abs(mitsu_d(np.array(0.0)) - 0.0864) < 1e-3       # flat at the origin


def test_torch_module_matches_reference():
    pytest.importorskip("torch.nn")  # local torch/ can otherwise look importable without PyTorch
    import torch
    MITSU = _load("mitsu_torch").MITSU
    u = torch.linspace(-20, 20, 4001, dtype=torch.float64, requires_grad=True)
    y = MITSU()(u)
    assert np.allclose(y.detach().numpy(), mitsu(u.detach().numpy()), atol=1e-12)
    y.sum().backward()
    assert np.allclose(u.grad.numpy(), mitsu_d(u.detach().numpy()), atol=1e-10)
    assert sum(p.numel() for p in MITSU().parameters()) == 0


def test_mitsuglu_is_the_gated_form():
    pytest.importorskip("torch.nn")  # local torch/ can otherwise look importable without PyTorch
    import torch
    glu = _load("mitsu_glu")
    torch.manual_seed(0)
    m = glu.MitsuGLU(dim=16, hidden_dim=40, gate_mult=3.04).double()
    x = torch.randn(5, 7, 16, dtype=torch.float64)
    want = m.w2(3.04 * glu.mitsu(m.w1(x)) * m.w3(x))
    assert m(x).shape == (5, 7, 16)
    assert torch.allclose(m(x), want)
    assert all(p.grad is None for p in m.parameters())
    m(x).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_gate_mult_value():
    """std(SiLU(z)) / std(MITSU(z)) for z ~ N(0,1): the documented 3.04."""
    z = np.random.default_rng(0).standard_normal(2_000_000)
    silu = z / (1 + np.exp(-z))
    assert abs(silu.std() / mitsu(z).std() - 3.04) < 0.02
