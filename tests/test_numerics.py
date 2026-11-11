"""Numerical regressions including the v1.0.0 float16 overflow at u=257."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mitsu import mitsu as np_mitsu, mitsu_d, softplus


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_numpy_finite_near_dtype_limits(dtype):
    largest = np.finfo(dtype).max
    x = np.array([-largest, -100, 0, 257, 300, largest], dtype=dtype)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for fn in (softplus, np_mitsu, mitsu_d):
            y = fn(x)
            assert y.dtype == dtype
            assert np.isfinite(y).all()
    assert mitsu_d(x)[-1] == pytest.approx(1)


def test_numpy_negative_tail_derivative_is_not_clipped():
    # The old clipped sigmoid assigned an artificial derivative around -4e-27.
    assert mitsu_d(np.array(-100.0)) == pytest.approx(-0.5 * np.exp(-100.0), rel=1e-12, abs=0)
    assert mitsu_d(np.array(-1000.0)) == 0


def test_numpy_scalar_and_integer_inputs():
    assert np_mitsu(300) == pytest.approx(np_mitsu(300.0))
    assert np_mitsu([0, 1, 2]).shape == (3,)
    with pytest.raises(TypeError):
        np_mitsu(np.array([1 + 2j]))


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32", "float64"])
def test_torch_finite_outputs_and_gradients_near_dtype_limits(dtype_name):
    pytest.importorskip("torch.nn")
    import torch
    from mitsu_torch import mitsu

    dtype = getattr(torch, dtype_name)
    largest = torch.finfo(dtype).max
    x = torch.tensor([-largest, -100, 0, 255, 256, 257, 300, 1000, largest], dtype=dtype, requires_grad=True)
    y = mitsu(x)
    y.sum().backward()
    assert y.dtype == dtype and y.device == x.device
    assert torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()
    assert x.grad[-1] == 1
    # Compare against high-precision evaluation rounded to the destination dtype.
    ref = torch.from_numpy(np_mitsu(x.detach().double().numpy())).to(dtype)
    tol = 8 * torch.finfo(dtype).eps
    torch.testing.assert_close(y.detach(), ref, rtol=tol, atol=tol)


def test_torch_gradcheck_and_gradgradcheck():
    pytest.importorskip("torch.nn")
    import torch
    from mitsu_torch import mitsu

    x = torch.tensor([-100, -3, -0.57, 0, 1, 10, 257, 1000], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(mitsu, (x,), eps=1e-5, atol=1e-8, rtol=1e-4)
    assert torch.autograd.gradgradcheck(mitsu, (x,), eps=1e-5, atol=1e-8, rtol=1e-4)
    with pytest.raises(TypeError):
        mitsu(torch.tensor([1, 2]))


@pytest.mark.parametrize("filename", ["mitsu_torch.py", "mitsu_glu.py"])
def test_standalone_copies_match_installable_api(filename):
    pytest.importorskip("torch.nn")
    import torch
    from mitsu_torch import mitsu

    path = Path(__file__).resolve().parents[1] / "torch" / filename
    spec = importlib.util.spec_from_file_location("standalone_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for dtype in (torch.float16, torch.bfloat16, torch.float64):
        x = torch.tensor([-100, -2, 0, 3, 257, 300, 1000], dtype=dtype, requires_grad=True)
        y = module.mitsu(x)
        torch.testing.assert_close(y, mitsu(x), rtol=0, atol=0)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32", "float64"])
def test_glu_representative_forward_and_backward(dtype_name):
    pytest.importorskip("torch.nn")
    import torch
    from mitsu_torch import MitsuGLU

    dtype = getattr(torch, dtype_name)
    torch.manual_seed(17)
    layer = MitsuGLU(16, 32, gate_mult=3.04).to(dtype=dtype)
    x = torch.randn(4, 7, 16, dtype=dtype, requires_grad=True)
    y = layer(x)
    y.float().square().mean().backward()
    assert y.dtype == dtype and torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())


def test_cpu_autocast_keeps_activation_input_dtype():
    pytest.importorskip("torch.nn")
    import torch
    from mitsu_torch import MITSU

    x = torch.tensor([257, 300, 1000], dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = MITSU()(x)
    y.sum().backward()
    assert y.dtype == x.dtype
    assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
