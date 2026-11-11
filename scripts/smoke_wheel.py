"""Install a built wheel in a temporary environment and import outside the repo.

Usage: python scripts/smoke_wheel.py dist/mitsu_activation-1.0.1-py3-none-any.whl
The temporary venv reuses installed dependencies, but force-installs the wheel.
No package-index access is needed for this check.
"""
from pathlib import Path
import subprocess
import sys
import json
import site
import tempfile
import venv


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    wheel = Path(sys.argv[1]).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="mitsu-wheel-") as temp:
        root = Path(temp)
        env = root / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env)
        python = env / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps",
                        "--force-reinstall", str(wheel)], cwd=root, check=True)
        code = r'''
import importlib.metadata as metadata
import pathlib, sys, json
# Dependencies may live in the caller's user site, disabled by -I. Append only
# site-package directories; the wheel's own venv stays first on sys.path.
for path in json.loads(sys.argv[1]):
    if path not in sys.path:
        sys.path.append(path)
import numpy as np
import mitsu
import mitsu_torch
import torch
assert metadata.version("mitsu-activation") == "1.0.1"
for module in (mitsu, mitsu_torch):
    assert pathlib.Path(module.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix)), module.__file__
x = torch.tensor([-10., 0., 1., 257., 1000.], dtype=torch.float64, requires_grad=True)
y = mitsu_torch.MITSU()(x)
np.testing.assert_allclose(y.detach().numpy(), mitsu.mitsu(x.detach().numpy()), rtol=1e-12, atol=1e-12)
y.sum().backward()
assert torch.isfinite(x.grad).all()
assert mitsu_torch.MitsuGLU(8, 16)(torch.ones(2, 8)).shape == (2, 8)
print("WHEEL SMOKE: PASS (installed imports, forward, backward, MitsuGLU)")
'''
        dependency_paths = list(dict.fromkeys(site.getsitepackages() + [site.getusersitepackages()]))
        subprocess.run([str(python), "-I", "-c", code, json.dumps(dependency_paths)], cwd=root, check=True)


if __name__ == "__main__":
    main()
