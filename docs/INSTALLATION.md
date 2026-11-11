# Install and reproduce v1.0.1

The distribution is **mitsu-activation**. The imports are `mitsu` (NumPy)
and `mitsu_torch` (PyTorch). Python 3.10 or later is required. NumPy is the
only dependency installed by the base package. The ZIP is a source release;
these instructions do not imply that a PyPI upload has taken place.

## Minimal NumPy installation

From the extracted repository root, preferably in a new virtual environment:

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install .
python -c "from mitsu import mitsu; print(mitsu(1.0))"
```

For local development, replace `pip install .` with `pip install -e .`.

## PyTorch modules

Choose the PyTorch build that fits the machine first. This tested CPU command
avoids installing CUDA libraries on machines that do not need them:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[torch]'
```

On Windows PowerShell, use double quotes around `".[torch]"` if needed.
For NVIDIA GPU installations, select the appropriate official PyTorch wheel
for the local driver; the CPU snapshot below is not a CUDA environment.

```python
import torch
from mitsu_torch import MITSU, MitsuGLU, mitsu

x = torch.tensor([0.0, 1.0, 257.0], dtype=torch.float16, requires_grad=True)
y = MITSU()(x)
y.sum().backward()
ffn = MitsuGLU(dim=64, hidden_dim=170, gate_mult=3.04)
```

The legacy files under `torch/` remain available for copy-and-paste use. Prefer
the installed package for new integrations. `torch/` is not the import path
for the MITSU package.

## Experiments and development

```bash
python -m pip install -e '.[torch,dev]'
python -m pytest -q
python experiments/04_dial_family/gate.py

# Includes scikit-learn and matplotlib for all historical experiments/figures:
python -m pip install -e '.[experiments,dev]'
```

`python -m pip install -r requirements.txt` is the equivalent full, editable
source-tree installation. Dataset downloads remain separate and are described
in each experiment folder. They require network access.

## Exact tested dependency snapshots

The release was checked on **Linux x86_64, CPython 3.12.14, CPU PyTorch**.
The pinned files include the installed transitive dependencies for these
scopes. They are platform-specific snapshots, not universal lockfiles, and
contain no claim of equivalent testing on Windows, macOS, CUDA or another
Python version.

| File | Scope |
|---|---|
| `requirements-core-lock-py312.txt` | NumPy-only runtime |
| `requirements-dev-lock-cpu-py312.txt` | NumPy, SciPy, CPU PyTorch, pytest, and build tools |
| `requirements-lock-cpu-py312.txt` | Development scope plus all experiment and figure dependencies |

To reproduce the full snapshot in a fresh Python 3.12 environment:

```bash
python -m pip install -r requirements-lock-cpu-py312.txt
python -m pip install --no-deps -e .
python -m pytest -q
python experiments/04_dial_family/gate.py
```

Use the minimal or development snapshot instead when that is all you need.
The development snapshot adds the CPU PyTorch index to resolve the exact
`+cpu` wheel. PyTorch binaries and other platform wheels are external downloads;
they are not bundled in this ZIP. The flexible dependency ranges in
`pyproject.toml` serve other supported platforms. CI is configured to run
Python 3.10, 3.11 and 3.12 on Linux; a configured CI matrix is not a claim that
those remote jobs have already run.

## Build and verify the installable release

```bash
python -m pip install -e '.[torch,dev]'
python -m build
python scripts/smoke_wheel.py dist/mitsu_activation-1.0.1-py3-none-any.whl
```

For the pinned build tool versions already installed from the snapshot, use
`python -m build --no-isolation`. The default build command uses a fresh build
environment and resolves the build requirements from `pyproject.toml`.

The smoke check force-installs the wheel into a temporary virtual environment,
runs Python in isolated mode from outside the repository, verifies the imported
files come from that environment, and checks NumPy/PyTorch forward values,
autograd and the MitsuGLU shape. It reuses installed dependency packages so
that this verification itself needs no package-index access.
