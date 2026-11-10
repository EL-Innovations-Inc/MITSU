# MITSU 三つ巴

**An activation function born from a symmetric inequality, taken from napkin to
scaling law — pre-registered at every step.**

```
mother function:   f(x,y,z) = (x^(y+z) + y^(x+z) + z^(x+y)) / (x+y+z)
activation:        MITSU(u) = s(s−1)/(s+2),   s = softplus(u)
```

![MITSU beside GELU, SiLU and ReLU: the curve and its slope](docs/figures/mitsu_curve.png)

## The core function

MITSU starts from one symmetric expression in three positive numbers,
`f(x, y, z) = (x^(y+z) + y^(x+z) + z^(x+y)) / (x + y + z)`. Each of the three "seats" is
raised to the power of the other two added together, and the total is divided by the sum of
all three, so every seat acts both as a base and as part of its neighbours' exponents, and
swapping any two seats changes nothing. To turn this into an activation function, the
incoming signal `u` is first made positive with `s = softplus(u)` and placed in seat x,
while the other two seats are held at 1. That gives `f(s, 1, 1) = (s² + 2) / (s + 2)`,
which rests at 1 when the signal is silent; subtracting that resting level leaves
`MITSU(u) = s(s − 1) / (s + 2)`. The result is a smooth curve that sits at zero for
strongly negative inputs, dips slightly below zero (to about −0.10 near `u ≈ −0.57`), and
then rises into a straight line of slope 1, approaching `u − 3`. Holding the two seats at
exactly 1 is what makes the tail a straight line: lower values flatten it and higher values
make it grow exponentially. The same expression with three different signals in the three
seats gives the three-seat competition block.

Properties, all checked in [`tests/`](tests/test_mitsu.py):

- **Smooth and non-monotonic** — minimum −0.101 at u ≈ −0.57.
- **Asymptotically linear** — MITSU(u) → u − 3 as u → +∞, → 0 as u → −∞.
- **Bounded slope** — the derivative stays inside (−0.5, 1); in practice it runs from
  about −0.04 up towards 1 and never exceeds it. SiLU's slope peaks near 1.10 and GELU's
  near 1.13.
- **No parameters** — a drop-in for `nn.GELU` / `nn.SiLU`.

`MitsuGLU` is SwiGLU with the SiLU gate swapped for MITSU; nothing else changes.

## What we claim (and what we don't)

MITSU is a **peer** of GELU/SiLU/SwiGLU, not an upgrade. Every experiment below
was pre-registered (hypotheses, decision rules, and gates frozen before data —
see the `prereg*.md` file beside each experiment), and the headline result is
**parity, everywhere we looked**:

| experiment | result |
|---|---|
| [01](experiments/01_classification) · MNIST / Fashion-MNIST MLPs (5 seeds, param-matched) | MITSU within noise of GELU; top of table on both |
| [02](experiments/02_char_gpt) · char-GPT twins, Tiny Shakespeare (identical init + batches) | val 1.7107 (MITSU) vs 1.7062 (GELU) — Δ 0.0045 nats |
| [03](experiments/03_scaling_ladder) · **nano scaling ladder**, MitsuGLU vs SwiGLU, WikiText-2, 32k→483k params | MITSU ahead at **4/4** paired rungs (each Δ < 0.004 nats); fitted exponents **α = 0.6093 vs 0.6063**; zero divergences in 8 runs |
| [04](experiments/04_dial_family) · **dial family**, 29 ways of filling the three seats, 472 runs | no variant confirmed better than the current (1, 1) setting; the three-seat block loses at a matched parameter budget |

The one open question with commercial-grade stakes: whether the bounded
derivative buys **fewer loss spikes at scale** (hypothesis S2 in
[`docs/PLAN.md`](docs/PLAN.md)). The GPU ladder in
[`docs/RING2_PLAYBOOK.md`](docs/RING2_PLAYBOOK.md) is designed to answer it.
Either answer will be published here.

## Use it

NumPy reference — `mitsu.py` is the single source of truth for the math:

```python
import numpy as np
from mitsu import mitsu, mitsu_d

u = np.linspace(-4, 4, 9)
y  = mitsu(u)      # the activation
dy = mitsu_d(u)    # its exact derivative
```

PyTorch — copy `torch/mitsu_torch.py` or `torch/mitsu_glu.py` into your project:

```python
from mitsu_torch import MITSU          # drop-in for nn.GELU / nn.SiLU
from mitsu_glu import MitsuGLU         # SwiGLU with the gate transplanted

act = MITSU()                                              # nanoGPT: self.gelu = MITSU()
ffn = MitsuGLU(dim=512, hidden_dim=1365, gate_mult=3.04)   # Llama-style feed-forward
```

`gate_mult` restores the gate's signal size at initialisation: MITSU is about 3.04× quieter
than SiLU on a standard-normal input. Set it to 1.0 to study the raw formula.

**The interactive lab** — [`lab/mitsu-activation-lab.html`](lab/mitsu-activation-lab.html)
is a single HTML file with no build step: forge the activation with live dials, train
perceptrons in-browser, stack block towers. GitHub shows its source rather than running it,
so clone the repository and open the file in a browser.

## Repo map

```
mitsu.py                      reference implementation (numpy, 20 lines)
torch/mitsu_torch.py          nn.Module drop-in for nn.GELU (patch nanoGPT in 1 line)
torch/mitsu_glu.py            MitsuGLU — SwiGLU with the gate transplanted
tests/                        unit tests for the math and the PyTorch drop-ins
lab/                          the interactive lab (single HTML file, no build)
experiments/01_classification MLPs & tri-blocks on digits/MNIST/Fashion + preregs 1–3
experiments/02_char_gpt       hand-backprop numpy transformer + trained nano weights
experiments/03_scaling_ladder MitsuGLU vs SwiGLU ladder, fits, preregs 5–6
experiments/04_dial_family    every other way of filling the three seats + prereg 7
docs/PLAN.md                  the from-scratch MITSU-8B battle plan
docs/RING2_PLAYBOOK.md        the $400 GPU scaling ladder, step by step
```

Experiments 01–03 train on CPU in pure numpy with hand-derived, finite-difference-
checked backprop (worst rel. err ~1e-6 for the transformer). Experiment 04 uses
PyTorch autograd behind its own sanity gate. The char-GPT checkpoints in
`experiments/02_char_gpt/` are real trained weights (2.2 MB each):

```
cd experiments/02_char_gpt && ./get_data.sh
python3 mitsu_gpt.py sample mitsu "ROMEO:"
```

It will produce Shakespeare-shaped dialogue and, occasionally, invent
characters. We are fond of `FillINGS MOVERD`.

## Install and test

```
pip install -r requirements.txt     # numpy alone is enough for mitsu.py
pytest -q                           # 8 tests: the math, the bounds, the PyTorch drop-ins
```

## Reproduce the ladder

```
cd experiments/03_scaling_ladder && ./get_data.sh
python3 ladder.py gradcheck   # gate: both families < 1e-4
python3 ladder.py setup       # freezes configs + gate_mult
python3 ladder.py run 600     # resumable; repeat until QUEUE COMPLETE
python3 ladder.py fit         # power laws + the verdict
```

Each experiment folder has its own README with the commands for that experiment.

## Method

Every experiment follows the same protocol: **pre-register** (predictions +
frozen decision rules) → **gradcheck gate** → run → **report against the
pre-registration**, misses included. The prereg files are unedited from before
each run (one exception, recorded in `experiments/04_dial_family/FROZEN.txt`: the
authorship line of prereg7); several contain predictions that turned out wrong, on purpose left
intact. Negative and parity results publish here just like wins.

## License & citation

MIT. If you use MITSU, cite via `CITATION.cff`.
Evan David Little · EL Innovations Inc. · Peterborough, Ontario · 2026.
三つ巴 — three forces, chasing each other's exponents, forever.
