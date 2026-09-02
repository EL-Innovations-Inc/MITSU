# MITSU 三つ巴

**An activation function born from a symmetric inequality, taken from napkin to
scaling law — pre-registered at every step.**

```
mother function:   f(x,y,z) = (x^(y+z) + y^(x+z) + z^(x+y)) / (x+y+z)
activation:        MITSU(u) = s(s−1)/(s+2),   s = softplus(u)
```

Smooth, non-monotonic (dip ≈ −0.101 at u ≈ −0.57), asymptotically linear, with a
derivative **bounded in (−0.5, 1)** — flatter-tailed than GELU/SiLU by construction.
`MitsuGLU` is SwiGLU with the SiLU gate swapped for MITSU; nothing else changes.

**[▶ Open the interactive lab](lab/mitsu-activation-lab.html)** — forge the
activation with live dials, train perceptrons in-browser, stack block towers.

## What we claim (and what we don't)

MITSU is a **peer** of GELU/SiLU/SwiGLU, not an upgrade. Every experiment below
was pre-registered (hypotheses, decision rules, and gates frozen before data —
see the `prereg*.md` file beside each experiment), and the headline result is
**parity, everywhere we looked**:

| experiment | result |
|---|---|
| MNIST / Fashion-MNIST MLPs (5 seeds, param-matched) | MITSU within noise of GELU; top of table on both |
| char-GPT twins, Tiny Shakespeare (identical init + batches) | val 1.7107 (MITSU) vs 1.7062 (GELU) — Δ 0.0045 nats |
| **nano scaling ladder**, MitsuGLU vs SwiGLU, WikiText-2, 32k→483k params | MITSU ahead at **4/4** paired rungs (each Δ < 0.004 nats); fitted exponents **α = 0.6093 vs 0.6063**; zero divergences in 8 runs |

The one open question with commercial-grade stakes: whether the bounded
derivative buys **fewer loss spikes at scale** (hypothesis S2 in
`docs/PLAN.md`). The GPU ladder in `docs/RING2_PLAYBOOK.md` is designed to
answer it. Either answer will be published here.

## Repo map

```
mitsu.py                      reference implementation (numpy, 20 lines)
torch/mitsu_torch.py          nn.Module drop-in for nn.GELU (patch nanoGPT in 1 line)
torch/mitsu_glu.py            MitsuGLU — SwiGLU with the gate transplanted
lab/                          the interactive lab (single HTML file, no build)
experiments/01_classification MLPs & tri-blocks on digits/MNIST/Fashion + preregs 1–3
experiments/02_char_gpt       hand-backprop numpy transformer + trained nano weights
experiments/03_scaling_ladder MitsuGLU vs SwiGLU ladder, fits, preregs 5–6
docs/PLAN.md                  the from-scratch MITSU-8B battle plan
docs/RING2_PLAYBOOK.md        the $400 GPU scaling ladder, step by step
```

Everything trains on CPU in pure numpy with hand-derived, finite-difference-
checked backprop (worst rel. err ~1e-6 for the transformer). The char-GPT
checkpoints in `experiments/02_char_gpt/` are real trained weights (2.2 MB):

```
cd experiments/02_char_gpt && ./get_data.sh
python3 mitsu_gpt.py sample mitsu "ROMEO:"
```

It will produce Shakespeare-shaped dialogue and, occasionally, invent
characters. We are fond of `FillINGS MOVERD`.

## Reproduce the ladder

```
cd experiments/03_scaling_ladder && ./get_data.sh
python3 ladder.py gradcheck   # gate: both families < 1e-4
python3 ladder.py setup       # freezes configs + gate_mult
python3 ladder.py run 600     # resumable; repeat until QUEUE COMPLETE
python3 ladder.py fit         # power laws + the verdict
```

## Method

Every experiment follows the same protocol: **pre-register** (predictions +
frozen decision rules) → **gradcheck gate** → run → **report against the
pre-registration**, misses included. The prereg files are unedited from before
each run; several contain predictions that turned out wrong, on purpose left
intact. Negative and parity results publish here just like wins.

## License & citation

MIT. If you use MITSU, cite via `CITATION.cff`.
Evan Little · EL Innovations Inc. · Whitby, Ontario · 2026.
三つ巴 — three forces, chasing each other's exponents, forever.
