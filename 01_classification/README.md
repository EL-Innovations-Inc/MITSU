> Historical experiment report. “Parity” below is the original decision-rule
> terminology, not a new equivalence-test conclusion. See
> [v1.0.1 statistical guidance](../../docs/STATISTICS.md) and
> [the new multi-seed pilot](../05_multiseed_scaling/README.md).

# 01 · Classification

MITSU against ReLU and GELU in small MLPs, and the three-seat ("tri") block against all of
them. Pure numpy with hand-derived backprop behind a finite-difference gradcheck gate.
Three pre-registered runs, 5 seeds each.

| run | script | pre-registration | question |
|---|---|---|---|
| 1 | `bench.py` | [`preregistration.md`](preregistration.md) | MITSU and the tri block on scikit-learn's bundled datasets, parameter-matched |
| 2 | `width.py` | [`prereg2.md`](prereg2.md) | is the tri block's digits deficit a width tax or the mechanism? |
| 3 | `big.py` | [`prereg3.md`](prereg3.md) | the same comparison on real MNIST and Fashion-MNIST |

## Results

Test accuracy %, mean ± std over 5 seeds.

**Run 3 — MNIST and Fashion-MNIST** (`results_mnist.json`, `results_fashion.json`)

| model | params | MNIST | Fashion-MNIST |
|---|---|---|---|
| linear | 7,850 | 90.65 ± 0.21 | 81.52 ± 0.57 |
| ReLU-64 | 55,050 | 94.75 ± 0.62 | 84.06 ± 0.35 |
| GELU-64 | 55,050 | 94.70 ± 0.54 | 84.10 ± 0.60 |
| **MITSU-64** | 55,050 | **95.00 ± 0.45** | 84.44 ± 0.45 |
| GELU-86 | 75,862 | 94.27 ± 0.81 | 84.00 ± 0.22 |
| **MITSU-86** | 75,862 | 94.67 ± 0.59 | **84.80 ± 0.23** |
| tri-64 | 75,850 | 94.54 ± 0.26 | 84.36 ± 0.61 |

MITSU-64 is within one pooled standard deviation of GELU-64 on both datasets, which is the
pre-registered parity call. The tri block is below the best 64-unit MLP by more than one
pooled standard deviation on MNIST and within it on Fashion-MNIST: a split result.

**Runs 1 and 2 — digits** (`results.json`, `results_width.json`; the other three bundled
datasets are ceiling-limited and are in the JSON files)

| model | params | digits |
|---|---|---|
| linear | 650 | 96.67 ± 0.74 |
| ReLU (width 28) | 2,922 | 96.78 ± 0.77 |
| GELU (width 28) | 2,922 | 97.00 ± 0.74 |
| **MITSU (width 28)** | 2,922 | **97.07 ± 0.74** |
| tri (width 16) | 2,842 | 95.26 ± 0.92 |
| ReLU-16 | 1,482 | 96.63 ± 0.62 |
| GELU-16 | 1,482 | 96.37 ± 0.76 |
| MITSU-16 | 1,482 | 96.04 ± 0.76 |
| tri-wide (width 28) | 6,982 | 96.37 ± 0.75 |

No run diverged.

## Reproduce

Needs numpy, scipy and scikit-learn.

```
python3 bench.py              # run 1, writes results.json
python3 width.py              # run 2, writes results_width.json

./get_data.sh                 # MNIST + Fashion-MNIST idx files into data/
python3 parse_idx.py          # idx files -> data/mnist.npz, data/fashion.npz
python3 big.py mnist          # run 3, writes results_mnist.json
python3 big.py fashion        #        writes results_fashion.json
```
