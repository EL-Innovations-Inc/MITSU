# Pre-registration 3 — real MNIST (GitHub mirror) + Fashion-MNIST (official GitHub)
Written BEFORE any training. 2026-09-01. Sources verified: fgnt/mnist idx files
(original 1996 gzip timestamps), zalandoresearch/fashion-mnist official data.

## Design
Train subset 10,000 (stratified, seed-dependent), full official 10,000 test set.
Batch 128, Adam lr 3e-3, 12 epochs, global grad-norm clip 5, z-score on train stats.
5 seeds. Models:
  linear / ReLU-64 / GELU-64 / MITSU-64 (width-matched to tri) /
  GELU-86 + MITSU-86 (param-matched to tri, ~75.9k) / tri-64 (stem + 2 blocks, 75.9k params).
Decision-relevant: both datasets. This probes the 64-unit, 2-block regime only —
not a scaling claim.

## Predictions
M1 headroom: linear→best MLP gap >3 pts on MNIST (~92 vs ~97) and Fashion (~84 vs ~88.5).
  These are finally real instruments, unlike 8×8 digits.
M2 tri: 0.5–1.5 pts below the best width-matched MLP-64 on BOTH datasets
  (digits pattern generalizes: mechanism cost at equal width).
M3 tri vs param-matched: also below GELU-86/MITSU-86 (param-matching favors MLP even more).
M4 MITSU-64 within 1 pooled std of GELU-64 on both (the running "respectable company" theme).
M5 stability: zero divergences at 784-dim fan-in across all tri runs.

## Pre-committed calls
- "Mechanism cost CONFIRMED" if tri-64 < best MLP-64 by >1 pooled std on both datasets.
- "Digits residual was noise" if tri-64 within 1 pooled std of best MLP-64 on both.
- Split result: report as split, no averaging away.
- MITSU-activation "keeps parity" iff |MITSU-64 − GELU-64| < 1 pooled std on both.
No post-hoc rescue.
