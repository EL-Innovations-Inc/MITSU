# Pre-registration — three-seat block on standardized datasets
Written BEFORE any benchmark run. 2026-09-01.

## Setup
Datasets: sklearn digits (1797×64, 10c), breast_cancer (569×30, 2c), wine (178×13, 3c), iris (150×4, 3c).
Stratified 70/30 split, z-scored on train stats. 5 seeds (seed controls split + init).
Models at ~matched parameter count (MLP width solved per dataset to slightly EXCEED tri params):
  linear softmax / ReLU MLP (2 hidden) / GELU MLP / MITSU MLP (a=b=1, centered) / TRI tower (stem→2 tri blocks, W=16→head).
Adam lr 3e-3, batch 32, epochs 80 (60 for digits), global grad-norm clip 5 for ALL models equally.
Tri math = the harness-validated version: bases 2·sigmoid, ECAP=6, ETA=0.15, −3 centering.
Gate: numpy port must pass finite-difference gradcheck (<1e-5 rel err) or no experiment runs.

## Predictions
P1 (stability, high conf): tri produces zero NaN/divergence on all datasets × seeds.
P2: iris/wine/cancer are ceiling-limited — all nonlinear models within ~1 pooled std; differences not meaningful.
P3: digits: ReLU/GELU land ~97±1%; tri within 1.5 pts of the best baseline; NO significant tri win.
P4 (risk): tri slightly BELOW baselines on digits (<1 pt) — 3× preacts per unit is param overhead.
P5: linear clearly below nonlinear on digits (headroom exists); near-parity on the easy three.

## Decision rules (pre-committed)
- "Competitive": tri within 1 pooled std of best baseline on digits AND P1 holds.
- "Edge": tri beats best baseline by >1 pooled std on ≥2 datasets (not expected).
- Anything else: report as a loss, no post-hoc rescue.
