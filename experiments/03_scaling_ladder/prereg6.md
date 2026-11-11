# Pre-registration 6 — nano scaling ladder (CPU pilot of prereg5)
Written AFTER gradcheck/setup, BEFORE any training. 2026-09-01.

## Frozen setup (from ladder.py setup)
Char-level WikiText-2 (10.78M train tokens, vocab capped 256). B=32, T=64,
Adam lr 2e-3 (100-step warmup), clip 1.0, standard parametrization (NOT muP —
nano-pilot limitation, noted). Tokens = 15× params. Rungs (params, steps):
31,936/234 · 88,032/645 · 176,976/1297 · 483,184/3539. Single seed per cell,
but PAIRED: per rung, both families share identical init arrays and batch order.
gate_mult measured (not predicted): std(silu(z))/std(mitsu(z)) over z~N(0,1)
= 3.0374, frozen. Gradcheck gate <1e-4: passed (4.6e-5 / 2.3e-5).

## Predictions
N1 parity: mean paired Delta = val(mitsuglu) - val(swiglu) across 4 rungs
   within +/-0.010 nats.
N2 slopes: fitted power-law exponents differ by |alpha_m - alpha_s| < 0.03.
N3 stability: zero divergences in all 8 runs.
N4 sanity: val loss strictly decreases with N within each family.

## Pre-committed nano-gate
PASS iff (mean paired Delta <= +0.010) AND (alpha_m >= alpha_s - 0.03) AND N3.
The L(8e9) extrapolation from nano points will be reported strictly labeled
"for entertainment" — 4 points spanning 32k-483k params cannot license an 8B
claim. The gate governs proceeding to the GPU ladder (prereg5), nothing more.
No post-hoc rescue. Limitations: 1 seed/cell (paired), SP not muP, char-level,
CPU float32.
