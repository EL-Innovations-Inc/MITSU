# Pre-registration 4 — MITSU-GPT nano: the transformer twin face-off
Written BEFORE training. 2026-09-01.

## Design
Char-level GPT, pure numpy, hand-backprop (gradcheck gate passed: 6.6e-6 / 2.7e-6).
Tiny Shakespeare (1,115,394 chars, vocab 65, from karpathy/char-rnn GitHub).
Config: T=64, d=64, 2 heads, 2 layers, MLP 4×, ~110k params. Train 90% / val 10%.
Adam lr 2e-3 (100-step warmup), clip 1.0, batch 32, 2000 steps.
TWINS: identical init (same seed, shapes identical since both activations are
parameter-free) and identical batch schedule. The ONLY difference is the MLP
activation: MITSU (centered, a=b=1) vs GELU (tanh approx, GPT-2 standard).
n=1 paired run — a pilot, stated plainly. The GPU multi-seed version is the follow-up.

## Predictions
G1: both reach val loss < 2.00 by step 2000 (untrained baseline = ln 65 ≈ 4.17).
G2: |val_MITSU − val_GELU| < 0.05 nats at step 2000 — parity, the "respectable
    company" theme carries to transformers. Direction pick, weakly held, based on
    the 5-of-6 pattern: MITSU ≤ GELU (lower loss).
G3: no divergence, no NaN, for either twin (MITSU's bounded derivative (−0.5, 1)
    should be at least as tame as GELU under clip 1.0).
G4: MITSU samples are qualitatively Shakespeare-shaped: character names, colons,
    line structure. (Subjective; the demo payoff, not a decision criterion.)

## Pre-committed calls
- "Parity" iff |Δval| < 0.05. - "Meaningful gap" iff |Δval| ≥ 0.10 (flagged for
  multi-seed replication before any claim). - Between 0.05 and 0.10: suggestive only.
- Any divergence = report, no rescue.
