# Pre-registration 5 — the MITSU scaling ladder (THE GATE for the 8B spend)
To be executed at P1, BEFORE any 8B compute is purchased. Skeleton written
2026-09-01; freeze final numeric thresholds before the first ladder run launches.

## Design
muP-parametrized Llama-3-skeleton models, matched pairs MitsuGLU vs SwiGLU.
Sizes N = 50M, 150M, 400M, 1B. Tokens D = 20N each (Chinchilla). FineWeb-Edu,
identical data order per pair. LR tuned once at 50M per family (8-point sweep),
muP-transferred upward. One seed per (size, family) + a replicate pair at 150M
as the replication gate. gate_mult chosen at 50M by coord check, frozen for ladder.

## Measurements
Final val loss L(N); fit L(N) = a·N^(-alpha) + c per family (3 free params, 4 points).
Stability log: any NaN, loss spike >0.3 nats, or grad-norm excursion >10x median.

## Pre-committed decision rule
PROCEED to 8B iff BOTH:
  (1) MITSU's fitted L(8B) extrapolation <= SwiGLU's + 0.010 nats, and
  (2) zero unrecovered stability incidents in MITSU runs.
HALT and publish curves otherwise. The 150M replicate must reproduce its
twin's Delta within 0.01 nats or the ladder result is void and re-run.

## Predictions (freeze before launch)
S1: parity within noise at every rung (|Delta| < 0.01), consistent with all
    prior MITSU results (digits, MNIST, Fashion, char-GPT).
S2: MITSU shows equal or fewer spike incidents (bounded derivative hypothesis).
S3 (risk): the flat origin costs MITSU 0.005-0.015 nats at 50M before
    gate_mult correction; corrected, the gap closes.
No post-hoc rescue. Either outcome is a result: MITSU scaling curves exist for
the first time, and only one outcome unlocks the $13k.
