# Pre-registration 8 — seat weights: the c family

Written BEFORE any training run on the benchmark seeds. 2026-11-10.
Evan David Little · EL Innovations Inc.

The only code executed before this file was frozen: the sanity gate (no training) and
throwaway timing runs on seed 999, which is not a benchmark seed (MNIST: `learnc`, `c=0.25`;
char-GPT: `learnc`, `c=8` for 60 steps). Only run time, parameter count and "finite: yes/no"
were printed from those; no accuracy or loss was looked at. The gate passed on its first run.

## 1. What is being tested

Weighted mother function, with a weight on each seat:

    f(x, y, z) = (i·x^(y+z) + j·y^(x+z) + k·z^(x+y)) / (i·x + j·y + k·z)

Signal in seat x as s = softplus(u), seats y = z = 1. The function rests at 1 for a silent
input; subtracting that leaves

    MITSU_c(u) = s(s − 1) / (s + c),      c = (j + k) / i

The three weights act through the single number c (checked by the gate). c = 2 is current MITSU.

| variant | c |
|---|---|
| `mitsu` (baseline) | 2 |
| `c=0.25` `c=0.5` `c=1` `c=4` `c=8` | fixed |
| `learnc` | one learned c per layer, c = exp(θ), starts at exactly 2, kept in [0.05, 50] |

7 variants. Raw formulas, no gain multiplier, as in prereg3, prereg4 and prereg7.

Side question recorded here: "can c act as a filter?" The learned values of c are the evidence
this experiment can give on whether a network wants to move the cut-off, and which way.

## 2. Benchmarks (inherited from prereg7, frozen)

**A. Classification** — MNIST and Fashion-MNIST, stratified 10,000-image training subset per
seed, full 10,000-image test set, z-score, He init, batch 128, Adam 3e-3, 12 epochs, clip 5,
float64. Net 784 → 64 → 64 → 10 (55,050 params; `learnc` +2). Seeds 0–4.

**B. Character-GPT** — Tiny Shakespeare, T=64, d=64, 2 heads, 2 blocks, FFN 4×, Adam 2e-3 with
100-step warmup, clip 1.0, batch 32, 2000 steps, float32, validation loss on the whole
validation split. Paired twins: same batch schedule and same initial weights per seed. Seeds 0–2.

The bench scripts are those of experiment 04 with only their description text changed.
`mitsu` is computed here from the closed form s(s−1)/(s+2); in prereg7 it went through the
general log-space dial formula. The two agree to rounding error per call, so the `mitsu`
numbers here should match prereg7's closely (MNIST 95.04, Fashion 84.33, char-GPT 1.7446) but
need not match to the last digit. That comparison is the harness check.

## 3. Sanity gate (must pass before any run)
G1 c = 2 equals closed-form MITSU · G2 s(s−1)/(s+c) equals the centred weighted mother function
for nine (i, j, k) triples · G3 resting level 0 · G4 autograd vs finite differences < 1e-5 for
every variant · G5 finite outputs and gradients at extreme inputs in float32 · G6 `learnc`
starts at exactly c = 2 and equals `mitsu` at initialisation.

## 4. Predictions (frozen; wrong ones stay in)

P1 Middle values: `c=1` and `c=4` are within 0.010 nats of `mitsu` on char-GPT and within one
   pooled standard deviation on both image sets.
P2 Char-GPT is a shallow bowl around c = 2. Low side: `c=0.5` worse by 0.005–0.03 nats,
   `c=0.25` worse by 0.01–0.05. High side: `c=8` worse by 0.005–0.04.
P3 Image sets: every fixed c is within one pooled standard deviation of `mitsu` on MNIST.
   On Fashion-MNIST at least one c passes the screen as better on seeds 0–4 (as nine variants
   did in prereg7) and then fails to confirm on seeds 5–9.
P4 `learnc` is at parity everywhere (within 0.010 nats; within one pooled std).
P5 The learned c ends below 2 in every layer of every run, and stays inside [1.2, 2.0).
P6 No divergence in any run.
P7 Headline: NO value of c is confirmed better than c = 2 under rule D3.

## 5. Pre-committed decision rules (identical to prereg7)

Every variant is compared to `mitsu` on the same seeds. Δ = variant − mitsu, per seed;
t = mean(Δ) / (sd(Δ)/√n), sd with n−1.

D1 Classification parity: |mean Δ| < 1 pooled std of the two cells.
D2 Screen — "candidate better": mean Δ in the good direction with |t| ≥ 2.78 (5 seeds) or
   ≥ 4.30 (3 seeds). "Worse" is the same test in the bad direction. For char-GPT the bands
   also apply: |Δ| < 0.05 parity, ≥ 0.10 meaningful.
D3 Confirm — a candidate counts as BETTER THAN CURRENT MITSU only if it (a) passes the screen
   on at least one benchmark, (b) is not "worse" on any other, and (c) repeats the
   same-direction pass on FRESH seeds of the same benchmark (classification seeds 5–9;
   char-GPT seeds 3–5). The confirmation run uses candidates only, plus `mitsu`.
D4 Any divergence is reported as a divergence. No rescue, no retuning, no re-runs.
D5 Nothing in sections 1–5 changes after the first benchmark run starts. Anything added later
   is labelled exploratory.
