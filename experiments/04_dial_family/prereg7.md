# Pre-registration 7 — the dial family: every way of filling the three seats

Written BEFORE any training run on the benchmark seeds. 2026-11-10.
Author of this prereg and of the predictions: Evan David Little.
The only code executed before this file was frozen:
the sanity gate (no training) and throwaway timing runs on seed 999, which is not a
benchmark seed (char-GPT: gelu, mitsu, log-inunit, tri, log-fixed(0.5,1) for 100 steps
each; MNIST: gelu, log-inunit@64, tri@22, inscalar). Only run time, parameter count and
"finite: yes/no" were printed from those; no accuracy or loss was looked at.

Gate history, for the record: the first gate run failed twice. (1) I had assumed the
repo's tri block rests at exactly 0 for u = 0; it rests at 3·MITSU(0)/(3+η) = −0.0752,
so my expectation was wrong and the check was corrected. (2) `inunit` produced NaN in
float32 when a dial underflowed to exactly 0; learned and input-driven dials now have a
floor of 0.05. The gate then passed in full.

## 1. What is being tested

Mother function f(x,y,z) = (x^(y+z) + y^(x+z) + z^(x+y)) / (x+y+z).
Seat x carries the signal, s = softplus(u). Each dial form is centred by subtracting
its resting level, so a = b = 1 is exactly current MITSU.

| family | y, z | variants |
|---|---|---|
| current MITSU | 1, 1 | `mitsu` |
| fixed dials | constants (a, b) | (0.9,0.9) (0.75,1) (0.5,1) (0.75,0.75) (0.5,0.5), plus two above-1 stress settings (1,1.25) (1.5,1.5) |
| learnable dials | one learned pair per layer | `learn01`: range (0.05, 1), starts 0.955 · `learn02`: range (0.05, 1.95), starts at exactly 1 |
| input-driven dials | computed from the layer's own input | `inscalar`: one pair per example/token · `inunit`: one pair per unit (3 pre-activations per unit) |
| three-seat competition | three different signals | `tri`: the repo's tri block, formula unchanged |
| log version | log f of every row above | `log-…` for each (13 variants) |
| references | — | `relu`, `gelu` (tanh approx), `silu` |

29 variants. Raw formulas throughout — no gain multiplier on any variant, matching how
`mitsu` was run in prereg3 and prereg4.

Definition choices made here and frozen:
- log-tri is the log of the pure competition term, log[Σ s_c^e_c / (S+η)], centred at u = 0.
  The repo's numerator (Σ MITSU + Σ powers − 3) can be negative, so it has no logarithm.
- Input-driven dials live in (0.05, 1). Their projections start at 0.1× the normal init
  with bias +3 (dials ≈ 0.95), so those variants start close to current MITSU.
- softplus is floored at 1e-6 so log(s) and fractional powers stay finite.

## 2. Benchmarks (both inherited, frozen)

**A. Classification** — prereg3 protocol. MNIST and Fashion-MNIST, stratified 10,000-image
training subset per seed, full 10,000-image test set, z-score, He init, batch 128, Adam
3e-3, 12 epochs, clip 5, float64. Net: 784 → H → H → 10. Seeds 0–4.
Light variants H = 64 (55,050 params; identical initial weights across variants per seed).
Heavy variants (`inunit`, `tri`, and their logs): H = 22 (53,558 params, param-matched,
slightly fewer than baseline) AND H = 64 (width-matched, ~3× params).

**B. Character-GPT** — prereg4 config. Tiny Shakespeare, T=64, d=64, 2 heads, 2 blocks,
FFN 4×, Adam 2e-3 with 100-step warmup, clip 1.0, batch 32, 2000 steps, float32.
Paired twins: same batch schedule and same non-FFN init per seed. Heavy variants use
FFN hidden 127 (slightly fewer params than 4×). Seeds 0–2, run seed by seed so an
interrupted queue still holds complete paired seeds. Measured cost is about 2–4 minutes
per run on this 2-core machine, so the full grid (87 runs) is about 2 hours.
Change from prereg4: val loss is measured on the whole validation split, not 8 batches.

Implementation is PyTorch autograd, not the repo's hand-derived numpy. The harness is
new, so absolute numbers need not match the repo's to the last digit; `mitsu` vs `gelu`
inside this harness is the internal check on that.

## 3. Sanity gate (replaces gradcheck; must pass before any run)
G1 dial(1,1) = closed-form MITSU · G2 log forms = log of the plain formula · G3 torch tri =
repo tri_fwd formula · G4 resting level 0 · G5 autograd vs finite differences < 1e-5 for
every variant · G6 finite outputs and gradients at extreme inputs for every variant whose
dials cannot exceed 1.

## 4. Predictions (mine; wrong ones stay in)

P1 Fixed dials close to 1 — (0.9,0.9), (0.75,1) — are at parity with `mitsu` on all three
   benchmarks.
P2 Above-1 dials do NOT diverge at this scale: (1,1.25) is at parity; (1.5,1.5) trains but
   is worse than `mitsu` on char-GPT by more than 0.02 nats. (So my earlier word
   "unusable" is predicted to be too strong for small, clipped, normalised networks.)
P3 Saturating dials (0.5,0.5): within 0.5 pt of `mitsu` on both image sets; worse on
   char-GPT by 0.02–0.10 nats.
P4 Learnable dials: parity everywhere. Learned values stay within 0.2 of where they started.
P5 `inscalar`: parity everywhere.
P6 Heavy variants at matched parameters (`inunit@22`, `tri@22`, and FFN-127 in the GPT)
   are BELOW `mitsu`: 0.3–2 pts on the image sets, 0.01–0.06 nats on char-GPT.
   At width 64 with 3× parameters they are at parity or slightly above.
P7 Log versions: on average worse than their non-log counterparts; on char-GPT the mean
   gap (log minus non-log, over the light variants) is > 0.02 nats.
P8 No NaN divergence in any run, with (1.5,1.5) and `learn02` the only ones I would not be
   surprised by.
P9 Headline: NO variant is confirmed better than current MITSU under rule D3.

## 5. Pre-committed decision rules

Every variant is compared to `mitsu` on the same seeds. Δ = variant − mitsu, per seed;
t = mean(Δ) / (sd(Δ)/√n), sd with n−1.

D1 Classification parity (repo convention): |mean Δ| < 1 pooled std of the two cells.
D2 Screen — "candidate better": on a benchmark, mean Δ in the good direction with
   |t| ≥ 2.78 (5 seeds) or ≥ 4.30 (3 seeds), i.e. two-sided 5% for the paired test.
   "Worse" is the same test in the bad direction.
   For char-GPT the repo's bands also apply: |Δ| < 0.05 parity, ≥ 0.10 meaningful.
D3 Confirm — a candidate counts as BETTER THAN CURRENT MITSU only if it (a) passes the
   screen on at least one benchmark, (b) is not "worse" on any other, and (c) repeats the
   same-direction pass on FRESH seeds of the same benchmark (classification seeds 5–9;
   char-GPT seeds 3–5). The confirmation run uses candidates only, plus `mitsu`.
   With ~28 comparisons per benchmark, about 1–2 screen passes per benchmark are expected
   from chance alone; that is what step (c) is for.
D4 Any divergence is reported as a divergence. No rescue, no retuning, no re-runs.
D5 Nothing in sections 1–5 changes after the first benchmark run starts. Anything added
   later is labelled exploratory.
