# Results 7 — the dial family, reported against prereg7

Evan David Little, 2026-11-10.
Pre-registration: `prereg7.md`, frozen 19:27 UTC before any benchmark run (hashes in `FROZEN.txt`).
472 training runs: 330 classification screen, 55 classification confirmation, 87 character-GPT.

## Headline

**No variant is confirmed better than current MITSU (dials 1, 1).** Prediction P9 held.

- Nine variants passed the first-round screen on Fashion-MNIST (+0.2 to +0.6 points). None repeated
  the pass on five fresh seeds, and six of the nine were measurably worse on the character-GPT.
- On the character-GPT, **(1, 1) sits at the bottom of the bowl**: every fixed setting below 1 was
  worse, and the further below, the worse. One setting above 1 tied; the other was worse.
- Dials above 1 **blew up in the plain classifier** (10 of 10 runs each) but trained normally in the
  layer-normalised transformer. The log version made them safe everywhere.
- The **three-seat block and per-unit input-driven dials lost at a matched parameter budget** on
  all three benchmarks.
- **Learnable dials and one-pair-per-example input-driven dials cost almost nothing and gained nothing.**

## What was tested

Mother function f(x,y,z) = (x^(y+z) + y^(x+z) + z^(x+y)) / (x+y+z), seat x = softplus(signal).
29 variants: current MITSU, 7 fixed dial pairs, 2 learnable, 2 input-driven, the repo's three-seat
block, the log of each of those 13, and ReLU / GELU / SiLU for reference. Raw formulas, no gain.

Benchmarks, both inherited from the repo:
- **Classification** (prereg3 protocol): MNIST and Fashion-MNIST, 784 → H → H → 10, 10,000-image
  training subset, 12 epochs, seeds 0–4. Test accuracy, higher is better.
- **Character-GPT** (prereg4 config): Tiny Shakespeare, 2 blocks, width 64, 2000 steps, seeds 0–2.
  Validation loss in nats, lower is better.

Harness check: `mitsu` scored 95.04% on MNIST here against 95.00% in the repo; `relu` 94.75 against 94.75.

## Results by family

Δ is variant minus `mitsu`, paired on the same seeds. ▲ = passed the pre-registered screen as better,
▼ = passed it as worse (|t| ≥ 2.78 with 5 seeds, ≥ 4.30 with 3). No mark = parity under that rule.
"Matched params" means slightly fewer parameters than the baseline; "width 64" means about 3× as many.

**References**

| variant | MNIST acc % (Δ) | Fashion acc % (Δ) | char-GPT loss (Δ) |
|---|---|---|---|
| `mitsu` (baseline, dials 1, 1) | 95.04 | 84.33 | 1.7446 |
| `gelu` | 94.57 (-0.46) | 84.07 (-0.26) | 1.7520 (+0.0074) |
| `silu` | 95.12 (+0.08) | 84.48 (+0.15) | 1.7737 (+0.0291) ▼ |
| `relu` | 94.75 (-0.29) | 84.10 (-0.23) | 1.7761 (+0.0316) ▼ |

**Fixed dials**

| variant | MNIST acc % (Δ) | Fashion acc % (Δ) | char-GPT loss (Δ) |
|---|---|---|---|
| `fixed(0.9,0.9)` | 95.19 (+0.16) | 84.90 (+0.57) ▲ | 1.7502 (+0.0056) ▼ |
| `fixed(0.75,1)` | 95.15 (+0.11) | 84.73 (+0.40) ▲ | 1.7510 (+0.0064) ▼ |
| `fixed(0.5,1)` | 94.80 (-0.23) | 84.70 (+0.37) ▲ | 1.7612 (+0.0166) ▼ |
| `fixed(0.75,0.75)` | 94.93 (-0.11) | 84.90 (+0.57) ▲ | 1.7646 (+0.0201) ▼ |
| `fixed(0.5,0.5)` | 93.07 (-1.97) ▼ | 84.34 (+0.01) | 1.8728 (+0.1282) ▼ |
| `fixed(1,1.25)` | diverged 5/5 | diverged 5/5 | 1.7443 (-0.0003) |
| `fixed(1.5,1.5)` | diverged 5/5 | diverged 5/5 | 1.7649 (+0.0203) ▼ |

**Learnable dials**

| variant | MNIST acc % (Δ) | Fashion acc % (Δ) | char-GPT loss (Δ) |
|---|---|---|---|
| `learn01` | 95.13 (+0.09) | 84.58 (+0.25) ▲ | 1.7494 (+0.0049) |
| `learn02` | diverged 5/5 | 84.39 (+0.06) | 1.7464 (+0.0019) |

**Input-driven dials**

| variant | MNIST acc % (Δ) | Fashion acc % (Δ) | char-GPT loss (Δ) |
|---|---|---|---|
| `inscalar` | 94.48 (-0.55) | 84.50 (+0.17) | 1.7498 (+0.0053) |
| `inunit` (matched params) | 93.43 (-1.61) ▼ | 83.81 (-0.52) ▼ | 1.7705 (+0.0259) |
| `inunit` (width 64, 3× params) | 93.54 (-1.50) | 84.26 (-0.07) | — |

**Three-seat competition**

| variant | MNIST acc % (Δ) | Fashion acc % (Δ) | char-GPT loss (Δ) |
|---|---|---|---|
| `tri` (matched params) | 92.99 (-2.05) ▼ | 83.43 (-0.90) ▼ | 1.8617 (+0.1171) ▼ |
| `tri` (width 64, 3× params) | 94.68 (-0.36) | 83.98 (-0.35) | — |

**Log versions**

| variant | MNIST acc % (Δ) | Fashion acc % (Δ) | char-GPT loss (Δ) |
|---|---|---|---|
| `log-mitsu` | 94.55 (-0.49) | 84.67 (+0.34) ▲ | 1.7497 (+0.0051) |
| `log-fixed(0.9,0.9)` | 94.43 (-0.60) | 84.49 (+0.16) | 1.7488 (+0.0042) |
| `log-fixed(0.75,1)` | 94.38 (-0.66) | 84.70 (+0.37) ▲ | 1.7562 (+0.0117) ▼ |
| `log-fixed(0.5,1)` | 93.97 (-1.07) ▼ | 84.71 (+0.38) | 1.7585 (+0.0139) ▼ |
| `log-fixed(0.75,0.75)` | 94.10 (-0.93) | 84.93 (+0.60) ▲ | 1.7650 (+0.0204) ▼ |
| `log-fixed(0.5,0.5)` | 92.92 (-2.12) ▼ | 84.42 (+0.09) | 1.8401 (+0.0956) ▼ |
| `log-fixed(1,1.25)` | 94.70 (-0.34) | 84.53 (+0.20) | 1.7497 (+0.0052) |
| `log-fixed(1.5,1.5)` | 94.91 (-0.13) | 84.63 (+0.30) | 1.7789 (+0.0344) ▼ |
| `log-learn01` | 94.66 (-0.38) | 84.52 (+0.19) ▲ | 1.7426 (-0.0019) |
| `log-learn02` | 94.61 (-0.43) | 84.28 (-0.05) | 1.7456 (+0.0011) |
| `log-inscalar` | 94.31 (-0.73) | 84.47 (+0.14) | 1.7440 (-0.0006) |
| `log-inunit` (matched params) | 94.00 (-1.04) ▼ | 84.50 (+0.17) | 1.7654 (+0.0209) |
| `log-inunit` (width 64) | 95.26 (+0.22) | 84.46 (+0.13) | — |
| `log-tri` (matched params) | 93.54 (-1.50) ▼ | 83.71 (-0.62) ▼ | 1.7471 (+0.0026) |
| `log-tri` (width 64) | 95.36 (+0.32) | 84.42 (+0.09) | — |

Reading the char-GPT column: the repo's bands also apply (|Δ| < 0.05 parity, ≥ 0.10 meaningful).
Most ▼ marks there are statistically detectable but inside the parity band. Only `fixed(0.5,0.5)`
(+0.128) and `tri` (+0.117) are meaningfully worse; `log-fixed(0.5,0.5)` (+0.096) is just under.

Full sorted tables with standard deviations, standard errors and t values: `results/tables_main.txt`.

## Confirmation run (rule D3)

Nine variants passed the Fashion-MNIST screen. As pre-registered, they were re-run with `mitsu` on
fresh seeds 5–9 (`gelu` was added as a reference). Required to confirm: |t| ≥ 2.78, same direction.

| variant | screen Δ (t), seeds 0–4 | confirmation Δ (t), seeds 5–9 | char-GPT | confirmed? |
|---|---|---|---|---|
| `fixed(0.75,0.75)` | +0.57 (8.07) | +0.46 (1.27) | worse ▼ | no |
| `log-fixed(0.75,0.75)` | +0.60 (4.58) | +0.33 (1.24) | worse ▼ | no |
| `fixed(0.9,0.9)` | +0.57 (5.74) | +0.17 (0.51) | worse ▼ | no |
| `fixed(0.75,1)` | +0.40 (5.25) | +0.20 (0.51) | worse ▼ | no |
| `log-fixed(0.75,1)` | +0.37 (3.32) | +0.19 (0.63) | worse ▼ | no |
| `fixed(0.5,1)` | +0.37 (3.88) | +0.22 (0.62) | worse ▼ | no |
| `log-mitsu` | +0.34 (3.16) | +0.11 (0.39) | parity | no |
| `learn01` | +0.25 (3.57) | +0.23 (0.68) | parity | no |
| `log-learn01` | +0.19 (3.66) | −0.11 (−0.32) | parity | no |

Every candidate fails condition (c), the fresh-seed repeat. The first six also fail condition (b),
being "worse" on another benchmark. No variant passed the screen as better on MNIST or on the
character-GPT, so no other confirmation run was triggered.

## Prediction scorecard

| # | prediction | outcome |
|---|---|---|
| P1 | Dials near 1 — (0.9,0.9), (0.75,1) — at parity on all three | **Half right.** Parity on MNIST. Ahead on the Fashion screen (not confirmed). Behind on char-GPT by 0.006 nats: tiny, inside the parity band, but past the t rule. |
| P2 | Above-1 dials do not diverge; (1,1.25) parity; (1.5,1.5) worse on char-GPT by > 0.02 | **Wrong on classification, right on char-GPT.** Both diverged in 5/5 runs on both image sets. On char-GPT (1,1.25) tied (−0.0003) and (1.5,1.5) was +0.0203 worse. |
| P3 | (0.5,0.5) within 0.5 pt on images; 0.02–0.10 worse on char-GPT | **Wrong on two of three.** MNIST −1.97 pts; Fashion +0.01 (right); char-GPT +0.128, worse than predicted. |
| P4 | Learnable dials at parity; values stay within 0.2 of start | **Mostly right.** Parity wherever they trained. Misses: `learn02` diverged 5/5 on MNIST, and `log-learn02` climbed to 1.20–1.29 on the image sets. |
| P5 | `inscalar` at parity everywhere | **Right.** Closest calls: MNIST −0.55 (t −2.22), char-GPT +0.005 (t 2.59 against 4.30). |
| P6 | Heavy variants below `mitsu` at matched params (0.3–2 pts, 0.01–0.06 nats); parity at width 64 | **Right in direction, off in size.** `inunit` −1.61 / −0.52 pts, +0.026 nats. `tri` −2.05 / −0.90 pts, and +0.117 nats — twice my range. `log-tri` tied on char-GPT (+0.003), better than predicted. Width 64: parity in all eight cells, though `inunit@64` on MNIST was −1.50 with noisy seeds. |
| P7 | Log versions worse on average; char-GPT gap > 0.02 nats | **Wrong.** Mean char-GPT gap was −0.002 (log marginally better). On MNIST the log cost the one-signal variants 0.15–0.84 pts but helped the heavy ones by 0.55–1.72. Fashion: no consistent difference (mean +0.11 pts). |
| P8 | No NaN divergence, except possibly (1.5,1.5) and `learn02` | **Wrong.** 25 of 330 classification screen runs diverged, including (1,1.25), which I did not expect. 0 of 55 confirmation runs and 0 of 87 char-GPT runs diverged. |
| P9 | No variant confirmed better than current MITSU | **Right.** |

## What each family showed

**Fixed dials.** On the character-GPT the cost grows with distance below 1:
(0.9,0.9) +0.006 → (0.75,1) +0.006 → (0.5,1) +0.017 → (0.75,0.75) +0.020 → (0.5,0.5) +0.128.
Above 1, (1,1.25) tied and (1.5,1.5) cost +0.020. On Fashion-MNIST the lower settings were ahead by
0.2–0.6 points in the screen, but that did not confirm. Dials above 1 are unsafe without
normalisation in front of them.

**Learnable dials.** Four extra parameters per network, no measurable gain or loss. Because the two
dials start equal and the formula is symmetric, they stay equal: this is really one learned number
per layer. Where they settled: 0.97 on the image sets when capped at 1; 0.98–1.05 when allowed above
1 (above 1 on MNIST, where it then overflowed at test time); 1.20–1.29 for the log version; 0.87–0.95 in the character-GPT.

**Input-driven dials.** One pair per example (`inscalar`): parity on all three, dials stayed at
0.91–1.00. One pair per unit (`inunit`): three times the weights per unit, and it lost at a matched
budget on both image sets (−1.6, −0.5 pts); +0.026 nats on char-GPT, not significant with 3 seeds.

**Three-seat competition.** At a matched budget the repo's block lost on all three benchmarks and was
the second-worst variant on char-GPT (+0.117). At 3× the parameters it reached parity on the image
sets. This agrees with the repo's own finding that the tri block does not pay for itself.

**Log versions.** Never better than `mitsu`, but not the penalty I predicted. Their value is safety:
the log tamed both above-1 settings on the image sets (parity instead of divergence) and brought the
three-seat block from +0.117 to +0.003 on char-GPT.

**References.** In this harness `mitsu` was ahead of SiLU and ReLU on char-GPT by about 0.03 nats
(detectable, inside the parity band) and ahead of GELU by 0.007 (not significant).

## Exploratory — not pre-registered, not evidence of a win

- **Pooling Fashion seeds 0–9** (`exploratory.py`): `fixed(0.75,0.75)` +0.52 pts (t 2.94, ahead on
  9 of 10 seeds); `log-fixed(0.75,0.75)` +0.47 (t 3.17). The other seven are below the 10-seed
  threshold of 2.26. This pooling reuses the seeds that selected the candidates, so it is biased
  upward. Both variants were worse on the character-GPT.
- **Why the image runs diverged** (one-off check on seed 0, script not kept). MNIST has 115 pixels
  that are constant in the training subset; the inherited z-score (divide by sd + 1e-8) turns 15
  test images into inputs as large as 2e10. `fixed(1,1.25)` blew up during training on MNIST
  (epoch 4) and at test time on one Fashion image. `fixed(1.5,1.5)` blew up during training in
  epoch 1 on both. `learn02` trained normally; its dials drifted to 1.03–1.05 and it overflowed
  only on those 15 MNIST test images (94.74% on the other 9,985). Under rule D4 all of these count
  as divergences; nothing was re-run or rescued.
- **Direction of learned dials.** They moved mostly up on the image sets and down in the character-GPT,
  the opposite of where the fixed-dial results pointed on each. The direction a dial learns is not
  a guide to which fixed setting is better.

## Deviations and caveats

- After the freeze I changed one line in each bench script: the separator for the optional variant
  list went from "," to ";" because variant names contain commas. The first confirmation command
  therefore ran only 5 of the 11 cells; the other 6 were run immediately after into the same file.
  No protocol, model or number is affected. This is why the two bench-script hashes differ from
  `FROZEN.txt`; `variants.py` and `gate.py` are unchanged.
- The authorship line at the top of `prereg7.md` was edited after the runs. No prediction, rule or
  number in it changed; `FROZEN.txt` records both the original and the current hash.
- `gelu` was added to the confirmation run as a reference; it is not a candidate.
- Small scale: 55k-parameter classifiers and a 111k-parameter transformer. Nothing here says how any
  variant behaves at ladder scale.
- 5 seeds and 3 seeds. With 3 seeds a t value is fragile: a very consistent 0.006-nat gap passes the
  rule. Read the char-GPT column with the 0.05 band in mind.
- One learning rate per benchmark, inherited, not tuned per variant.
- Raw formulas with no gain multiplier, as in prereg3 and prereg4. MitsuGLU was not part of this test.
- New PyTorch autograd harness, not the repo's numpy code. Validation loss is on the whole
  validation split, not 8 sampled batches.

## Reproduce

Data is not included (42 MB); see `data/README.md`. Needs Python 3, numpy, torch.

    python3 gate.py
    python3 bench_cls.py mnist 0 1 0 5
    python3 bench_cls.py fashion 0 1 0 5
    python3 bench_gpt.py 0 1 0 3
    python3 analyze.py
    python3 bench_cls.py fashion 0 1 5 10 _confirm "mitsu;gelu;fixed(0.75,0.75);..."
    python3 analyze.py _confirm

About 25 minutes of CPU for classification and 3 hours for the character-GPT on one core.
