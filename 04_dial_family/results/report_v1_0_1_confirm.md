# v1.0.1 paired reanalysis (_confirm)

Post hoc reanalysis of the historical dial-family results. Original preregistrations, raw records and reports are unchanged. These intervals do not independently reproduce training.

Equivalence requires the paired 95% Student-t confidence interval to lie strictly inside ±0.5 percentage points (classification) or ±0.05 nats (char-GPT). These v1.0.1 analysis choices are not retrospectively preregistered. Difference screens are uncorrected for multiple comparisons; neither equivalence nor a screen establishes superiority across tasks.

## fashion

Deltas and intervals are variant minus MITSU, in percentage points.

| Variant | Valid pairs | Mean delta | Paired 95% CI | Interpretation |
|---|---:|---:|---|---|
| `fixed(0.75,0.75)` | 5 | +0.4640 | [-0.5510, +1.4790] | inconclusive |
| `log-fixed(0.75,0.75)` | 5 | +0.3320 | [-0.4085, +1.0725] | inconclusive |
| `learn01` | 5 | +0.2300 | [-0.7052, +1.1652] | inconclusive |
| `fixed(0.5,1)` | 5 | +0.2200 | [-0.7707, +1.2107] | inconclusive |
| `fixed(0.75,1)` | 5 | +0.2020 | [-0.8927, +1.2967] | inconclusive |
| `log-fixed(0.75,1)` | 5 | +0.1880 | [-0.6403, +1.0163] | inconclusive |
| `fixed(0.9,0.9)` | 5 | +0.1740 | [-0.7767, +1.1247] | inconclusive |
| `log-mitsu` | 5 | +0.1060 | [-0.6503, +0.8623] | inconclusive |
| `mitsu` | 5 | +0.0000 | unavailable | baseline |
| `log-learn01` | 5 | -0.1100 | [-1.0553, +0.8353] | inconclusive |
| `gelu` | 5 | -0.2400 | [-1.1485, +0.6685] | inconclusive |

