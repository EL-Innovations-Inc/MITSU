# v1.0.1 paired reanalysis

Post hoc reanalysis of the seat-weight (c family) results. Original preregistrations, raw records and reports are unchanged. These intervals do not independently reproduce training.

Equivalence requires the paired 95% Student-t confidence interval to lie strictly inside ±0.5 percentage points (classification) or ±0.05 nats (char-GPT). These v1.0.1 analysis choices are not retrospectively preregistered. Difference screens are uncorrected for multiple comparisons; neither equivalence nor a screen establishes superiority across tasks.

## mnist

Deltas and intervals are variant minus MITSU, in percentage points.

| Variant | Valid pairs | Mean delta | Paired 95% CI | Interpretation |
|---|---:|---:|---|---|
| `c=4` | 5 | +0.1180 | [-0.4907, +0.7267] | inconclusive |
| `learnc` | 5 | +0.0780 | [-0.3949, +0.5509] | inconclusive |
| `mitsu` | 5 | +0.0000 | unavailable | baseline |
| `c=8` | 5 | -0.2460 | [-1.0629, +0.5709] | inconclusive |
| `c=1` | 5 | -0.3900 | [-1.1431, +0.3631] | inconclusive |
| `c=0.25` | 5 | -0.5280 | [-1.1974, +0.1414] | inconclusive |
| `c=0.5` | 5 | -0.7100 | [-1.8526, +0.4326] | inconclusive |

## fashion

Deltas and intervals are variant minus MITSU, in percentage points.

| Variant | Valid pairs | Mean delta | Paired 95% CI | Interpretation |
|---|---:|---:|---|---|
| `c=1` | 5 | +0.2440 | [-0.1677, +0.6557] | inconclusive |
| `c=0.5` | 5 | +0.0940 | [-0.5866, +0.7746] | inconclusive |
| `c=0.25` | 5 | +0.0740 | [-0.3449, +0.4929] | equivalent within margin |
| `c=8` | 5 | +0.0600 | [-0.6485, +0.7685] | inconclusive |
| `mitsu` | 5 | +0.0000 | unavailable | baseline |
| `c=4` | 5 | -0.0340 | [-0.6827, +0.6147] | inconclusive |
| `learnc` | 5 | -0.1540 | [-0.5214, +0.2134] | inconclusive |

## gpt

Deltas and intervals are variant minus MITSU, in nats.

| Variant | Valid pairs | Mean delta | Paired 95% CI | Interpretation |
|---|---:|---:|---|---|
| `learnc` | 3 | -0.0018 | [-0.0110, +0.0073] | equivalent within margin |
| `mitsu` | 3 | +0.0000 | unavailable | baseline |
| `c=1` | 3 | +0.0014 | [-0.0140, +0.0168] | equivalent within margin |
| `c=4` | 3 | +0.0031 | [-0.0006, +0.0068] | equivalent within margin |
| `c=8` | 3 | +0.0134 | [+0.0027, +0.0240] | equivalent within margin |
| `c=0.5` | 3 | +0.0201 | [-0.0006, +0.0408] | equivalent within margin |
| `c=0.25` | 3 | +0.0305 | [+0.0249, +0.0362] | equivalent within margin |

