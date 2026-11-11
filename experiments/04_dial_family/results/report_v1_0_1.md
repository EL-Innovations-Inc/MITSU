# v1.0.1 paired reanalysis

Post hoc reanalysis of the historical dial-family results. Original preregistrations, raw records and reports are unchanged. These intervals do not independently reproduce training.

Equivalence requires the paired 95% Student-t confidence interval to lie strictly inside ±0.5 percentage points (classification) or ±0.05 nats (char-GPT). These v1.0.1 analysis choices are not retrospectively preregistered. Difference screens are uncorrected for multiple comparisons; neither equivalence nor a screen establishes superiority across tasks.

## mnist

Deltas and intervals are variant minus MITSU, in percentage points.

| Variant | Valid pairs | Mean delta | Paired 95% CI | Interpretation |
|---|---:|---:|---|---|
| `log-tri@64` | 5 | +0.3200 | [-0.4853, +1.1253] | inconclusive |
| `log-inunit@64` | 5 | +0.2180 | [-0.7314, +1.1674] | inconclusive |
| `fixed(0.9,0.9)` | 5 | +0.1560 | [-0.6004, +0.9124] | inconclusive |
| `fixed(0.75,1)` | 5 | +0.1100 | [-0.6907, +0.9107] | inconclusive |
| `learn01` | 5 | +0.0880 | [-0.6511, +0.8271] | inconclusive |
| `silu` | 5 | +0.0800 | [-0.7597, +0.9197] | inconclusive |
| `mitsu` | 5 | +0.0000 | unavailable | baseline |
| `fixed(0.75,0.75)` | 5 | -0.1080 | [-0.9263, +0.7103] | inconclusive |
| `log-fixed(1.5,1.5)` | 5 | -0.1300 | [-0.9198, +0.6598] | inconclusive |
| `fixed(0.5,1)` | 5 | -0.2340 | [-1.0253, +0.5573] | inconclusive |
| `relu` | 5 | -0.2900 | [-1.2676, +0.6876] | inconclusive |
| `log-fixed(1,1.25)` | 5 | -0.3400 | [-1.2476, +0.5676] | inconclusive |
| `tri@64` | 5 | -0.3600 | [-1.2115, +0.4915] | inconclusive |
| `log-learn01` | 5 | -0.3780 | [-1.1060, +0.3500] | inconclusive |
| `log-learn02` | 5 | -0.4280 | [-1.1966, +0.3406] | inconclusive |
| `gelu` | 5 | -0.4640 | [-1.8012, +0.8732] | inconclusive |
| `log-mitsu` | 5 | -0.4860 | [-1.2163, +0.2443] | inconclusive |
| `inscalar` | 5 | -0.5540 | [-1.2461, +0.1381] | inconclusive |
| `log-fixed(0.9,0.9)` | 5 | -0.6040 | [-1.4495, +0.2415] | inconclusive |
| `log-fixed(0.75,1)` | 5 | -0.6580 | [-1.4290, +0.1130] | inconclusive |
| `log-inscalar` | 5 | -0.7280 | [-1.4786, +0.0226] | inconclusive |
| `log-fixed(0.75,0.75)` | 5 | -0.9340 | [-1.9184, +0.0504] | inconclusive |
| `log-inunit@22` | 5 | -1.0420 | [-1.8438, -0.2402] | worse (uncorrected screen) |
| `log-fixed(0.5,1)` | 5 | -1.0700 | [-1.6710, -0.4690] | worse (uncorrected screen) |
| `inunit@64` | 5 | -1.4980 | [-3.2269, +0.2309] | inconclusive |
| `log-tri@22` | 5 | -1.5020 | [-2.6225, -0.3815] | worse (uncorrected screen) |
| `inunit@22` | 5 | -1.6080 | [-2.4805, -0.7355] | worse (uncorrected screen) |
| `fixed(0.5,0.5)` | 5 | -1.9660 | [-2.8581, -1.0739] | worse (uncorrected screen) |
| `tri@22` | 5 | -2.0480 | [-3.0604, -1.0356] | worse (uncorrected screen) |
| `log-fixed(0.5,0.5)` | 5 | -2.1180 | [-2.9811, -1.2549] | worse (uncorrected screen) |
| `fixed(1,1.25)` | 0 | — | unavailable | diverged |
| `fixed(1.5,1.5)` | 0 | — | unavailable | diverged |
| `learn02` | 0 | — | unavailable | diverged |

## fashion

Deltas and intervals are variant minus MITSU, in percentage points.

| Variant | Valid pairs | Mean delta | Paired 95% CI | Interpretation |
|---|---:|---:|---|---|
| `log-fixed(0.75,0.75)` | 5 | +0.6000 | [+0.2360, +0.9640] | better (uncorrected screen) |
| `fixed(0.75,0.75)` | 5 | +0.5740 | [+0.3766, +0.7714] | better (uncorrected screen) |
| `fixed(0.9,0.9)` | 5 | +0.5720 | [+0.2953, +0.8487] | better (uncorrected screen) |
| `fixed(0.75,1)` | 5 | +0.4040 | [+0.1903, +0.6177] | better (uncorrected screen) |
| `log-fixed(0.5,1)` | 5 | +0.3780 | [-0.1875, +0.9435] | inconclusive |
| `log-fixed(0.75,1)` | 5 | +0.3700 | [+0.0610, +0.6790] | better (uncorrected screen) |
| `fixed(0.5,1)` | 5 | +0.3660 | [+0.1040, +0.6280] | better (uncorrected screen) |
| `log-mitsu` | 5 | +0.3420 | [+0.0416, +0.6424] | better (uncorrected screen) |
| `log-fixed(1.5,1.5)` | 5 | +0.3020 | [-0.1940, +0.7980] | inconclusive |
| `learn01` | 5 | +0.2460 | [+0.0545, +0.4375] | equivalent within margin |
| `log-fixed(1,1.25)` | 5 | +0.2040 | [-0.4073, +0.8153] | inconclusive |
| `log-learn01` | 5 | +0.1880 | [+0.0452, +0.3308] | equivalent within margin |
| `inscalar` | 5 | +0.1740 | [-0.1677, +0.5157] | inconclusive |
| `log-inunit@22` | 5 | +0.1700 | [-0.2089, +0.5489] | inconclusive |
| `log-fixed(0.9,0.9)` | 5 | +0.1600 | [-0.1826, +0.5026] | inconclusive |
| `silu` | 5 | +0.1500 | [-0.4420, +0.7420] | inconclusive |
| `log-inscalar` | 5 | +0.1360 | [-0.3043, +0.5763] | inconclusive |
| `log-inunit@64` | 5 | +0.1280 | [-1.0489, +1.3049] | inconclusive |
| `log-fixed(0.5,0.5)` | 5 | +0.0920 | [-0.2690, +0.4530] | equivalent within margin |
| `log-tri@64` | 5 | +0.0860 | [-0.4237, +0.5957] | inconclusive |
| `learn02` | 5 | +0.0560 | [-0.3963, +0.5083] | inconclusive |
| `fixed(0.5,0.5)` | 5 | +0.0140 | [-0.2140, +0.2420] | equivalent within margin |
| `mitsu` | 5 | +0.0000 | unavailable | baseline |
| `log-learn02` | 5 | -0.0520 | [-0.5314, +0.4274] | inconclusive |
| `inunit@64` | 5 | -0.0740 | [-0.4453, +0.2973] | equivalent within margin |
| `relu` | 5 | -0.2320 | [-0.7628, +0.2988] | inconclusive |
| `gelu` | 5 | -0.2580 | [-0.6865, +0.1705] | inconclusive |
| `tri@64` | 5 | -0.3480 | [-0.9872, +0.2912] | inconclusive |
| `inunit@22` | 5 | -0.5240 | [-0.8742, -0.1738] | worse (uncorrected screen) |
| `log-tri@22` | 5 | -0.6240 | [-1.1769, -0.0711] | worse (uncorrected screen) |
| `tri@22` | 5 | -0.9040 | [-1.5465, -0.2615] | worse (uncorrected screen) |
| `fixed(1,1.25)` | 0 | — | unavailable | diverged |
| `fixed(1.5,1.5)` | 0 | — | unavailable | diverged |

## gpt

Deltas and intervals are variant minus MITSU, in nats.

| Variant | Valid pairs | Mean delta | Paired 95% CI | Interpretation |
|---|---:|---:|---|---|
| `log-learn01` | 3 | -0.0019 | [-0.0163, +0.0125] | equivalent within margin |
| `log-inscalar` | 3 | -0.0006 | [-0.0086, +0.0074] | equivalent within margin |
| `fixed(1,1.25)` | 3 | -0.0003 | [-0.0109, +0.0103] | equivalent within margin |
| `mitsu` | 3 | +0.0000 | unavailable | baseline |
| `log-learn02` | 3 | +0.0011 | [-0.0123, +0.0144] | equivalent within margin |
| `learn02` | 3 | +0.0019 | [-0.0041, +0.0078] | equivalent within margin |
| `log-tri` | 3 | +0.0026 | [-0.0287, +0.0339] | equivalent within margin |
| `log-fixed(0.9,0.9)` | 3 | +0.0042 | [-0.0074, +0.0159] | equivalent within margin |
| `learn01` | 3 | +0.0049 | [-0.0058, +0.0156] | equivalent within margin |
| `log-mitsu` | 3 | +0.0051 | [-0.0145, +0.0247] | equivalent within margin |
| `log-fixed(1,1.25)` | 3 | +0.0052 | [-0.0139, +0.0243] | equivalent within margin |
| `inscalar` | 3 | +0.0053 | [-0.0035, +0.0141] | equivalent within margin |
| `fixed(0.9,0.9)` | 3 | +0.0056 | [+0.0014, +0.0099] | equivalent within margin |
| `fixed(0.75,1)` | 3 | +0.0064 | [+0.0010, +0.0118] | equivalent within margin |
| `gelu` | 3 | +0.0074 | [-0.0195, +0.0344] | equivalent within margin |
| `log-fixed(0.75,1)` | 3 | +0.0117 | [+0.0050, +0.0184] | equivalent within margin |
| `log-fixed(0.5,1)` | 3 | +0.0139 | [+0.0076, +0.0202] | equivalent within margin |
| `fixed(0.5,1)` | 3 | +0.0166 | [+0.0084, +0.0249] | equivalent within margin |
| `fixed(0.75,0.75)` | 3 | +0.0201 | [+0.0050, +0.0351] | equivalent within margin |
| `fixed(1.5,1.5)` | 3 | +0.0203 | [+0.0131, +0.0276] | equivalent within margin |
| `log-fixed(0.75,0.75)` | 3 | +0.0204 | [+0.0029, +0.0379] | equivalent within margin |
| `log-inunit` | 3 | +0.0209 | [-0.0151, +0.0568] | inconclusive |
| `inunit` | 3 | +0.0259 | [-0.0064, +0.0583] | inconclusive |
| `silu` | 3 | +0.0291 | [+0.0257, +0.0326] | equivalent within margin |
| `relu` | 3 | +0.0316 | [+0.0212, +0.0419] | equivalent within margin |
| `log-fixed(1.5,1.5)` | 3 | +0.0344 | [+0.0199, +0.0488] | equivalent within margin |
| `log-fixed(0.5,0.5)` | 3 | +0.0956 | [+0.0764, +0.1147] | worse (uncorrected screen) |
| `tri` | 3 | +0.1171 | [+0.0905, +0.1437] | worse (uncorrected screen) |
| `fixed(0.5,0.5)` | 3 | +0.1282 | [+0.0882, +0.1683] | worse (uncorrected screen) |

