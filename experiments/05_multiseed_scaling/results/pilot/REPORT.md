# v1.0.1 multi-seed larger-model results

Measured CPU FP32 runs; final full-window validation loss on Tiny Shakespeare. Delta = MitsuGLU minus SwiGLU; positive favors SwiGLU. Intervals are paired Student-t 95% CIs across seeds when all configured pairs are complete and finite. Primary family means use exactly the same valid pairs.

| Parameters | Complete / expected pairs | MitsuGLU loss | SwiGLU loss | Mean delta | 95% CI |
|---:|---:|---:|---:|---:|---|
| 997,632 | 3 / 3 | 2.559333 | 2.546729 | +0.012604 | [+0.003227, +0.021980] |
| 1,756,160 | 3 / 3 | 2.578167 | 2.556840 | +0.021327 | [-0.027692, +0.070346] |

Each configured run: 100 updates, batch 8, context 64; 3 paired seeds per size and 51,200 training tokens per completed model. Models exceed the original 483,184-parameter maximum. This remains a budget-limited training diagnostic, not a compute-optimal scaling-law experiment.

At 997,632 parameters, the observed paired mean favors SwiGLU. Inference status: complete paired Student-t interval; descriptive exploratory evidence. This is not a general superiority claim.
At 1,756,160 parameters, the observed paired mean favors SwiGLU. Inference status: complete paired Student-t interval; descriptive exploratory evidence. This is not a general superiority claim.

## Stability, exposure, and runtime diagnostics (all recorded runs)

| Parameters | Family | Divergences | Loss spikes | Updates observed | Train tokens observed | Mean train seconds | Mean train tokens/s |
|---:|---|---:|---:|---:|---:|---:|---:|
| 997,632 | mitsuglu | 0 | 0 | 300 | 153,600 | 3.43 | 14,910 |
| 997,632 | swiglu | 0 | 0 | 300 | 153,600 | 3.21 | 16,153 |
| 1,756,160 | mitsuglu | 0 | 0 | 300 | 153,600 | 6.75 | 7,637 |
| 1,756,160 | swiglu | 0 | 0 | 300 | 153,600 | 5.86 | 8,927 |

Loss spike definition: after at least 20 completed steps, loss > prior-20-step median + 0.5 nats. Global gradient clipping at norm 1.0 is active in both families. Zero observed spikes or divergences under this recipe cannot establish a stability advantage. Training histories and gradient norms remain in the JSONL records.

Runtime is observed on a shared CPU while up to two workers run concurrently; do not use these numbers to rank activation kernels. See the dedicated [activation microbenchmark](../../../../benchmarks/README.md).

## Limits and provenance

- 3 seeds give limited uncertainty estimates; intervals cover seed variation conditional on one dataset split and one fixed recipe.
- No family-specific hyperparameter tuning. No multiple-dataset replication. No word-level tokenizer or long-context evaluation.
- The gain multiplier is 3.0374 for MITSU; the raw derivative bound does not bound the scaled GLU block.
- The 100-update pilot and 1,000-update extension have separate manifests and raw records. The extension was designed after inspecting the pilot and is exploratory, not an independent replication. `full.json` is an unrun suggested follow-up.
- Initial parameter hashes match within each reported pair. Dataset SHA-256, code/config hashes, exact versions and machine context are in `manifest.json`. The freeze is local, not independently timestamped preregistration.
- No scaling exponent is fitted: varying width at a fixed update budget is not evidence for compute-optimal scaling.
- Incomplete, divergent, or mismatched pairs are listed below and cause inference to be withheld for the whole size. Valid-pair means, if shown, are descriptive only. Zero seed variance also withholds an interval.

Excluded/incomplete pairs: none.
Unexpected records: none.
