# Changelog

## 1.0.1 — 2026-11-11

- **Numerics:** prevent FP16 intermediate overflow at input 257; promote FP16/BF16
  activation arithmetic internally to FP32 and preserve output dtype/autograd.
  Stabilize NumPy evaluation/derivatives near dtype limits and correct the
  artificially clipped negative derivative tail. Add numerical regression tests.
- **Statistics:** report paired 95% confidence intervals and explicit practical
  margins. Distinguish equivalence, difference screens and inconclusive results.
  Save the post hoc reanalysis separately from historical raw results,
  summaries, preregistrations and reports.
- **Evidence:** add a multi-seed paired transformer pilot above the original
  483k-parameter maximum, with a locally frozen protocol, raw training and
  gradient histories, full held-out validation, uncertainty and exposure counts.
  Include an actual 1,000-step exploratory extension along the same seeded
  trajectories and an unrun five-seed profile; no billion-parameter stability claim
  is inferred from the short CPU pilot.
- **Stability explanation:** document how gate gain changes derivatives and why
  the scalar slope bound does not bound a complete GLU/network Jacobian.
- **Usability and performance:** add the installable `mitsu-activation` package,
  versioned API, tested CPU/Python 3.12 dependency snapshots, wheel/sdist smoke
  checks and CI. Include actual CPU timing/tensor-storage comparisons and a
  CUDA-capable benchmark harness. CPU peak allocator memory remains explicitly
  unavailable; no GPU measurements are invented.

- **Seat-weight experiment:** add experiment 06, a pre-registered test of the
  constant c in s(s-1)/(s+c) (c = 2 is current MITSU): six fixed values and a
  learnable c on the experiment-04 benchmarks. No value is confirmed better than
  c = 2. The v1.0.1 confidence-interval report is included alongside the
  pre-registered analysis.
- **Release layout:** built distributions are no longer kept in the source tree
  (`dist/` is git-ignored); attach the wheel and source archive to the release
  instead. `SHA256SUMS` now covers exactly the files that are committed, so
  `scripts/verify_checksums.py` also passes on a git checkout. CI actions are
  updated to current major versions and can be started manually.

The underlying activation formula is unchanged. The original interactive lab,
experiments, character-model checkpoints and license remain in the release.
See VALIDATION.md for the checks actually executed for this ZIP.
