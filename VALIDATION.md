# v1.0.1 validation

Checks executed locally on Linux x86_64, CPython 3.12.14, NumPy 2.3.5,
SciPy 1.17.0, and PyTorch 2.15.0+cpu. Exact environment snapshots are in the
requirements lock files and the measured run manifests. The configured remote
CI matrix has not been executed from this workspace.

## Software checks

- Full test suite: **48 tests passed** after final integration.
- Dial-family sanity gate: PASS. The only observed extreme-input nonfinite
  cases were the gate's documented above-one-dial stress cases. No unexpected
  case failed the gate.
- Historical NumPy transformer gradient checks: MITSU 6.55e-6, GELU 2.71e-6
  worst relative errors, within the documented 1e-4 gate.
- Historical gated-ladder gradient checks: MitsuGLU 4.58e-5, SwiGLU 2.34e-5,
  within the documented 1e-4 gate.
- The reproduced FP16 input-257 failure is covered by regression tests. Tests
  also cover dtype-limit values, finite gradients, first/second derivatives,
  representative GLU backward passes, CPU autocast and standalone copies.
- Statistical tests cover CI calculations, equivalence versus nonsignificance,
  duplicate/missing seeds, invalid/diverged records and degenerate variance.
- Nineteen protected historical preregistration, raw-result, checkpoint,
  summary and ladder-state/fit files match the supplied ZIP byte-for-byte.
- Original NumPy-only tests in a clean environment: 6 passed, 2 PyTorch tests
  skipped. Full-suite dependencies are needed for statistical and integration
  tests; a minimal runtime install is deliberately NumPy-only.
- Wheel and source distribution build and installed-wheel smoke checks: PASS
  after final integration. The smoke check executes outside the checkout
  and verifies import provenance, values, autograd and GLU output shape.

## Measured research and performance scope

Experiment 05 contains two model sizes (997,632 and 1,756,160 parameters),
three paired seeds at each size, on a fixed Tiny Shakespeare split. The
100-step pilot and exploratory 1,000-step extension retain separate manifests,
raw records and reports. Both horizons reuse the same seeds/initializations:
they are correlated training trajectories, not independent replication.
Within each size/seed, the two families start with identical parameter hashes
and receive identical batches. Validation uses all 111,488 tokens in complete
64-token windows, with the final remainder excluded consistently.

Frozen input hashes, exact record counts, paired initialization hashes and
first-100-step loss/gradient history identity checks all passed. The extension
contains 12,000 recorded model updates and 6,144,000 processed training tokens
across both families. No defined loss spike or divergence was observed.

The extension was specified after observing the pilot and changes only the
number of updates. It is explicitly exploratory. Six size/seed pairs are
reported at each horizon; results across horizons are not pooled for inference.
The reported confidence intervals describe seed variation conditional on this
one dataset/recipe. These runs do not validate a billion-parameter model or
establish a large-scale stability advantage. No historical training other than
the added experiment 05 was rerun for this release.

Performance measurements cover 36 CPU cells: three gates, two workloads,
two execution modes and three dtypes. All measured outputs and gradients were
finite. Raw samples and configuration are in `benchmarks/results_cpu.json`;
`benchmarks/RESULTS.md` is generated from that JSON. The final microbenchmark
run completed after the original pilot workers stopped; this remains a shared
host, with no isolation or GPU measurement. Tensor storage totals are exact
for the explicitly counted tensors and do not represent CPU peak allocation.
Native CPU peaks remain unavailable (`null`). CUDA and compilation support
are runnable harness features, not shipped accelerator results.

## Archive integrity

The complete source ZIP includes the original lab and checkpoints, new code,
results, documentation, and built distributions. It excludes transient caches
and build metadata. `SHA256SUMS` covers the release files; after extraction run:

```bash
python scripts/verify_checksums.py
```

The checksum list detects changed/missing bytes; it is not a digital signature.
