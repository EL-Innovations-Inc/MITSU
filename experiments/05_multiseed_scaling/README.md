# 05 · Multi-seed larger-model follow-up (v1.0.1)

This experiment adds actual, paired three-seed measurements at model sizes larger
than the old 483k-parameter pilot. It also supplies a longer follow-up configuration.
The shipped runs at 100 and 1,000 updates are **budget-limited diagnostics**, not
evidence at commercial LLM scale and not a new scaling-law fit.

Read the original [100-update pilot report](results/pilot/REPORT.md) and
[1,000-update extension report](results/extension/REPORT.md), machine-readable
summary JSON, full per-step worker JSONL files and run manifests alongside each. Both model families have the same
initial weights and batch order within every seed pair. See [PROTOCOL.md](PROTOCOL.md)
for decisions made before official runs, including the spike definition.

## Reproduce

From the repository root, install the experiments dependencies, then:

```bash
python experiments/05_multiseed_scaling/get_data.py
python experiments/05_multiseed_scaling/bench.py --freeze-only --output /tmp/mitsu-pilot
python experiments/05_multiseed_scaling/bench.py --output /tmp/mitsu-pilot
python experiments/05_multiseed_scaling/analyze.py /tmp/mitsu-pilot
```

The download verifies the exact dataset SHA-256. The original UTF-8 text is
included for offline reproducibility. Tiny Shakespeare originates from
[Karpathy's char-rnn dataset](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)
and contains public-domain Shakespeare excerpts. The data itself is not newly
licensed by this software's MIT grant.

Use a **new output directory** for a clean rerun. A manifest is written before
training; code/config/dataset changes against an existing freeze are rejected.
Completed records are skipped when resuming the same queue. Interrupted individual
runs restart from their initial seed; this runner does not save model checkpoints.
If using two workers, keep the worker count fixed throughout:

```bash
python experiments/05_multiseed_scaling/bench.py --output /tmp/mitsu-pilot --worker 0 --workers 2
python experiments/05_multiseed_scaling/bench.py --output /tmp/mitsu-pilot --worker 1 --workers 2
```

Run the `--freeze-only` command above first, then launch those two commands in
separate terminals. Two workers started at the same moment on a directory with no
manifest can collide while the manifest is being written, and one of them stops
with a JSON error; starting it again continues normally. Each worker owns a disjoint
set of full pairs; the run order reverses on odd seeds. Avoid concurrent identical
worker IDs writing to the same directory.

## Reproduce the 1,000-update extension

```bash
python experiments/05_multiseed_scaling/bench.py --config experiments/05_multiseed_scaling/extension.json --output /tmp/mitsu-extension
python experiments/05_multiseed_scaling/analyze.py /tmp/mitsu-extension
```

The extension was designed after inspecting the pilot. It restarts the same
initial seeds with only the update limit increased to 1,000; the first 100
updates reproduce the original trajectories. These are six seed-size pairs
measured at two correlated horizons, **not independent replication** and not
24 independent trials. Never pool the two horizons for significance or seed
counts. All original 100-update records are retained. Run `python
experiments/05_multiseed_scaling/audit.py` to verify source/config/dataset hashes,
complete records, pair initial weights, and exact first-100-step loss and
gradient-norm history prefixes. The resulting `PREFIX_AUDIT.json` records those
checks.

## Longer full profile (not run)

```bash
python experiments/05_multiseed_scaling/bench.py --config experiments/05_multiseed_scaling/full.json --output /tmp/mitsu-full
python experiments/05_multiseed_scaling/analyze.py /tmp/mitsu-full
```

`full.json` requests five seeds, widths 192/256/512, four blocks, 1,000 updates.
This is still a research benchmark, not billion-parameter validation. Increasing
training tokens per parameter, adding datasets, profiling accelerator mixed
precision, and repeating complete runs remain necessary.
