# v1.0.1 multi-seed larger-model pilot protocol

This protocol and machine-readable `pilot.json` are locally frozen in the run
manifest before the official pilot begins. This is not independently timestamped
preregistration. A software correctness smoke test may precede the freeze; it uses
synthetic data or seed 999, never the official seeds. All official runs are retained.

- Task: character-level next-token prediction on the verified Tiny Shakespeare
  text, contiguous first 90% train / last 10% validation, vocabulary from full text.
- Two transformer widths, 192 and 256; two pre-normalized blocks, four attention
  heads, context 64, learned positions, untied token/output embeddings. GLU hidden
  width 3x. Each size exceeds the previous 483,184-parameter maximum.
- MitsuGLU with frozen gain 3.0374 versus unscaled SwiGLU. Identical initial
  weights and training batch sequence within every pair. Gain is inherited from
  experiment 03, never tuned on this dataset. Family run order reverses on odd seeds.
- Three seeds (0, 1, 2), 100 updates each, batch 8. AdamW 0.002 peak learning
  rate, 20-step linear warmup, weight decay 0.01, global gradient clip 1.0.
- CPU FP32, one intra-op thread, deterministic algorithms. No checkpoint selection;
  report final losses. Validation covers all complete non-overlapping 64-token
  windows (at most 63 leftover tokens omitted), same for every model.
- Primary outcome: paired final validation loss delta (MitsuGLU minus SwiGLU),
  mean and two-sided Student-t 95% CI across seeds, separately for each size.
  Negative favors MitsuGLU. No superiority or equivalence conclusion is planned
  from this short pilot and three seeds, regardless of a CI excluding zero.
- Diagnostics: full per-step training loss and pre-clip gradient norm; runtime and
  tokens/second; clipping count; divergence if loss or gradient norm is nonfinite;
  loss spike when a loss exceeds the prior 20-step rolling median by >0.5 nats,
  beginning at step 21. Do not equate zero observed spikes with scale stability.
- No retuning, failed-run rescue, dropped seeds, or early stopping. Completed
  records are skipped only when resuming an interrupted queue. Incomplete pairs
  and divergences are listed; inference uses complete finite pairs only.
- Runtime measurements are noisy shared CPU observations, not hardware rankings.
  This task does not test FP16, large-corpus word-level language models, long
  training, compute-optimal scaling, or billion-parameter behavior.

`full.json` supplies an unrun follow-up: widths 192/256/512, four blocks, five
paired seeds, 1,000 updates. It is a reproducible suggested configuration, not
finished evidence. Repeating on multiple datasets and longer budgets remains
necessary before making large-scale claims.

## Exploratory 1,000-update extension (designed after the pilot)

After inspecting the 100-update outcomes, v1.0.1 adds a longer follow-up with
`extension.json`. The only configuration change is updates 100 → 1,000; the
20-update warmup, seeds, sizes, gain, optimizer, data order, loss-spike rule and
all remaining settings stay identical. Runs restart from the original initial
seeds rather than continue checkpoint files, so the first 100 updates reproduce
the pilot trajectory. The extension is locally frozen in a new output directory
before its runs. It is exploratory follow-up prompted by the pilot, not a
retrospective preregistration or an independent replication. All original pilot
records are retained. No further horizon increases or recipe changes are planned.

Training exposure is 512,000 tokens/model at 1,000 updates, still a short budget
relative to model size. Report all three paired seeds at each size, paired 95%
Student-t intervals, final validation loss, and the same diagnostic histories.
Do not infer superiority/equivalence from incomplete pairs or zero-variance seed
deltas; diagnostics remain descriptive when inference is withheld.
