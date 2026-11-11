# Statistical interpretation in v1.0.1

Version 1.0.1 adds a **post hoc reanalysis** of experiment 04. It does not edit
`prereg7.md`, `FROZEN.txt`, raw JSONL observations, historical summaries, logs or
`results7.md`. The old documents describe the rules and language used at the
time. Their use of “parity” must not be read as a formal equivalence result.

## Difference and equivalence answer different questions

A two-sided test that does not reject a zero difference has insufficient evidence
of a difference. It does **not** demonstrate equivalence. A wide interval can
contain both zero and practically important advantages or disadvantages.

The new analysis pairs each variant with MITSU by seed and estimates the mean of
`variant - MITSU`. It reports a two-sided 95% Student-t confidence interval with
`n - 1` degrees of freedom, using the sample standard deviation of the **paired
differences**. This accounts for pairing rather than treating the two cells as
independent samples. Accuracy deltas are in percentage points; loss deltas are in
nats. Higher accuracy and lower loss are favorable.

Practical equivalence is established only if the entire paired 95% confidence
interval lies **strictly within** a prespecified margin for this reanalysis:

| Metric | Symmetric practical margin | Status |
|---|---:|---|
| MNIST/Fashion-MNIST accuracy | ±0.5 percentage points | New, post hoc v1.0.1 choice |
| Character-GPT validation loss | ±0.05 nats | Magnitude inherited from prereg4; interval rule is new and post hoc |

This rule is more conservative than a standard 5% two one-sided test (TOST),
which uses a 90% interval. No claim is made that these new rules were registered
before the historical observations. They are sensitivity checks on existing data,
not independent confirmation or fresh-seed validation. Margins should be justified
for the intended application and frozen before a future confirmatory run.

A small effect can be statistically detectable **and** practically equivalent.
The JSON therefore records `difference_screen` separately from
`equivalence_established`. A favorable difference screen is uncorrected for the
many variants tested; it remains exploratory and is not a family-wise
superiority claim. Confidence intervals are marginal, not simultaneous.

## Assumptions and edge cases

- Seeds are treated as independent experimental repeats; within each repeat,
  variant and baseline are paired. Seeds vary training randomness, not the
  underlying dataset, so intervals do not quantify generalization to new tasks.
- Student-t inference assumes approximately normal paired differences. With only
  three or five pairs this assumption is difficult to assess, and uncertainty
  can be substantial. More independent seeds are the preferred improvement.
- Duplicate `(variant, seed)` observations raise an error. They are never silently
  overwritten or counted as independent observations.
- Pairing uses matching seeds. Missing baseline/variant seeds are listed; incomplete
  pairing withholds an inferential verdict instead of implying a complete run.
- Diverged or nonfinite pairs are explicitly listed and withhold inference for
  that cell. Descriptive statistics on remaining valid pairs are labeled by their
  actual count and must not be mistaken for the complete experiment.
- Fewer than two valid pairs cannot support a Student-t interval. Zero observed
  paired variance also withholds inference, rather than treating sample constancy
  as certainty about population variation. Baseline self-comparisons are labeled
  `baseline`, not evidence of an equivalence experiment.

## Reproduce the reanalysis

From `experiments/04_dial_family`:

```bash
python analyze.py
python analyze.py _confirm
```

These commands require NumPy and SciPy. They create:

- `results/summary_v1_0_1.json` and `results/report_v1_0_1.md`
- `results/summary_v1_0_1_confirm.json` and `results/report_v1_0_1_confirm.md`

They leave historical `summary.json`, `summary_confirm.json`, and the original
figure/table generators intact. JSON outputs use standard `null` values for
unavailable quantities rather than nonstandard JSON NaN/Infinity.
