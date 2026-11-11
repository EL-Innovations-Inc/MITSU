"""Report matched complete seed pairs; withhold inference for incomplete evidence."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
from scipy.stats import t

HERE = Path(__file__).resolve().parent


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def average(values):
    values = [x for x in values if finite(x)]
    return statistics.mean(values) if values else None


def fmt(value, spec=".6f"):
    return format(value, spec) if finite(value) else "—"


def paired_summary(rows, width, expected_seeds, expected_steps, tokens_per_step):
    by = {}
    selected = [r for r in rows if r["width"] == width]
    for r in selected:
        key = (r["seed"], r["family"])
        if key in by:
            raise ValueError(f"Duplicate result: width={width}, seed/family={key}")
        by[key] = r
    pairs, exclusions = [], []
    unexpected = [list(key) for key in by if key[0] not in expected_seeds or key[1] not in ("mitsuglu", "swiglu")]
    for seed in expected_seeds:
        a, b = by.get((seed, "mitsuglu")), by.get((seed, "swiglu"))
        if a is None or b is None:
            exclusions.append(dict(seed=seed, reason="incomplete pair")); continue
        if a["initial_parameter_sha256"] != b["initial_parameter_sha256"]:
            raise ValueError(f"Pair initial weights differ at width={width} seed={seed}")
        if a["parameters"] != b["parameters"]:
            raise ValueError("Pair parameter budgets differ")
        if a["diverged"] or b["diverged"] or any(not finite(r["validation_loss"]) for r in (a, b)):
            exclusions.append(dict(seed=seed, reason="divergent/nonfinite pair")); continue
        if any(r["steps_completed"] != expected_steps for r in (a, b)):
            exclusions.append(dict(seed=seed, reason="training horizon differs from configured updates")); continue
        pairs.append(dict(seed=seed, mitsu=a["validation_loss"], silu=b["validation_loss"], delta=a["validation_loss"] - b["validation_loss"]))
    deltas = [p["delta"] for p in pairs]
    n = len(deltas)
    mean = average(deltas)
    se = statistics.stdev(deltas) / math.sqrt(n) if n > 1 else None
    # All expected pairs must survive; missing/failed runs must never quietly
    # become a selected subset supporting inference. Identical seed deltas also
    # cannot justify a zero-width uncertainty estimate.
    if exclusions or unexpected:
        inference_status = "withheld: missing, failed, mismatched, or unexpected records"
    elif n < 2:
        inference_status = "withheld: fewer than two complete pairs"
    elif se is None or se <= 0:
        inference_status = "withheld: degenerate seed variance"
    else:
        inference_status = "complete paired Student-t interval; descriptive exploratory evidence"
    ci = [mean - float(t.ppf(.975, n - 1)) * se, mean + float(t.ppf(.975, n - 1)) * se] if inference_status.startswith("complete") else None
    families = {}
    for family in ("mitsuglu", "swiglu"):
        fr = [r for r in selected if r["family"] == family]
        families[family] = dict(runs=len(fr), divergences=sum(bool(r["diverged"]) for r in fr),
            loss_spikes=sum(len(r["loss_spike_steps"]) for r in fr),
            steps_completed_total=sum(r["steps_completed"] for r in fr),
            training_tokens_total=sum(r["steps_completed"] for r in fr) * tokens_per_step,
            training_seconds_mean=average([r["training_seconds"] for r in fr]),
            train_tokens_per_second_mean=average([r["train_tokens_per_second"] for r in fr]),
            validation_loss_mean_all_finite_records=average([r["validation_loss"] for r in fr]))
    return dict(width=width, parameters=next((r["parameters"] for r in selected), None),
        n_pairs=n, expected_pairs=len(expected_seeds), paired_mean_delta=mean,
        paired_mitsu_mean=average([p["mitsu"] for p in pairs]), paired_silu_mean=average([p["silu"] for p in pairs]),
        paired_standard_error=se, paired_ci95=ci, inference_status=inference_status,
        pairs=pairs, exclusions=exclusions, unexpected_records=unexpected, families=families,
        interpretation="Exploratory budget-limited evidence; no established equivalence or large-scale stability claim.")


def write_report(directory, cfg, summary):
    lines = ["# v1.0.1 multi-seed larger-model results", "",
        "Measured CPU FP32 runs; final full-window validation loss on Tiny Shakespeare. Delta = MitsuGLU minus SwiGLU; positive favors SwiGLU. Intervals are paired Student-t 95% CIs across seeds when all configured pairs are complete and finite. Primary family means use exactly the same valid pairs.", "",
        "| Parameters | Complete / expected pairs | MitsuGLU loss | SwiGLU loss | Mean delta | 95% CI |",
        "|---:|---:|---:|---:|---:|---|"]
    for s in summary:
        ci = s["paired_ci95"]
        interval = f"[{fmt(ci[0], '+.6f')}, {fmt(ci[1], '+.6f')}]" if ci else "withheld"
        lines.append(f"| {fmt(s['parameters'], ',.0f')} | {s['n_pairs']} / {s['expected_pairs']} | {fmt(s['paired_mitsu_mean'])} | {fmt(s['paired_silu_mean'])} | {fmt(s['paired_mean_delta'], '+.6f')} | {interval} |")
    lines += ["", f"Each configured run: {cfg['steps']:,} updates, batch {cfg['batch']}, context {cfg['context']}; {len(cfg['seeds'])} paired seeds per size and {cfg['steps'] * cfg['batch'] * cfg['context']:,} training tokens per completed model. Models exceed the original 483,184-parameter maximum. This remains a budget-limited training diagnostic, not a compute-optimal scaling-law experiment.", ""]
    for s in summary:
        direction = "SwiGLU" if finite(s["paired_mean_delta"]) and s["paired_mean_delta"] > 0 else "MitsuGLU" if finite(s["paired_mean_delta"]) and s["paired_mean_delta"] < 0 else "neither family"
        lines.append(f"At {fmt(s['parameters'], ',.0f')} parameters, the observed paired mean favors {direction}. Inference status: {s['inference_status']}. This is not a general superiority claim.")
    lines += ["", "## Stability, exposure, and runtime diagnostics (all recorded runs)", "",
        "| Parameters | Family | Divergences | Loss spikes | Updates observed | Train tokens observed | Mean train seconds | Mean train tokens/s |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for s in summary:
        for family, f in s["families"].items():
            lines.append(f"| {fmt(s['parameters'], ',.0f')} | {family} | {f['divergences']} | {f['loss_spikes']} | {f['steps_completed_total']:,} | {f['training_tokens_total']:,} | {fmt(f['training_seconds_mean'], '.2f')} | {fmt(f['train_tokens_per_second_mean'], ',.0f')} |")
    lines += ["", "Loss spike definition: after at least 20 completed steps, loss > prior-20-step median + 0.5 nats. Global gradient clipping at norm 1.0 is active in both families. Zero observed spikes or divergences under this recipe cannot establish a stability advantage. Training histories and gradient norms remain in the JSONL records.", "",
        "Runtime is observed on a shared CPU while up to two workers run concurrently; do not use these numbers to rank activation kernels. See the dedicated [activation microbenchmark](../../../../benchmarks/README.md).", "",
        "## Limits and provenance", "",
        f"- {len(cfg['seeds'])} seeds give limited uncertainty estimates; intervals cover seed variation conditional on one dataset split and one fixed recipe.",
        "- No family-specific hyperparameter tuning. No multiple-dataset replication. No word-level tokenizer or long-context evaluation.",
        "- The gain multiplier is 3.0374 for MITSU; the raw derivative bound does not bound the scaled GLU block.",
        "- The 100-update pilot and 1,000-update extension have separate manifests and raw records. The extension was designed after inspecting the pilot and is exploratory, not an independent replication. `full.json` is an unrun suggested follow-up.",
        "- Initial parameter hashes match within each reported pair. Dataset SHA-256, code/config hashes, exact versions and machine context are in `manifest.json`. The freeze is local, not independently timestamped preregistration.",
        "- No scaling exponent is fitted: varying width at a fixed update budget is not evidence for compute-optimal scaling.",
        "- Incomplete, divergent, or mismatched pairs are listed below and cause inference to be withheld for the whole size. Valid-pair means, if shown, are descriptive only. Zero seed variance also withholds an interval.", ""]
    exclusions = [{"width": s["width"], **e} for s in summary for e in s["exclusions"]]
    lines += ["Excluded/incomplete pairs: " + (json.dumps(exclusions) if exclusions else "none.")]
    unexpected = [{"width": s["width"], "records": s["unexpected_records"]} for s in summary if s["unexpected_records"]]
    lines += ["Unexpected records: " + (json.dumps(unexpected) if unexpected else "none."), ""]
    (directory / "REPORT.md").write_text("\n".join(lines))
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", nargs="?", type=Path, default=HERE / "results" / "pilot")
    a = p.parse_args()
    manifest = json.loads((a.directory / "manifest.json").read_text())
    rows = []
    for path in sorted(a.directory.glob("worker*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    cfg = manifest["config"]
    summary = [paired_summary(rows, w, cfg["seeds"], cfg["steps"], cfg["batch"] * cfg["context"]) for w in cfg["widths"]]
    (a.directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = write_report(a.directory, cfg, summary)
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
