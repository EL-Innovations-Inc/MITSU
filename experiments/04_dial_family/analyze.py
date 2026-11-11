"""v1.0.1 reanalysis: paired confidence intervals and practical equivalence.

This is a POST HOC reanalysis, not a change to prereg7. Original raw results,
summary.json, summary_confirm.json and reports remain historical records.

Usage: python analyze.py [_confirm]
Writes results/summary_v1_0_1[<tag>].json and report_v1_0_1[<tag>].md.
See ../../docs/STATISTICS.md for assumptions and interpretation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
MARGINS = {"classification_pp": 0.5, "gpt_nats": 0.05}


def load(fn):
    path = HERE / "results" / fn
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def paired_statistics(deltas, margin, alpha=0.05):
    """Student-t interval for a paired mean; conservative CI equivalence.

    The 95% CI must lie STRICTLY inside (-margin, margin). This is more
    conservative than the 90% interval used by a 5% TOST. No inference is
    produced for nonfinite differences, fewer than two pairs, or zero sample
    variance: observed constancy is not evidence of zero population variance.
    """
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be positive and finite")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    delta = np.asarray(deltas, dtype=float)
    if delta.ndim != 1:
        raise ValueError("paired differences must be one-dimensional")
    if not np.isfinite(delta).all():
        raise ValueError("paired differences must all be finite")
    n = len(delta)
    out = dict(n=n, delta=float(delta.mean()) if n else None, se=None,
               t=None, p_two_sided=None, ci95=None, equivalent=None,
               margin=float(margin), alpha=float(alpha), status="insufficient paired seeds")
    if n < 2:
        return out
    se = float(delta.std(ddof=1) / np.sqrt(n))
    out["se"] = se
    if se == 0:
        out["status"] = "degenerate paired variance"
        return out
    t_stat = out["delta"] / se
    critical = float(student_t.ppf(1 - alpha / 2, n - 1))
    ci = [out["delta"] - critical * se, out["delta"] + critical * se]
    out.update(t=float(t_stat), p_two_sided=float(2 * student_t.sf(abs(t_stat), n - 1)),
               ci95=ci, equivalent=bool(ci[0] > -margin and ci[1] < margin), status="ok")
    return out


def _index(rows, key, seeds=None):
    by = {}
    for row in rows:
        if seeds is not None and row["seed"] not in seeds:
            continue
        name, seed = row[key], row["seed"]
        cell = by.setdefault(name, {})
        if seed in cell:
            raise ValueError(f"duplicate result for {name!r}, seed {seed!r}; resolve duplicates explicitly")
        cell[seed] = row
    return by


def _finite(value):
    return value is not None and bool(np.isfinite(value))


def table(rows, key, metric, higher_better, scale, label, seeds=None, margin=None):
    """Return per-cell paired summaries; preserve descriptive legacy field names.

    n counts valid pairs. Missing, diverged or nonfinite pairs are listed and
    prevent an inferential verdict, even when a subset yields a finite CI.
    """
    margin = MARGINS["classification_pp" if higher_better else "gpt_nats"] if margin is None else margin
    by = _index(rows, key, seeds)
    if "mitsu" not in by:
        return None
    base = by["mitsu"]
    out = []
    for name, cell in by.items():
        paired = sorted(set(cell) & set(base))
        missing_variant = sorted(set(base) - set(cell))
        missing_baseline = sorted(set(cell) - set(base))
        failed = [s for s in paired if cell[s].get("diverged", False) or base[s].get("diverged", False)]
        invalid = [s for s in paired if s not in failed and
                   not (_finite(cell[s].get(metric)) and _finite(base[s].get(metric)))]
        valid = [s for s in paired if s not in failed and s not in invalid]
        v = np.array([cell[s][metric] * scale for s in valid], dtype=float)
        b = np.array([base[s][metric] * scale for s in valid], dtype=float)
        stats = paired_statistics(v - b, margin)
        div = sum(bool(r.get("diverged", False)) for r in cell.values())
        pooled = float(np.sqrt((np.var(v) + np.var(b)) / 2)) if len(v) else None
        difference = "not established"
        if stats["ci95"] is not None:
            low, high = stats["ci95"]
            if low > 0 or high < 0:
                good = stats["delta"] > 0 if higher_better else stats["delta"] < 0
                difference = "better" if good else "worse"
        if name == "mitsu":
            verdict = "baseline"
        elif failed or div:
            verdict = "diverged"
        elif invalid:
            verdict = "invalid observations"
        elif missing_variant or missing_baseline:
            verdict = "incomplete paired data"
        elif stats["status"] != "ok":
            verdict = stats["status"]
        elif stats["equivalent"]:
            verdict = "equivalent within margin"
        elif difference != "not established":
            verdict = difference + " (uncorrected screen)"
        else:
            verdict = "inconclusive"
        eligible = name != "mitsu" and not (failed or div or invalid or missing_variant or missing_baseline)
        out.append(dict(name=name, **stats,
                        mean=float(v.mean()) if len(v) else None,
                        std=float(v.std()) if len(v) else None, pooled=pooled,
                        within_pooled=bool(abs(stats["delta"]) < pooled) if pooled is not None else None,
                        diverged=int(div), paired_failures=failed, invalid_seeds=invalid,
                        paired_seeds=valid, n_paired_total=len(paired),
                        missing_variant_seeds=missing_variant, missing_baseline_seeds=missing_baseline,
                        inference_eligible=eligible and stats["status"] == "ok",
                        difference_screen=difference if eligible else "withheld",
                        equivalence_established=bool(eligible and stats["equivalent"]),
                        params=next(iter(cell.values())).get("params"), verdict=verdict,
                        dials={s: cell[s].get("dials") for s in valid}))
    out.sort(key=lambda r: (r["mean"] is None,
                           (-r["mean"] if higher_better else r["mean"]) if r["mean"] is not None else 0))
    print(f"\n=== {label} ===")
    print("variant                    n       mean      delta          paired 95% CI  verdict")
    for r in out:
        ci = "unavailable" if r["ci95"] is None else f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
        mean = "n/a" if r["mean"] is None else f"{r['mean']:.4f}"
        delta = "n/a" if r["delta"] is None else f"{r['delta']:+.4f}"
        print(f"{r['name']:25s} {r['n']:2d} {mean:>10s} {delta:>10s} {ci:>24s}  {r['verdict']}")
    return out


def _clean_json(value):
    """Convert legacy JSON NaN dials to standards-compliant nulls."""
    if isinstance(value, dict):
        return {key: _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_report(summary, path, tag):
    lines = ["# v1.0.1 paired reanalysis" + (f" ({tag})" if tag else ""), "",
             "Post hoc reanalysis of the historical dial-family results. Original preregistrations, "
             "raw records and reports are unchanged. These intervals do not independently reproduce training.", "",
             "Equivalence requires the paired 95% Student-t confidence interval to lie strictly inside "
             "±0.5 percentage points (classification) or ±0.05 nats (char-GPT). These v1.0.1 "
             "analysis choices are not retrospectively preregistered. Difference screens are uncorrected "
             "for multiple comparisons; neither equivalence nor a screen establishes superiority across tasks.", ""]
    for dataset, records in summary.items():
        unit = "nats" if dataset == "gpt" else "percentage points"
        lines += [f"## {dataset}", "", f"Deltas and intervals are variant minus MITSU, in {unit}.", "",
                  "| Variant | Valid pairs | Mean delta | Paired 95% CI | Interpretation |",
                  "|---|---:|---:|---|---|"]
        for r in records:
            delta = "—" if r["delta"] is None else f"{r['delta']:+.4f}"
            ci = "unavailable" if r["ci95"] is None else f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
            lines.append(f"| `{r['name']}` | {r['n']} | {delta} | {ci} | {r['verdict']} |")
        lines += [""]
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", nargs="?", default="", choices=("", "_confirm"))
    args = parser.parse_args(argv)
    summary = {}
    for dataset in ("mnist", "fashion"):
        rows = load(f"cls_{dataset}{args.tag}.jsonl")
        if rows:
            summary[dataset] = table(rows, "cell", "acc", True, 100.0, f"{dataset} accuracy %")
    rows = load(f"gpt{args.tag}.jsonl")
    if rows:
        summary["gpt"] = table(rows, "variant", "val", False, 1.0, "char-GPT loss, nats")
    summary = {key: value for key, value in summary.items() if value is not None}
    target = HERE / "results" / f"summary_v1_0_1{args.tag}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(_clean_json(summary), indent=2, allow_nan=False) + "\n")
    write_report(summary, HERE / "results" / f"report_v1_0_1{args.tag}.md", args.tag)
    return summary


if __name__ == "__main__":
    main()
