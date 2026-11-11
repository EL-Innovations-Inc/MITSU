#!/usr/bin/env python3
"""Regenerate a readable timing table from measured JSON (no benchmark rerun)."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/RESULTS.md"))
    args = parser.parse_args()
    report = json.loads(args.source.read_text())
    env, cfg = report["environment"], report["config"]
    groups = {}
    skipped = []
    for row in report["results"]:
        key = (row["dtype"], row["workload"], row["mode"])
        if row["status"] == "ok":
            groups.setdefault(key, {})[row["family"]] = row["median_ms"]
        else:
            skipped.append(f"{key}: {row['family']} — {row['reason']}")
    lines = [f"# Recorded {cfg['device']} performance sample", "",
             f"Measured {env.get('measured_utc', 'timestamp not recorded')} using PyTorch {env['pytorch']}, "
             f"{cfg['threads']} thread(s), {'compiled' if cfg['compiled'] else 'eager'} execution. "
             "This was a shared host without process isolation; repeat on deployment hardware before making a performance decision.", "",
             f"Median milliseconds across {cfg['repeats']} repeats after {cfg['warmup']} warmups. Lower is faster. "
             f"See `{args.source.name}` for every timing sample, ranges, complete configuration and explicit tensor sizes.", "",
             "| dtype | workload | operation | MITSU ms | SiLU ms | GELU ms |",
             "|---|---|---|---:|---:|---:|"]
    for (dtype, workload, mode), values in groups.items():
        cells = [f"{values[name]:.4f}" if name in values else "unavailable" for name in ["mitsu", "silu", "gelu"]]
        lines.append(f"| {dtype} | {workload} | {mode} | " + " | ".join(cells) + " |")
    lines += ["", "MITSU's implementation contains multiple tensor operations and promotes low-precision inputs internally for numerical safety; the built-in gates have specialized kernels. Projection work reduces relative activation overhead in FFNs, and small timing differences are sensitive to shared-host noise. These measurements do not establish training quality or an ordering on other hardware.", "",
              "CPU native peak allocator memory is not measured and remains `null`. Explicit tensor sizes count inputs, outputs, parameters and gradients, excluding saved autograd tensors, temporary tensors and backend workspaces. These counts must not be interpreted as peak-memory measurements. CUDA runs additionally report separate allocator peaks.", "",
              f"{sum(r['status'] == 'ok' and r['finite_outputs_and_gradients'] for r in report['results'])}/{len(report['results'])} measured cells had finite outputs and gradients.", ""]
    if cfg["device"] == "cpu":
        lines += ["No CUDA measurements are included; CPU timings cannot establish GPU speed.", ""]
    if skipped:
        lines += ["Unsupported or failed cells:", ""] + [f"- {s}" for s in skipped] + [""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
