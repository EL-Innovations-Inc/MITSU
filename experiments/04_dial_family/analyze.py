"""Report results against prereg7. Every variant is paired with `mitsu` on the
same seeds:  delta = variant - mitsu,  t = mean(delta) / (sd(delta)/sqrt(n)).

Usage:  python3 analyze.py            # prints tables, writes results/summary.json
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TCRIT = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 10: 2.262}


def load(fn):
    p = os.path.join(HERE, "results", fn)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def table(rows, key, metric, higher_better, scale, label, seeds=None):
    by = {}
    for r in rows:
        if seeds is not None and r["seed"] not in seeds:
            continue
        by.setdefault(r[key], {})[r["seed"]] = r
    if "mitsu" not in by:
        return None
    base = by["mitsu"]
    out = []
    for name, d in by.items():
        ss = sorted(set(d) & set(base))
        v = np.array([d[s][metric] for s in ss], float) * scale
        b = np.array([base[s][metric] for s in ss], float) * scale
        div = int(sum(d[s]["diverged"] for s in d))
        n = len(ss)
        dl = v - b
        mean_d = float(np.mean(dl)) if n else float("nan")
        se = float(np.std(dl, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        t = mean_d / se if (n > 1 and se > 0) else (0.0 if name == "mitsu" else float("nan"))
        pooled = float(np.sqrt((np.var(v) + np.var(b)) / 2)) if n else float("nan")
        good = mean_d > 0 if higher_better else mean_d < 0
        crit = TCRIT.get(n, 2.776)
        verdict = "baseline" if name == "mitsu" else ("diverged" if div else ("parity" if not (abs(t) >= crit) else ("BETTER (screen)" if good else "worse")))
        out.append(dict(name=name, n=n, mean=float(np.mean(v)) if n else float("nan"), std=float(np.std(v)) if n else float("nan"),
                        delta=mean_d, se=se, t=t, pooled=pooled, within_pooled=bool(abs(mean_d) < pooled) if n else False,
                        diverged=div, params=d[ss[0]]["params"] if n else None, verdict=verdict,
                        dials={s: d[s].get("dials") for s in ss}))
    out.sort(key=lambda r: -r["mean"] if higher_better else r["mean"])
    print(f"\n=== {label} ===")
    print(f"{'variant':24s} {'n':>2s} {'mean':>8s} {'std':>6s} {'delta':>8s} {'se':>6s} {'t':>7s} {'div':>3s} {'params':>7s}  verdict")
    for r in out:
        print(f"{r['name']:24s} {r['n']:2d} {r['mean']:8.4f} {r['std']:6.4f} {r['delta']:+8.4f} {r['se']:6.4f} {r['t']:+7.2f} {r['diverged']:3d} {r['params']:7d}  {r['verdict']}")
    return out


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    summ = {}
    for ds in ("mnist", "fashion"):
        rows = load(f"cls_{ds}{tag}.jsonl")
        if rows:
            summ[ds] = table(rows, "cell", "acc", True, 100.0, f"{ds} test accuracy % (higher is better){tag}")
    rows = load(f"gpt{tag}.jsonl")
    if rows:
        full = sorted({r["seed"] for r in rows})
        nvar = len({r["variant"] for r in rows})
        complete = [s for s in full if sum(1 for r in rows if r["seed"] == s) == nvar]
        summ["gpt"] = table(rows, "variant", "val", False, 1.0, f"char-GPT val loss, nats (lower is better){tag}; complete seeds {complete}", seeds=complete or None)
    json.dump(summ, open(os.path.join(HERE, "results", f"summary{tag}.json"), "w"), indent=1)
