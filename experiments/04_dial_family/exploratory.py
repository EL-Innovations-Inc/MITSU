"""EXPLORATORY (not pre-registered): pool the Fashion screen seeds (0-4) with the
confirmation seeds (5-9) for the nine screen candidates. Paired with mitsu."""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
L = lambda f: [json.loads(l) for l in open(os.path.join(HERE, "results", f))]
rows = L("cls_fashion.jsonl") + L("cls_fashion_confirm.jsonl")
acc = {}
for r in rows:
    acc.setdefault(r["cell"], {})[r["seed"]] = r["acc"] * 100
names = sorted({r["cell"] for r in L("cls_fashion_confirm.jsonl")} - {"mitsu"})
print(f"{'variant':22s} {'n':>2s} {'delta':>7s} {'t':>6s}  ahead   (critical |t| for 10 seeds: 2.26)")
res = []
for n in names:
    ss = sorted(set(acc[n]) & set(acc["mitsu"]))
    d = np.array([acc[n][s] - acc["mitsu"][s] for s in ss])
    res.append((d.mean(), n, len(ss), d.mean() / (d.std(ddof=1) / np.sqrt(len(d))), int((d > 0).sum())))
for m, n, k, t, a in sorted(res, reverse=True):
    print(f"{n:22s} {k:2d} {m:+7.3f} {t:+6.2f}  {a}/{k}")
