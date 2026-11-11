"""EXPLORATORY (not pre-registered): can c act as a filter?

Four deterministic checks on the curve itself plus a read-out of the learned c
values. No training happens here.

    python3 filter_check.py
"""
import json, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


def softplus(u):
    return np.where(u > 30, u, np.log1p(np.exp(np.minimum(u, 30))))


def act(u, c):
    s = softplus(u)
    return s * (s - 1) / (s + c)


u = np.linspace(-12, 40, 520001)

print("A. Cut-off: where the unit opens, by c")
print(f"{'c':>5s} {'slope at u=0':>13s} {'half-open at u':>15s} {'tail = u minus':>15s} {'out for u=1':>12s} {'out for u=8':>12s}")
for c in CS:
    d = np.gradient(act(u, c), u)
    half = u[np.argmax(d >= 0.5)]
    print(f"{c:5g} {d[np.argmin(abs(u))]:13.3f} {half:15.2f} {1 + c:15.2f} {float(act(np.array(1.0), c)):12.3f} {float(act(np.array(8.0), c)):12.3f}")

print("\nB. Noise gate: spikes of height 8 on 10% of positions, unit Gaussian noise everywhere")
rng = np.random.default_rng(0)
n = 400_000
spike = rng.random(n) < 0.10
x = rng.standard_normal(n) + 8.0 * spike
print(f"{'':>10s} {'mean out at spikes':>19s} {'rms out elsewhere':>18s} {'ratio':>8s}")
print(f"{'input':>10s} {x[spike].mean():19.3f} {np.sqrt((x[~spike] ** 2).mean()):18.3f} {x[spike].mean() / np.sqrt((x[~spike] ** 2).mean()):8.1f}")
erf = np.vectorize(math.erf)
refs = {"relu": np.maximum(x, 0), "gelu": x * 0.5 * (1 + erf(x / math.sqrt(2)))}
for name, y in refs.items():
    print(f"{name:>10s} {y[spike].mean():19.3f} {np.sqrt((y[~spike] ** 2).mean()):18.3f} {y[spike].mean() / np.sqrt((y[~spike] ** 2).mean()):8.1f}")
for c in CS:
    y = act(x, c)
    print(f"{'c=%g' % c:>10s} {y[spike].mean():19.3f} {np.sqrt((y[~spike] ** 2).mean()):18.3f} {y[spike].mean() / np.sqrt((y[~spike] ** 2).mean()):8.1f}")

print("\nC. Is c just a bias shift? Largest gap between MITSU_c(u) and MITSU_2(u - (c - 2)), u in [-12, 40]")
for c in CS:
    gap = np.abs(act(u, c) - act(u - (c - 2.0), 2.0))
    print(f"   c={c:<5g} max gap {gap.max():.3f} at u={u[gap.argmax()]:+.2f}")

print("\nD. What the networks chose")
def L(f):
    p = os.path.join(HERE, "results", f)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
for bench, main, key, scale in (("mnist", "cls_mnist.jsonl", "cell", 100), ("fashion", "cls_fashion.jsonl", "cell", 100), ("gpt", "gpt.jsonl", "variant", 1)):
    rows = L(main); metric = "val" if bench == "gpt" else "acc"
    base = {r["seed"]: r[metric] for r in rows if r[key] == "mitsu"}
    for layer in ("l1", "l2"):
        v = [r["dials"][layer][0] for r in rows if r[key] == "learnc"]
        print(f"   {bench:8s} one c per layer, {layer}: " + " ".join(f"{x:.2f}" for x in v))
    ur = L(f"exploratory_unit_{bench}.jsonl")
    if ur:
        d = np.array([r[metric] - base[r["seed"]] for r in ur]) * scale
        t = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
        print(f"   {bench:8s} one c per unit: mean {np.mean([r[metric] for r in ur]) * scale:.4f}  delta vs mitsu {d.mean():+.4f}  t {t:+.2f}  diverged {sum(r['diverged'] for r in ur)}")
        for layer in ("l1", "l2"):
            allc = np.array([x for r in ur for x in r["dials"][layer]])
            per = [np.std(r["dials"][layer]) for r in ur]
            print(f"      {layer}: {len(allc)} units over {len(ur)} runs  mean {allc.mean():.2f}  spread within a layer (sd) {np.mean(per):.2f}"
                  f"  range {allc.min():.2f}-{allc.max():.2f}  5%-95% {np.quantile(allc, .05):.2f}-{np.quantile(allc, .95):.2f}  below 2: {100 * (allc < 2).mean():.0f}%")
