"""Width-matched follow-up (pre-registration: prereg2.md).

Run 1 (bench.py) compared models at ~matched PARAMETER count, which forces the
tri-block tower to a narrower layer (16 units) than the MLPs (~28-38). This
script isolates the confound by matching WIDTH in both directions:

  * MLP-NARROW: relu/gelu/mitsu MLPs at H = 16 (tri's run-1 width) — do plain
    MLPs also suffer at 16 units/layer?
  * tri-WIDE:   the tri tower at W = run-1 MLP width per dataset — does width
    alone close tri's gap, at ~2.4x the parameters?

Run-1 cells carry over unchanged because every RNG is seeded identically.
Protocol is otherwise frozen to run 1: same splits, seeds, optimizer, epochs.

Usage:  python3 width.py        (writes results_width.json beside itself)
"""
import numpy as np, json
from bench import MLP, Tri, nparams, train_eval
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split

# dataset -> (loader, run-1 param-matched MLP width Hprev, used here as tri-WIDE's W)
sets = {'digits': (load_digits(), 28), 'cancer': (load_breast_cancer(), 33),
        'wine': (load_wine(), 36), 'iris': (load_iris(), 38)}
out = {}
for name, (ds, Hprev) in sets.items():
    X, y = ds.data.astype(float), ds.target
    d, C = X.shape[1], len(np.unique(y))
    epochs = 60 if name == 'digits' else 80          # frozen in prereg1
    out[name] = {}
    # Four new cells; lambdas close over d/C/Hprev on purpose.
    cells = [('relu16', lambda r: MLP(d, 16, C, 'relu', r)),
             ('gelu16', lambda r: MLP(d, 16, C, 'gelu', r)),
             ('mitsu16', lambda r: MLP(d, 16, C, 'mitsu', r)),
             ('triW', lambda r: Tri(d, Hprev, 2, C, r))]
    for cname, build in cells:
        accs = []; divs = 0
        for seed in range(5):
            # Identical split + standardization recipe as bench.py run 1.
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8   # train-set stats only
            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
            m = build(np.random.default_rng(seed))     # same init stream as run 1
            acc, div = train_eval(m, Xtr, ytr, Xte, yte, epochs, seed=seed)
            accs.append(acc); divs += div
            if seed == 0: p = nparams(m)
        out[name][cname] = dict(mean=float(np.mean(accs)), std=float(np.std(accs)), params=p, diverged=divs)
        print(f'{name:7s} {cname:8s} params={p:5d}  acc={np.mean(accs)*100:5.2f} ± {np.std(accs)*100:4.2f}  div={divs}')
json.dump(out, open('results_width.json', 'w'), indent=1)
