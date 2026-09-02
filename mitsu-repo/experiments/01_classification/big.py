import numpy as np, json, sys, time
from bench import MLP, Tri, Linear, nparams, train_eval

name = sys.argv[1]
z = np.load(f'data/{name}.npz')
XTR, YTR, XTE, YTE = z['Xtr'].astype(np.float64), z['ytr'], z['Xte'].astype(np.float64), z['yte']
d, C = 784, 10
cells = [('linear', lambda r: Linear(d,C,r)),
         ('relu64', lambda r: MLP(d,64,C,'relu',r)),
         ('gelu64', lambda r: MLP(d,64,C,'gelu',r)),
         ('mitsu64', lambda r: MLP(d,64,C,'mitsu',r)),
         ('gelu86', lambda r: MLP(d,86,C,'gelu',r)),
         ('mitsu86', lambda r: MLP(d,86,C,'mitsu',r)),
         ('tri64', lambda r: Tri(d,64,2,C,r))]
out = {}
for cname, build in cells:
    accs = []; divs = 0; t0 = time.time()
    for seed in range(5):
        rng = np.random.default_rng(1000+seed)
        idx = []
        for c in range(C):
            ci = np.where(YTR==c)[0]; idx.append(rng.choice(ci, 1000, replace=False))
        idx = np.concatenate(idx)
        Xtr, ytr = XTR[idx], YTR[idx]
        mu, sd = Xtr.mean(0), Xtr.std(0)+1e-8
        Xtr, Xte = (Xtr-mu)/sd, (XTE-mu)/sd
        m = build(np.random.default_rng(seed))
        acc, div = train_eval(m, Xtr, ytr, Xte, YTE, epochs=12, lr=3e-3, bs=128, seed=seed)
        accs.append(acc); divs += div
        if seed == 0: p = nparams(m)
    out[cname] = dict(mean=float(np.mean(accs)), std=float(np.std(accs)), params=p, diverged=divs)
    print(f'{name} {cname:8s} params={p:6d}  acc={np.mean(accs)*100:5.2f} ± {np.std(accs)*100:4.2f}  div={divs}  ({time.time()-t0:.0f}s)', flush=True)
json.dump(out, open(f'/home/claude/results_{name}.json','w'), indent=1)
