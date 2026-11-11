"""Classification bench — MNIST / Fashion-MNIST, protocol inherited from the
repo's prereg3 / big.py:

  stratified 10,000-image training subset (1,000 per class, drawn per seed),
  full official 10,000-image test set, z-score on train stats, He init,
  batch 128, Adam lr 3e-3, 12 epochs, global grad-norm clip 5, float64.

Model: input -> H -> H -> 10, both hidden layers built from one variant spec.
Every variant uses H = 64 (55,050 parameters; `learnc` adds one number per
layer) and starts from identical weights for a given seed.

This is the bench script of experiment 04, reused unchanged apart from this
description; the variants come from this folder's variants.py.

Usage:  python3 bench_cls.py <mnist|fashion> <worker> <nworkers> <seed0> <seed1>
Appends one JSON line per run to results/cls_<dataset>.jsonl (resumable).
"""
import gzip, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from variants import specs, Layer

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    def img(f):
        with gzip.open(f) as h:
            return np.frombuffer(h.read(), np.uint8, offset=16).reshape(-1, 784).astype(np.float64)

    def lab(f):
        with gzip.open(f) as h:
            return np.frombuffer(h.read(), np.uint8, offset=8).astype(np.int64)

    d = os.path.join(HERE, "data")
    return (img(f"{d}/{name}-train-images-idx3-ubyte.gz"), lab(f"{d}/{name}-train-labels-idx1-ubyte.gz"),
            img(f"{d}/{name}-t10k-images-idx3-ubyte.gz"), lab(f"{d}/{name}-t10k-labels-idx1-ubyte.gz"))


class Net(nn.Module):
    def __init__(self, spec, d, H, C, seed):
        super().__init__()
        rng = np.random.default_rng(seed)
        aux = np.random.default_rng(50_000 + seed)
        he = lambda o, i: torch.from_numpy(rng.standard_normal((o, i)) * np.sqrt(2.0 / i))
        he_aux = lambda o, i: torch.from_numpy(aux.standard_normal((o, i)) * np.sqrt(2.0 / i))
        self.l1 = Layer(spec, d, H, he, init_aux=he_aux)
        self.l2 = Layer(spec, H, H, he, init_aux=he_aux)
        self.out = nn.Linear(H, C)
        with torch.no_grad():
            self.out.weight.copy_(he(C, H)); self.out.bias.zero_()

    def forward(self, x):
        return self.out(self.l2(self.l1(x)))


def cells():
    out = []
    for name, spec in specs().items():
        if spec.get("heavy"):
            out.append((name + "@22", name, 22))
            out.append((name + "@64", name, 64))
        else:
            out.append((name, name, 64))
    return out


def run(dsname, data, cell, vname, H, seed):
    XTR, YTR, XTE, YTE = data
    rng = np.random.default_rng(1000 + seed)
    idx = np.concatenate([rng.choice(np.where(YTR == c)[0], 1000, replace=False) for c in range(10)])
    Xtr, ytr = XTR[idx], YTR[idx]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = torch.from_numpy((Xtr - mu) / sd); Xte = torch.from_numpy((XTE - mu) / sd)
    ytr = torch.from_numpy(ytr); yte = torch.from_numpy(YTE)
    net = Net(specs()[vname], 784, H, 10, seed)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    brng = np.random.default_rng(seed)
    diverged, n, t0 = 0, len(Xtr), time.time()
    for ep in range(12):
        perm = torch.from_numpy(brng.permutation(n))
        for i in range(0, n, 128):
            b = perm[i:i + 128]
            loss = F.cross_entropy(net(Xtr[b]), ytr[b])
            if not torch.isfinite(loss):
                diverged = 1; break
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
        if diverged:
            break
    with torch.no_grad():
        for l in (net.l1, net.l2):
            l.record = True
        logits = net(Xte)
        acc = float((logits.argmax(1) == yte).double().mean()) if bool(torch.isfinite(logits).all()) else float("nan")
        if not np.isfinite(acc):
            diverged = 1
    info = {}
    for nm, l in (("l1", net.l1), ("l2", net.l2)):
        if l.kind == "learn":
            info[nm] = l.dials()
        elif l.last_ab is not None:
            info[nm] = l.last_ab
    return dict(dataset=dsname, cell=cell, variant=vname, H=H, seed=seed, acc=acc, diverged=diverged,
                params=nparam, dials=info, sec=round(time.time() - t0, 1))


if __name__ == "__main__":
    dsname, worker, nworkers, s0, s1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    tag = sys.argv[6] if len(sys.argv) > 6 else ""
    only = sys.argv[7].split(";") if len(sys.argv) > 7 else None   # ";" because variant names contain commas
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"cls_{dsname}{tag}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line); done.add((r["cell"], r["seed"]))
    data = load(dsname)
    todo = [(c, s) for c in cells() for s in range(s0, s1) if (only is None or c[0] in only)]
    todo = [t for k, t in enumerate(todo) if k % nworkers == worker and (t[0][0], t[1]) not in done]
    for (cell, vname, H), seed in todo:
        r = run(dsname, data, cell, vname, H, seed)
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"{dsname} {cell:22s} seed {seed}  acc {r['acc']*100:6.2f}  div {r['diverged']}  params {r['params']}  {r['sec']}s", flush=True)
    print("WORKER DONE", flush=True)
