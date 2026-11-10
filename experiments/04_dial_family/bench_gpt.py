"""Character-GPT bench — Tiny Shakespeare, config inherited from the repo's
prereg4 / mitsu_gpt.py:

  context 64, width 64, 2 heads, 2 pre-norm blocks, learned positions,
  untied head, no biases, init std 0.06 (output projections / sqrt(2L)),
  Adam lr 2e-3 with 100-step warmup, global clip 1.0, batch 32, 2000 steps,
  train 90% / val 10%, float32.

Twins: for a given seed every variant shares the same batch schedule, and
every non-FFN weight starts identical. One-pre-activation variants also share
identical FFN weights. The ONLY difference is the feed-forward nonlinearity.

FFN: out(act(fc(x))). Light variants use hidden 256 (4x). Heavy variants
(three pre-activations per unit: inunit, tri) use hidden 127, which gives
slightly FEWER parameters than the 4x baseline.

Difference from the repo run: validation loss is measured on the WHOLE
validation split (non-overlapping windows), not on 8 sampled batches.

Usage:  python3 bench_gpt.py <worker> <nworkers> <seed0> <seed1> [steps]
Appends one JSON line per run to results/gpt.jsonl (resumable).
"""
import json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from variants import specs, Layer

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
T, D, NH, NL, STD = 64, 64, 2, 2, 0.06


class Block(nn.Module):
    def __init__(self, spec, gA, gB, gAux):
        super().__init__()
        self.ln1 = nn.LayerNorm(D, eps=1e-5); self.ln2 = nn.LayerNorm(D, eps=1e-5)
        self.qkv = nn.Linear(D, 3 * D, bias=False); self.proj = nn.Linear(D, D, bias=False)
        with torch.no_grad():
            self.qkv.weight.copy_(torch.randn(3 * D, D, generator=gA) * STD)
            self.proj.weight.copy_(torch.randn(D, D, generator=gA) * STD / math.sqrt(2 * NL))
        m = 127 if spec.get("heavy") else 4 * D
        self.ffn = Layer(spec, D, m, lambda o, i: torch.randn(o, i, generator=gB) * STD, bias=False,
                         init_aux=lambda o, i: torch.randn(o, i, generator=gAux) * STD)
        self.out = nn.Linear(m, D, bias=False)
        with torch.no_grad():
            self.out.weight.copy_(torch.randn(D, m, generator=gB) * STD / math.sqrt(2 * NL))

    def forward(self, x, mask):
        B, Tt, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, Tt, 3, NH, D // NH).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // NH) + mask[:Tt, :Tt]
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, Tt, D)
        x = x + self.proj(o)
        return x + self.out(self.ffn(self.ln2(x)))


class GPT(nn.Module):
    def __init__(self, spec, V, seed):
        super().__init__()
        gA = torch.Generator().manual_seed(seed)             # everything outside the FFN
        gB = torch.Generator().manual_seed(100_000 + seed)   # FFN signal weights
        gAux = torch.Generator().manual_seed(200_000 + seed) # dial projections
        self.tok = nn.Parameter(torch.randn(V, D, generator=gA) * STD)
        self.pos = nn.Parameter(torch.randn(T, D, generator=gA) * STD)
        self.head = nn.Linear(D, V, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(V, D, generator=gA) * STD)
        self.blocks = nn.ModuleList([Block(spec, gA, gB, gAux) for _ in range(NL)])
        self.lnf = nn.LayerNorm(D, eps=1e-5)
        self.register_buffer("mask", torch.triu(torch.full((T, T), float("-inf")), 1))

    def forward(self, idx):
        x = self.tok[idx] + self.pos[: idx.shape[1]]
        for b in self.blocks:
            x = b(x, self.mask)
        return self.head(self.lnf(x))


def data():
    txt = open(os.path.join(HERE, "data", "shakespeare.txt")).read()
    chars = sorted(set(txt)); stoi = {c: i for i, c in enumerate(chars)}
    tok = torch.tensor([stoi[c] for c in txt], dtype=torch.long)
    n = int(0.9 * len(tok))
    return tok[:n], tok[n:], len(chars)


@torch.no_grad()
def val_loss(net, va):
    nwin = (len(va) - 1) // T
    X = va[: nwin * T].view(nwin, T); Y = va[1: nwin * T + 1].view(nwin, T)
    tot = 0.0
    for i in range(0, nwin, 256):
        lg = net(X[i:i + 256])
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), Y[i:i + 256].reshape(-1), reduction="sum").item()
    return tot / (nwin * T)


def run(vname, seed, steps, tr, va, V):
    net = GPT(specs()[vname], V, seed)
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    brng = np.random.default_rng(9000 + seed)
    diverged, gmax, t0, hist, val_mid = 0, 0.0, time.time(), [], None
    for step in range(1, steps + 1):
        off = torch.from_numpy(brng.integers(0, len(tr) - T - 1, 32))
        ix = off[:, None] + torch.arange(T)[None, :]
        x, y = tr[ix], tr[ix + 1]
        lg = net(x)
        loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        if not torch.isfinite(loss):
            diverged = 1; break
        for g in opt.param_groups:
            g["lr"] = 2e-3 * min(1.0, step / 100)
        opt.zero_grad(); loss.backward()
        gn = float(nn.utils.clip_grad_norm_(net.parameters(), 1.0))
        if not math.isfinite(gn):
            diverged = 1; break
        gmax = max(gmax, gn)
        opt.step()
        if step % 50 == 0:
            hist.append(round(loss.item(), 4))
        if step == steps // 2:
            val_mid = val_loss(net, va)
    vl = val_loss(net, va) if not diverged else float("nan")
    if not math.isfinite(vl):
        diverged = 1
    info = {}
    with torch.no_grad():
        for b in net.blocks:
            b.ffn.record = True
        if not diverged:
            net(va[: 64 * T].view(64, T))
        for i, b in enumerate(net.blocks):
            if b.ffn.kind == "learn":
                info[f"l{i+1}"] = b.ffn.dials()
            elif b.ffn.last_ab is not None:
                info[f"l{i+1}"] = b.ffn.last_ab
    return dict(variant=vname, seed=seed, steps=steps, val=vl, val_mid=val_mid, diverged=diverged, params=nparam,
                gradnorm_max=round(gmax, 3), train_hist=hist, dials=info, sec=round(time.time() - t0, 1))


if __name__ == "__main__":
    worker, nworkers, s0, s1 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
    tag = sys.argv[6] if len(sys.argv) > 6 else ""
    only = sys.argv[7].split(";") if len(sys.argv) > 7 else None   # ";" because variant names contain commas
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"gpt{tag}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line); done.add((r["variant"], r["seed"]))
    tr, va, V = data()
    # seeds outermost, so an interrupted queue still holds complete paired seeds
    todo = [(v, s) for s in range(s0, s1) for v in specs() if (only is None or v in only)]
    todo = [t for k, t in enumerate(todo) if k % nworkers == worker and t not in done]
    for vname, seed in todo:
        r = run(vname, seed, steps, tr, va, V)
        with open(path, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"gpt {vname:22s} seed {seed}  val {r['val']:.4f}  div {r['diverged']}  params {r['params']}  gmax {r['gradnorm_max']}  {r['sec']}s", flush=True)
    print("WORKER DONE", flush=True)
