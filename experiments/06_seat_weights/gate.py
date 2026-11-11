"""Sanity gate — must pass before any training run (prereg8, section 3).

  G1  c = 2 equals the closed-form MITSU reference
  G2  s(s-1)/(s+c) equals the centred weighted mother function for several (i, j, k)
  G3  every form rests at 0 for a silent input
  G4  autograd matches finite differences for every variant (float64)
  G5  outputs and gradients are finite at extreme inputs (float32)
  G6  learnc starts at exactly c = 2 and matches `mitsu` at initialisation
"""
import sys
import torch
import torch.nn.functional as F
from variants import mitsu_c, weighted_mother, specs, Layer, C_VALUES

torch.set_default_dtype(torch.float64)
ok = True


def report(name, val, tol):
    global ok
    good = bool(val < tol); ok &= good
    print(f"{name:62s} {val:10.3e}  (< {tol:g})  {'PASS' if good else 'FAIL'}")


u = torch.linspace(-12, 12, 4801)
s = F.softplus(u)
report("G1 c=2 vs MITSU closed form s(s-1)/(s+2), max abs err", (mitsu_c(u, 2.0) - s * (s - 1) / (s + 2)).abs().max().item(), 1e-12)

worst = 0.0
for i, j, k in [(1, 1, 1), (2, 1, 1), (4, 1, 1), (8, 1, 1), (1, 2, 2), (1, 4, 4), (1, 3, 1), (3, 0.5, 2.5), (0.7, 0.2, 1.9)]:
    worst = max(worst, (weighted_mother(u, i, j, k) - mitsu_c(u, (j + k) / i)).abs().max().item())
report("G2 weighted mother function vs s(s-1)/(s+c), c=(j+k)/i", worst, 1e-12)

worst = max(abs(mitsu_c(torch.tensor([-30.0]), c).item()) for c in [2.0] + C_VALUES)
report("G3 resting level at u = -30, max |out|", worst, 1e-10)

SP = specs()
worst_name, worst = "", 0.0
for name, spec in SP.items():
    g = torch.Generator().manual_seed(1)
    layer = Layer(spec, 5, 4, lambda o, i: torch.randn(o, i, generator=g) * 0.7)
    x = torch.randn(6, 5, generator=g) * 1.5
    params = list(layer.parameters())
    loss_fn = lambda: (layer(x) ** 2).sum()
    grads = torch.autograd.grad(loss_fn(), params)
    for p, gr in zip(params, grads):
        flat, gflat = p.data.view(-1), gr.reshape(-1)
        for ix in torch.randint(0, flat.numel(), (min(4, flat.numel()),), generator=g).tolist():
            old = flat[ix].item(); eps = 1e-6
            flat[ix] = old + eps; lp = loss_fn().item()
            flat[ix] = old - eps; lm = loss_fn().item()
            flat[ix] = old
            num, ana = (lp - lm) / (2 * eps), gflat[ix].item()
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            if rel > worst:
                worst, worst_name = rel, name
report(f"G4 autograd vs finite diff, worst rel err ({worst_name})", worst, 1e-5)

bad = []
for name, spec in SP.items():
    g = torch.Generator().manual_seed(2)
    layer = Layer(spec, 3, 4, lambda o, i: torch.randn(o, i, generator=g)).float()
    for scale in (0.0, 1.0, 30.0, 300.0, 1e6):
        x = (torch.randn(8, 3, generator=g) * scale).float()
        out = layer(x)
        gr = torch.autograd.grad(out.sum(), list(layer.parameters()), allow_unused=True)
        if not (bool(torch.isfinite(out).all()) and all(bool(torch.isfinite(t).all()) for t in gr if t is not None)):
            bad.append(f"{name}@{scale:g}")
print("G5 non-finite at extreme inputs (float32):", bad if bad else "none", " ->", "PASS" if not bad else "FAIL")
ok &= not bad

g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
la = Layer(SP["mitsu"], 5, 4, lambda o, i: torch.randn(o, i, generator=g1))
lb = Layer(SP["learnc"], 5, 4, lambda o, i: torch.randn(o, i, generator=g2))
x = torch.randn(6, 5)
report("G6 learnc initial c vs 2", abs(lb.dials()[0] - 2.0), 1e-12)
report("G6 learnc vs mitsu output at initialisation, max abs err", (la(x) - lb(x)).abs().max().item(), 1e-12)

print("\nGATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
