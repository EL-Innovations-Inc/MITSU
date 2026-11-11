"""Sanity gate — must pass before any training run (prereg7, section 3).

Autograd replaces the hand-derived backward used elsewhere in the repo, so the
gate checks the things autograd cannot guarantee:
  G1  dial(a=b=1) equals the closed-form MITSU reference
  G2  log variants equal log of the plain formula
  G3  the torch tri block equals the repo's numpy tri_fwd formula
  G4  every dial form rests at 0 for a silent input
  G5  autograd matches finite differences for every variant (float64)
  G6  outputs and gradients are finite at extreme inputs
"""
import sys
import numpy as np
import torch
from variants import dial, tri, mitsu_ref, specs, Layer, FIXED, ECAP, ETA

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
ok = True


def report(name, val, tol):
    global ok
    good = bool(val < tol)
    ok &= good
    print(f"{name:58s} {val:10.3e}  (< {tol:g})  {'PASS' if good else 'FAIL'}")


u = torch.linspace(-12, 12, 4801)

# G1
report("G1 dial(1,1) vs MITSU closed form, max abs err", (dial(u, 1.0, 1.0) - mitsu_ref(u)).abs().max().item(), 1e-10)

# G2
worst = 0.0
for a, b in [(1.0, 1.0)] + FIXED:
    s = torch.nn.functional.softplus(u)
    f = (s ** (a + b) + a ** (s + b) + b ** (s + a)) / (s + a + b)
    rest = (a ** b + b ** a) / (a + b)
    worst = max(worst, (dial(u, a, b, True) - torch.log(f / rest)).abs().max().item())
    worst = max(worst, (dial(u, a, b, False) - (f - rest)).abs().max().item())
report("G2 dial / log-dial vs direct formula, max abs err", worst, 1e-9)

# G3 — numpy transcription of the repo's tri_fwd
rng = np.random.default_rng(0)
Un = rng.standard_normal((64, 7, 3)) * 2.5
sg = 1 / (1 + np.exp(-Un)); sn = np.maximum(2 * sg, 1e-9)
Sn = sn.sum(-1, keepdims=True); en = np.minimum(Sn - sn, ECAP); pw = sn ** en
spn = np.log1p(np.exp(Un)); mn = (spn * (spn - 1) / (spn + 2)).sum(-1)
ref = (mn + pw.sum(-1) - 3.0) / (Sn[..., 0] + ETA)
report("G3 torch tri vs repo tri_fwd formula, max abs err", np.abs(tri(torch.from_numpy(Un)).numpy() - ref).max(), 1e-10)
reflog = np.log(pw.sum(-1) / (Sn[..., 0] + ETA)) - np.log(3 / (3 + ETA))
report("G3 torch log-tri vs direct formula, max abs err", np.abs(tri(torch.from_numpy(Un), True).numpy() - reflog).max(), 1e-10)

# G4
worst = 0.0
for a, b in [(1.0, 1.0)] + FIXED:
    for lg in (False, True):
        worst = max(worst, abs(dial(torch.tensor([-13.0]), a, b, lg).item()))
report("G4 resting level at u = -13, max |out| over dial forms", worst, 1e-4)
# the repo's tri block rests at 3*MITSU(0)/(3+ETA) = -0.0752 at u = 0, not at 0 ("~0" in its notes)
tri0 = 3 * mitsu_ref(torch.zeros(())).item() / (3 + ETA)
report("G4 tri at u = (0,0,0) vs 3*MITSU(0)/(3+eta), abs err", abs(tri(torch.zeros(1, 1, 3)).item() - tri0), 1e-12)
report("G4 log-tri at u = (0,0,0), |out|", abs(tri(torch.zeros(1, 1, 3), True).item()), 1e-12)

# G5 — autograd vs finite differences through a whole Layer, all variants
SP = specs()
worst_name, worst = "", 0.0
for name, spec in SP.items():
    if name == "relu":
        continue   # kink at 0; nothing to check
    g = torch.Generator().manual_seed(1)
    init = lambda o, i: torch.randn(o, i, generator=g) * 0.7
    layer = Layer(spec, 5, 4, init)
    x = torch.randn(6, 5, generator=g) * 1.5
    params = [p for p in layer.parameters()]

    def loss_fn():
        return (layer(x) ** 2).sum()

    L = loss_fn(); grads = torch.autograd.grad(L, params)
    for p, gr in zip(params, grads):
        flat = p.data.view(-1)
        for ix in torch.randint(0, flat.numel(), (4,), generator=g).tolist():
            old = flat[ix].item(); eps = 1e-6
            flat[ix] = old + eps; lp = loss_fn().item()
            flat[ix] = old - eps; lm = loss_fn().item()
            flat[ix] = old
            num = (lp - lm) / (2 * eps); ana = gr.view(-1)[ix].item()
            rel = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            if rel > worst:
                worst, worst_name = rel, name
report(f"G5 autograd vs finite diff, worst rel err ({worst_name})", worst, 1e-5)

# G6 — extremes, in float32 (the precision the transformer runs in)
bad = []
for name, spec in SP.items():
    g = torch.Generator().manual_seed(2)
    layer = Layer(spec, 3, 4, lambda o, i: torch.randn(o, i, generator=g)).float()
    for scale in (0.0, 1.0, 30.0, 300.0):
        x = (torch.randn(8, 3, generator=g) * scale).float()
        out = layer(x)
        gr = torch.autograd.grad(out.sum(), list(layer.parameters()), allow_unused=True)
        fin = bool(torch.isfinite(out).all()) and all(bool(torch.isfinite(t).all()) for t in gr if t is not None)
        if not fin:
            bad.append(f"{name}@{scale:g}")
print("G6 non-finite at extreme inputs (float32):", bad if bad else "none")
# Dials above 1 are EXPECTED to be able to overflow (that is the hypothesis
# under test in prereg7 P2); the gate only requires everything else be finite.
unexpected = [b for b in bad if not any(k in b for k in ("(1,1.25)", "(1.5,1.5)", "learn02"))]
ok &= (len(unexpected) == 0)
print("   unexpected:", unexpected if unexpected else "none", " ->", "PASS" if not unexpected else "FAIL")

print("\nGATE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
