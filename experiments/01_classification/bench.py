"""Classification benchmark: MITSU vs ReLU/GELU on sklearn's bundled datasets.

Pre-registration: preregistration.md (run 1), extended by width.py (prereg2)
and big.py (prereg3). This file is the shared engine: models, the tri-block
math, the optimizer, and the run-1 experiment grid under __main__.

Models
------
  Linear  softmax regression baseline (the "how nonlinear is this dataset" floor)
  MLP     2 hidden layers, activation in {'relu', 'gelu', 'mitsu'}
  Tri     the "three-seat" MITSU competition block: a stem, then L blocks in
          which each output unit is computed from a TRIPLE of pre-activations
          competing through the mother function's cross-exponents (details at
          tri_fwd below).

Everything is hand-derived numpy with a finite-difference gradcheck gate
(gradcheck() below); the pre-registered rule is that no experiment runs unless
the worst relative error is < 1e-5. Reproduce run 1 with:  python3 bench.py
"""
import numpy as np, json, sys
from scipy.special import erf
from sklearn.datasets import load_digits, load_breast_cancer, load_wine, load_iris
from sklearn.model_selection import train_test_split

# Tri-block constants (frozen in prereg1):
#   ECAP caps competition exponents so co-active strong channels cannot go
#   double-exponential; ETA regularizes the divisive denominator at silence.
ECAP, ETA = 6.0, 0.15


# ---------------------------------------------------------------------------
# Activations (elementwise; see mitsu.py at repo root for provenance)
# ---------------------------------------------------------------------------
def softplus(u): return np.where(u > 30, u, np.log1p(np.exp(np.minimum(u, 30))))
def sigm(u): return 1/(1+np.exp(-np.clip(u, -500, 500)))
def mitsu(u):
    """Centered MITSU: s(s-1)/(s+2), s = softplus(u)."""
    s = softplus(u); return s*(s-1)/(s+2)
def mitsu_d(u):
    """dMITSU/du = (s^2+4s-2)/(s+2)^2 * sigmoid(u) — bounded in (-0.5, 1)."""
    s = softplus(u); return (s*s+4*s-2)/((s+2)**2)*sigm(u)
def gelu(u):
    """Exact GELU via erf (the reference form, not the tanh approximation)."""
    return 0.5*u*(1+erf(u/np.sqrt(2)))
def gelu_d(u):
    """dGELU/du = Phi(u) + u*phi(u)."""
    return 0.5*(1+erf(u/np.sqrt(2))) + u*np.exp(-u*u/2)/np.sqrt(2*np.pi)


# ---------------------------------------------------------------------------
# Tri block: U (B, N, 3) -> out (B, N)
#
# Per output unit, three pre-activations (u0, u1, u2) fill the mother
# function's three seats. With bounded bases s_c = 2*sigmoid(u_c) and
# S = s0+s1+s2, each channel contributes s_c^min(S - s_c, ECAP): a channel
# whose base is below 1 is SUPPRESSED as its neighbours light up (its exponent
# grows), while a base above 1 is amplified — soft winner-take-all with the
# threshold exactly at s = 1, i.e. u = 0. The unit's output is
#
#   out = [ sum_c MITSU(u_c)  +  sum_c s_c^e_c  - 3 ] / (S + ETA)
#
# The "-3" is the resting level of the cross-power sum (at u = 0 all three
# powers equal 1), so silence maps to ~0; division by S + ETA is divisive
# normalization. Bases were originally softplus, which detonated deep towers
# exactly as the double-exponential analysis predicted; 2*sigmoid keeps the
# same threshold with nowhere to run (validated fix, gradcheck ~1e-9).
# ---------------------------------------------------------------------------
def tri_fwd(U):
    sg = sigm(U); s = np.maximum(2*sg, 1e-9)         # bounded bases in (0, 2)
    S = s.sum(-1, keepdims=True)                     # (B, N, 1) total drive
    e = np.minimum(S - s, ECAP)                      # each seat's exponent
    pw = s**e                                        # competition terms
    m = mitsu(U).sum(-1)                             # cooperative MITSU term
    out = (m + pw.sum(-1) - 3.0)/(S[..., 0] + ETA)
    return out, (U, sg, s, S, e, pw)

def tri_bwd(dOut, cache):
    """Analytic gradient of tri_fwd w.r.t. U. Key subtlety: the exponent
    e_c = min(S - s_c, ECAP) does NOT depend on s_c itself (S - s_c is the sum
    of the OTHERS), so d(s_c^e_c)/ds_c = e_c * s_c^(e_c - 1) unconditionally,
    while each cross term s_d^e_d (d != c) contributes ln(s_d) * s_d^e_d
    through its exponent — unless capped, where that path is dead."""
    U, sg, s, S, e, pw = cache
    Sd = S[..., 0] + ETA
    T3 = mitsu(U).sum(-1) + pw.sum(-1) - 3.0         # the numerator
    capped = (S - s) >= ECAP
    logs_pw = np.where(capped, 0.0, np.log(s)*pw)    # cross-derivative factors
    cross = logs_pw.sum(-1, keepdims=True) - logs_pw # sum over d != c
    dPds = e*s**(e-1) + cross                        # d(sum powers)/d s_c
    dsdu = 2*sg*(1-sg)                               # d s_c / d u_c
    # Quotient rule: numerator' * Sd - numerator * Sd' (Sd' = dsdu), over Sd^2.
    dU = dOut[..., None]*(((mitsu_d(U) + dsdu*dPds)*Sd[..., None] - T3[..., None]*dsdu)/(Sd*Sd)[..., None])
    return np.clip(dU, -1e4, 1e4)                    # belt-and-braces guard


# ---------------------------------------------------------------------------
# Models. All linear layers use He init (std = sqrt(2/fan_in)); every model
# stores weights in self.W / biases in self.b so the Adam loop is generic.
# ---------------------------------------------------------------------------
def he(rng, o, i): return rng.standard_normal((o, i))*np.sqrt(2.0/i)

class MLP:
    """input -> H -> H -> C with a chosen elementwise activation."""
    def __init__(self, d, H, C, act, rng):
        self.act = act
        self.W = [he(rng, H, d), he(rng, H, H), he(rng, C, H)]
        self.b = [np.zeros(H), np.zeros(H), np.zeros(C)]
    def forward(self, X):
        f = {'relu': (lambda u: np.maximum(u, 0), lambda u: (u > 0).astype(float)),
             'gelu': (gelu, gelu_d), 'mitsu': (mitsu, mitsu_d)}[self.act]
        u1 = X@self.W[0].T+self.b[0]; h1 = f[0](u1)
        u2 = h1@self.W[1].T+self.b[1]; h2 = f[0](u2)
        z  = h2@self.W[2].T+self.b[2]
        return z, (X, u1, h1, u2, h2, f)
    def backward(self, dz, c):
        X, u1, h1, u2, h2, f = c
        g = [None]*3; gb = [None]*3
        g[2] = dz.T@h2; gb[2] = dz.sum(0)
        dh2 = dz@self.W[2]; du2 = dh2*f[1](u2)
        g[1] = du2.T@h1; gb[1] = du2.sum(0)
        dh1 = du2@self.W[1]; du1 = dh1*f[1](u1)
        g[0] = du1.T@X; gb[0] = du1.sum(0)
        return g, gb

class Linear:
    """Softmax regression: the linear floor every dataset gets measured against."""
    def __init__(self, d, C, rng):
        self.W = [he(rng, C, d)]; self.b = [np.zeros(C)]
    def forward(self, X): return X@self.W[0].T+self.b[0], (X,)
    def backward(self, dz, c):
        return [dz.T@c[0]], [dz.sum(0)]

class Tri:
    """MITSU stem, then L tri blocks. Each block's dense layer produces 3*Wd
    pre-activations, reshaped to (B, Wd, 3) so every output unit owns a triple."""
    def __init__(self, d, Wd, L, C, rng):
        self.L = L; self.Wd = Wd
        self.W = [he(rng, Wd, d)] + [he(rng, 3*Wd, Wd) for _ in range(L)] + [he(rng, C, Wd)]
        self.b = [np.zeros(Wd)] + [np.zeros(3*Wd) for _ in range(L)] + [np.zeros(C)]
    def forward(self, X):
        u0 = X@self.W[0].T+self.b[0]; h = mitsu(u0)
        caches = []; hs = [h]
        for l in range(1, self.L+1):
            U = (h@self.W[l].T+self.b[l]).reshape(-1, self.Wd, 3)
            h, cache = tri_fwd(U); caches.append(cache); hs.append(h)
        z = h@self.W[-1].T+self.b[-1]
        return z, (X, u0, hs, caches)
    def backward(self, dz, c):
        X, u0, hs, caches = c
        g = [None]*len(self.W); gb = [None]*len(self.b)
        g[-1] = dz.T@hs[-1]; gb[-1] = dz.sum(0)
        dh = dz@self.W[-1]
        for l in range(self.L, 0, -1):
            dU = tri_bwd(dh, caches[l-1]).reshape(len(X), 3*self.Wd)
            g[l] = dU.T@hs[l-1]; gb[l] = dU.sum(0)
            dh = dU@self.W[l]
        du0 = dh*mitsu_d(u0)
        g[0] = du0.T@X; gb[0] = du0.sum(0)
        return g, gb

def nparams(m): return sum(w.size for w in m.W)+sum(b.size for b in m.b)


def softmax_ce(z, y):
    """Stable softmax cross-entropy. Returns (mean loss, dL/dz, probs);
    dz already includes the 1/batch factor so backward() sums are means."""
    z = z - z.max(1, keepdims=True); ez = np.exp(z); p = ez/ez.sum(1, keepdims=True)
    loss = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()
    dz = p.copy(); dz[np.arange(len(y)), y] -= 1; dz /= len(y)
    return loss, dz, p


def gradcheck():
    """Pre-registered gate: central finite differences (eps = 1e-6) against the
    analytic gradients, 6 random entries per tensor, for a MITSU MLP and a
    2-block Tri tower. Returns the worst relative error; must be < 1e-5."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for build in [lambda: MLP(5, 4, 3, 'mitsu', rng), lambda: Tri(5, 4, 2, 3, rng)]:
        m = build(); X = rng.standard_normal((7, 5))*0.8; y = rng.integers(0, 3, 7)
        z, c = m.forward(X); loss, dz, _ = softmax_ce(z, y)
        g, gb = m.backward(dz, c)
        for arrs, grads in [(m.W, g), (m.b, gb)]:
            for k in range(len(arrs)):
                idx = [tuple(rng.integers(0, s) for s in arrs[k].shape) for _ in range(6)]
                for ix in idx:
                    eps = 1e-6; old = arrs[k][ix]
                    arrs[k][ix] = old+eps; lp, _, _ = softmax_ce(m.forward(X)[0], y)
                    arrs[k][ix] = old-eps; lm, _, _ = softmax_ce(m.forward(X)[0], y)
                    arrs[k][ix] = old
                    num = (lp-lm)/(2*eps); ana = grads[k][ix]
                    rel = abs(num-ana)/max(1e-8, abs(num)+abs(ana))
                    worst = max(worst, rel)
    return worst


def train_eval(model, Xtr, ytr, Xte, yte, epochs, lr=3e-3, bs=32, seed=0):
    """Adam (0.9/0.999, bias-corrected) with GLOBAL grad-norm clip 5 applied
    identically to every model — fairness of the comparison depends on all
    families sharing one optimizer recipe. Returns (test accuracy, diverged)."""
    rng = np.random.default_rng(seed+777)            # shuffle stream, decoupled from init
    mW = [np.zeros_like(w) for w in model.W]; vW = [np.zeros_like(w) for w in model.W]
    mB = [np.zeros_like(b) for b in model.b]; vB = [np.zeros_like(b) for b in model.b]
    t = 0; diverged = False
    for ep in range(epochs):
        idx = rng.permutation(len(Xtr))
        for i in range(0, len(Xtr), bs):
            j = idx[i:i+bs]
            z, c = model.forward(Xtr[j])
            loss, dz, _ = softmax_ce(z, ytr[j])
            if not np.isfinite(loss): diverged = True; break
            g, gb = model.backward(dz, c)
            gn = np.sqrt(sum((x*x).sum() for x in g)+sum((x*x).sum() for x in gb))
            sc = min(1.0, 5.0/max(gn, 1e-12))        # global-norm clip
            t += 1; b1c = 1-0.9**t; b2c = 1-0.999**t # Adam bias corrections
            for k in range(len(model.W)):
                for buf, val, mm, vv in [(model.W[k], g[k]*sc, mW, vW), (model.b[k], gb[k]*sc, mB, vB)]:
                    mm[k] = 0.9*mm[k]+0.1*val; vv[k] = 0.999*vv[k]+0.001*val*val
                    buf -= lr*(mm[k]/b1c)/(np.sqrt(vv[k]/b2c)+1e-8)
        if diverged: break
    z, _ = model.forward(Xte)
    acc = float((z.argmax(1) == yte).mean())
    return acc, diverged


def mlp_width(d, C, target):
    """Smallest H with (d+1)H + (H+1)H + (H+1)C >= target params, i.e. the MLP
    width that slightly EXCEEDS the tri tower's budget (conservative: any tri
    win then cannot be a parameter artifact). Solves the quadratic directly."""
    a, b, c = 1, d+2+C, C-target
    return int(np.ceil((-b+np.sqrt(b*b-4*a*c))/2))


if __name__ == '__main__':
    print('bench.py takes no arguments: running the full pre-registered run-1 grid')
    print('(4 datasets x 5 models x 5 seeds; several minutes on CPU; writes results.json)')
    # Gate first — the pre-registered rule is no gradcheck pass, no experiment.
    g = gradcheck()
    print(f'GRADCHECK worst rel err: {g:.2e}')
    if g > 1e-5: sys.exit('GATE FAILED — no experiment.')
    sets = {'digits': load_digits(), 'cancer': load_breast_cancer(),
            'wine': load_wine(), 'iris': load_iris()}
    results = {}
    for name, ds in sets.items():
        X, y = ds.data.astype(float), ds.target
        d, C = X.shape[1], len(np.unique(y))
        epochs = 60 if name == 'digits' else 80
        Wd, L = 16, 2                                 # tri tower shape (frozen)
        tri_p = (d+1)*Wd + L*(Wd+1)*3*Wd + (Wd+1)*C   # tri parameter budget
        H = mlp_width(d, C, tri_p)                    # param-matched MLP width
        results[name] = {}
        for model_name in ['linear', 'relu', 'gelu', 'mitsu', 'tri']:
            accs = []; divs = 0
            for seed in range(5):
                # Seed controls split AND init: fully paired across models.
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
                mu, sd = Xtr.mean(0), Xtr.std(0)+1e-8
                Xtr, Xte = (Xtr-mu)/sd, (Xte-mu)/sd
                rng = np.random.default_rng(seed)
                m = (Linear(d, C, rng) if model_name == 'linear' else
                     Tri(d, Wd, L, C, rng) if model_name == 'tri' else
                     MLP(d, H, C, model_name, rng))
                acc, div = train_eval(m, Xtr, ytr, Xte, yte, epochs, seed=seed)
                accs.append(acc); divs += div
                if seed == 0: pcount = nparams(m)
            results[name][model_name] = dict(mean=float(np.mean(accs)), std=float(np.std(accs)),
                                             params=pcount, diverged=divs)
            print(f'{name:7s} {model_name:6s} params={pcount:5d}  acc={np.mean(accs)*100:5.2f} ± {np.std(accs)*100:4.2f}  div={divs}')
    json.dump(results, open('results.json', 'w'), indent=1)
