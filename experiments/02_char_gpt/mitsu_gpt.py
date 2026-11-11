"""MITSU-GPT nano — a character-level GPT in pure numpy, hand-derived backprop.

Pre-registration: prereg4.md. The experiment is a TWIN face-off: two identical
transformers (same init seed — the parameter shapes are identical because both
activations are parameter-free — and the same batch schedule) that differ in
exactly one line: the MLP activation, MITSU vs GELU. Any difference in the
loss curves is attributable to the nonlinearity alone.

Architecture (Cfg defaults): context T=64, width d=64, 2 heads, 2 pre-norm
blocks, learned positional embeddings, untied head; ~110k parameters. Trained
on Tiny Shakespeare (get_data.sh), 90/10 split, char-level (vocab 65).

Everything is float32 for speed but the gradcheck runs the same code in
float64; the pre-registered gate for this file was < 1e-4 worst relative error
(measured 6.6e-6 / 2.7e-6). Shipped checkpoints ckpt_mitsu.npz / ckpt_gelu.npz
are the actual weights from the pre-registered run (val 1.7107 vs 1.7062).

Usage:
    python mitsu_gpt.py gradcheck
    python mitsu_gpt.py train  {mitsu|gelu} <steps> <max_seconds>   # resumable
    python mitsu_gpt.py sample {mitsu|gelu} [prompt]
"""
import numpy as np, time, os
HERE = os.path.dirname(os.path.abspath(__file__))   # all files live beside the script


# ---------------------------------------------------------------------------
# Activations. GELU uses the tanh approximation (the GPT-2 convention) so the
# baseline matches what the incumbent architecture actually shipped with.
# ---------------------------------------------------------------------------
def softplus(u): return np.where(u > 30, u, np.log1p(np.exp(np.minimum(u, 30))))
def sigm(u): return 1/(1+np.exp(-np.clip(u, -60, 60)))
def mitsu(u):
    s = softplus(u); return s*(s-1)/(s+2)
def mitsu_d(u):
    s = softplus(u); return (s*s+4*s-2)/((s+2)**2)*sigm(u)
SQ2PI = np.sqrt(2/np.pi)
def gelu(u):  # tanh approximation (matches GPT-2)
    return 0.5*u*(1+np.tanh(SQ2PI*(u+0.044715*u**3)))
def gelu_d(u):
    t = np.tanh(SQ2PI*(u+0.044715*u**3))
    return 0.5*(1+t) + 0.5*u*(1-t*t)*SQ2PI*(1+3*0.044715*u*u)
ACT = {'mitsu': (mitsu, mitsu_d), 'gelu': (gelu, gelu_d)}


class Cfg:
    """Model hyperparameters. hd = per-head dim, m = MLP hidden width (4x)."""
    def __init__(self, V=65, T=64, d=64, H=2, L=2, act='mitsu', dtype=np.float32):
        self.V, self.T, self.d, self.H, self.L, self.act, self.dtype = V, T, d, H, L, act, dtype
        self.hd = d//H; self.m = 4*d


def init_params(cfg, seed=0):
    """GPT-2-style init: N(0, 0.06) everywhere, with the two residual-branch
    output projections (attn proj, MLP out) scaled by 1/sqrt(2L) so the
    residual stream's variance stays O(1) at depth L."""
    rng = np.random.default_rng(seed); dt = cfg.dtype
    P = {}
    std = 0.06
    P['tok'] = (rng.standard_normal((cfg.V, cfg.d))*std).astype(dt)
    P['pos'] = (rng.standard_normal((cfg.T, cfg.d))*std).astype(dt)
    for l in range(cfg.L):
        P[f'ln1g{l}'] = np.ones(cfg.d, dt); P[f'ln1b{l}'] = np.zeros(cfg.d, dt)
        P[f'qkv{l}']  = (rng.standard_normal((cfg.d, 3*cfg.d))*std).astype(dt)
        P[f'proj{l}'] = (rng.standard_normal((cfg.d, cfg.d))*std/np.sqrt(2*cfg.L)).astype(dt)
        P[f'ln2g{l}'] = np.ones(cfg.d, dt); P[f'ln2b{l}'] = np.zeros(cfg.d, dt)
        P[f'fc{l}']   = (rng.standard_normal((cfg.d, cfg.m))*std).astype(dt)
        P[f'out{l}']  = (rng.standard_normal((cfg.m, cfg.d))*std/np.sqrt(2*cfg.L)).astype(dt)
    P['lnfg'] = np.ones(cfg.d, dt); P['lnfb'] = np.zeros(cfg.d, dt)
    P['head'] = (rng.standard_normal((cfg.d, cfg.V))*std).astype(dt)
    return P


# ---------------------------------------------------------------------------
# LayerNorm. ln_b implements the standard derivation: with xh the normalized
# input and dxh = dy * gamma,
#   dx = ( dxh - mean(dxh) - xh * mean(dxh * xh) ) / sigma
# Means are over the feature axis; eps = 1e-5 lives inside inv.
# ---------------------------------------------------------------------------
def ln_f(x, g, b):
    mu = x.mean(-1, keepdims=True); xc = x - mu
    var = (xc*xc).mean(-1, keepdims=True); inv = 1/np.sqrt(var + 1e-5)
    xh = xc*inv
    return g*xh + b, (xh, inv, g)
def ln_b(dy, c):
    xh, inv, g = c
    dg = (dy*xh).sum((0, 1)); db = dy.sum((0, 1))
    dxh = dy*g
    dx = (dxh - dxh.mean(-1, keepdims=True) - xh*(dxh*xh).mean(-1, keepdims=True))*inv
    return dx, dg, db


def forward(P, cfg, idx, targets=None):
    """Forward pass. Shapes for one block:
        x    (B, T, d)    residual stream
        qkv  (B, T, 3d) -> q, k, v each (B, H, T, hd)
        A, S (B, H, T, T) masked attention logits / weights
        o    (B, T, d)    heads merged back
    Returns (logits, mean CE loss or None, cache for backward)."""
    B, T = idx.shape; f = ACT[cfg.act]
    x = P['tok'][idx] + P['pos'][:T]
    cache = {'idx': idx, 'B': B, 'T': T, 'blocks': []}
    mask = np.triu(np.full((T, T), -np.inf, cfg.dtype), 1)   # causal: no peeking
    for l in range(cfg.L):
        h, c1 = ln_f(x, P[f'ln1g{l}'], P[f'ln1b{l}'])
        qkv = h @ P[f'qkv{l}']
        q, k, v = [a.reshape(B, T, cfg.H, cfg.hd).transpose(0, 2, 1, 3) for a in np.split(qkv, 3, -1)]
        A = q @ k.transpose(0, 1, 3, 2)/np.sqrt(cfg.hd) + mask
        A = A - A.max(-1, keepdims=True)                     # softmax stability
        eA = np.exp(A); S = eA/eA.sum(-1, keepdims=True)
        o = (S @ v).transpose(0, 2, 1, 3).reshape(B, T, cfg.d)
        y = o @ P[f'proj{l}']
        x = x + y                                            # attention residual
        h2, c2 = ln_f(x, P[f'ln2g{l}'], P[f'ln2b{l}'])
        u = h2 @ P[f'fc{l}']; a = f[0](u); m = a @ P[f'out{l}']
        x = x + m                                            # MLP residual
        cache['blocks'].append((h, q, k, v, S, o, h2, u, a, c1, c2))
    xf, cf = ln_f(x, P['lnfg'], P['lnfb'])
    logits = xf @ P['head']
    cache['xf'], cache['cf'] = xf, cf
    if targets is None: return logits, None, cache
    # Cross-entropy over all B*T positions (mean); stable log-softmax.
    z = logits - logits.max(-1, keepdims=True)
    ez = np.exp(z); p = ez/ez.sum(-1, keepdims=True)
    loss = -np.log(np.maximum(p[np.arange(B)[:, None], np.arange(T)[None, :], targets], 1e-12)).mean()
    cache['p'] = p; cache['targets'] = targets
    return logits, float(loss), cache


def backward(P, cfg, cache):
    """Exact reverse of forward. Attention backward, for reference:
        dV = S^T dO
        dS = dO V^T
        dA = S * (dS - rowsum(dS * S))         # softmax Jacobian
        dQ = dA K / sqrt(hd),  dK = dA^T Q / sqrt(hd)
    Masked positions carry S = 0, so their gradients vanish automatically —
    no explicit mask handling is needed on the way back."""
    B, T = cache['B'], cache['T']; f = ACT[cfg.act]
    G = {k: np.zeros_like(w) for k, w in P.items()}
    dlog = cache['p'].copy()
    dlog[np.arange(B)[:, None], np.arange(T)[None, :], cache['targets']] -= 1
    dlog /= (B*T)                                            # mean over positions
    G['head'] = cache['xf'].reshape(-1, cfg.d).T @ dlog.reshape(-1, cfg.V)
    dxf = dlog @ P['head'].T
    dx, G['lnfg'], G['lnfb'] = ln_b(dxf, cache['cf'])
    for l in range(cfg.L-1, -1, -1):
        h, q, k, v, S, o, h2, u, a, c1, c2 = cache['blocks'][l]
        # ---- MLP branch (residual: gradient flows both through and around) --
        dm = dx
        G[f'out{l}'] = a.reshape(-1, cfg.m).T @ dm.reshape(-1, cfg.d)
        da = dm @ P[f'out{l}'].T
        du = da*f[1](u)                                      # the only line where
        G[f'fc{l}'] = h2.reshape(-1, cfg.d).T @ du.reshape(-1, cfg.m)  # twins differ
        dh2 = du @ P[f'fc{l}'].T
        dx2, G[f'ln2g{l}'], G[f'ln2b{l}'] = ln_b(dh2, c2)
        dx = dx + dx2
        # ---- attention branch ----------------------------------------------
        dy = dx
        G[f'proj{l}'] = o.reshape(-1, cfg.d).T @ dy.reshape(-1, cfg.d)
        do = (dy @ P[f'proj{l}'].T).reshape(B, T, cfg.H, cfg.hd).transpose(0, 2, 1, 3)
        dS = do @ v.transpose(0, 1, 3, 2)
        dv = S.transpose(0, 1, 3, 2) @ do
        dA = S*(dS - (dS*S).sum(-1, keepdims=True))
        dq = dA @ k/np.sqrt(cfg.hd)
        dk = dA.transpose(0, 1, 3, 2) @ q/np.sqrt(cfg.hd)
        dqkv = np.concatenate([g.transpose(0, 2, 1, 3).reshape(B, T, cfg.d) for g in (dq, dk, dv)], -1)
        G[f'qkv{l}'] = h.reshape(-1, cfg.d).T @ dqkv.reshape(-1, 3*cfg.d)
        dh = dqkv @ P[f'qkv{l}'].T
        dx1, G[f'ln1g{l}'], G[f'ln1b{l}'] = ln_b(dh, c1)
        dx = dx + dx1
    G['pos'][:T] += dx.sum(0)                                # only first T rows used
    np.add.at(G['tok'], cache['idx'], dx)                    # scatter-add embeddings
    return G


def gradcheck(act='mitsu'):
    """Gate: float64 tiny config (V=11, T=5, d=8, 1 block), central differences
    eps = 1e-6, 6 random entries per tensor. Pre-registered threshold < 1e-4."""
    cfg = Cfg(V=11, T=5, d=8, H=2, L=1, act=act, dtype=np.float64)
    P = init_params(cfg, seed=3)
    rng = np.random.default_rng(4)
    idx = rng.integers(0, cfg.V, (2, cfg.T)); tgt = rng.integers(0, cfg.V, (2, cfg.T))
    _, loss, cache = forward(P, cfg, idx, tgt)
    G = backward(P, cfg, cache)
    worst = 0.0
    for name, w in P.items():
        for _ in range(6):
            ix = tuple(rng.integers(0, s) for s in w.shape)
            eps = 1e-6; old = w[ix]
            w[ix] = old+eps; lp = forward(P, cfg, idx, tgt)[1]
            w[ix] = old-eps; lm = forward(P, cfg, idx, tgt)[1]
            w[ix] = old
            num = (lp-lm)/(2*eps)
            rel = abs(num-G[name][ix])/max(1e-8, abs(num)+abs(G[name][ix]))
            worst = max(worst, rel)
    return worst


def get_batch(tokens, cfg, B, rng):
    """B random windows: x = tokens[i:i+T], y = the same window shifted by 1."""
    ix = rng.integers(0, len(tokens)-cfg.T-1, B)
    x = np.stack([tokens[i:i+cfg.T] for i in ix])
    y = np.stack([tokens[i+1:i+cfg.T+1] for i in ix])
    return x, y


def train(act, steps=2000, B=32, lr=2e-3, seed=0, max_sec=1200, tag=None):
    """Chunked, resumable training. The checkpoint stores params + Adam state +
    step + history; on resume the batch-schedule RNG is REPLAYED to the saved
    step, so a resumed run and an uninterrupted run see identical batches —
    this is what keeps the twins (and any restart) exactly paired."""
    tag = tag or act
    text = open(os.path.join(HERE, 'shakespeare.txt')).read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    tokens = np.array([stoi[c] for c in text], np.int64)
    n = int(0.9*len(tokens)); tr, va = tokens[:n], tokens[n:]
    cfg = Cfg(act=act)
    ck = os.path.join(HERE, f'ckpt_{tag}.npz')
    if os.path.exists(ck):
        z = np.load(ck, allow_pickle=True)
        P = {k: z[k] for k in z.files if k not in ('m', 'v', 'step', 'hist')}
        M = z['m'].item(); V = z['v'].item(); step0 = int(z['step']); hist = list(z['hist'])
    else:
        P = init_params(cfg, seed=seed)
        M = {k: np.zeros_like(w) for k, w in P.items()}
        V = {k: np.zeros_like(w) for k, w in P.items()}
        step0 = 0; hist = []
    brng = np.random.default_rng(9000+seed)  # batch schedule rng — same for twins
    for _ in range(step0):                   # replay schedule so resumes stay aligned
        brng.integers(0, len(tr)-cfg.T-1, B)
    vrng = np.random.default_rng(123)
    vbat = [get_batch(va, cfg, B, vrng) for _ in range(8)]   # fixed val batches
    t0 = time.time()
    for step in range(step0+1, steps+1):
        x, y = get_batch(tr, cfg, B, brng)
        _, loss, cache = forward(P, cfg, x, y)
        if not np.isfinite(loss): print('DIVERGED at', step); break
        G = backward(P, cfg, cache)
        gn = np.sqrt(sum(float((g*g).sum()) for g in G.values()))
        sc = min(1.0, 1.0/max(gn, 1e-12))    # global-norm clip 1.0 (transformer-tight)
        lr_t = lr*min(1.0, step/100)         # 100-step linear warmup
        b1c = 1-0.9**step; b2c = 1-0.999**step
        for kk in P:
            g = G[kk]*sc
            M[kk] = 0.9*M[kk]+0.1*g; V[kk] = 0.999*V[kk]+0.001*g*g
            P[kk] -= (lr_t*(M[kk]/b1c)/(np.sqrt(V[kk]/b2c)+1e-8)).astype(P[kk].dtype)
        if step % 100 == 0 or step == steps:
            vl = np.mean([forward(P, cfg, vx, vy)[1] for vx, vy in vbat])
            hist.append((step, float(loss), float(vl)))
            print(f'{tag} step {step:4d}  train {loss:.4f}  val {vl:.4f}  ({time.time()-t0:.0f}s)', flush=True)
        if time.time()-t0 > max_sec:
            print('time guard — checkpointing at step', step); break
    np.savez(ck, m=np.array(M, dtype=object), v=np.array(V, dtype=object),
             step=step, hist=np.array(hist), **P)
    return hist


def sample(tag='mitsu', n=600, temp=0.8, topk=40, prompt='\n', seed=1):
    """Autoregressive sampling with temperature + top-k, rolling T-token window.
    Reads shakespeare.txt only to rebuild the character vocabulary."""
    text = open(os.path.join(HERE, 'shakespeare.txt')).read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    cfg = Cfg(act='mitsu' if 'mitsu' in tag else 'gelu')
    z = np.load(os.path.join(HERE, f'ckpt_{tag}.npz'), allow_pickle=True)
    P = {k: z[k] for k in z.files if k not in ('m', 'v', 'step', 'hist')}
    rng = np.random.default_rng(seed)
    ids = [stoi[c] for c in prompt]
    for _ in range(n):
        ctx = np.array([ids[-cfg.T:]])
        logits, _, _ = forward(P, cfg, ctx)
        lg = logits[0, -1]/temp
        k_ = np.argsort(lg)[-topk:]                          # top-k filter
        pz = np.exp(lg[k_]-lg[k_].max()); pz /= pz.sum()
        ids.append(int(rng.choice(k_, p=pz)))
    return ''.join(chars[i] for i in ids)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('usage: python mitsu_gpt.py gradcheck')
        print('       python mitsu_gpt.py train  {mitsu|gelu} <steps> <max_seconds>')
        print('       python mitsu_gpt.py sample {mitsu|gelu} [prompt]')
        print("IDLE tip: import mitsu_gpt as m; print(m.sample('mitsu', prompt='ROMEO:'))")
        sys.exit(0)
    if sys.argv[1] == 'gradcheck':
        for a in ('mitsu', 'gelu'):
            print(a, 'gradcheck worst rel err:', f'{gradcheck(a):.2e}')
    elif sys.argv[1] == 'train':
        train(sys.argv[2], steps=int(sys.argv[3]), max_sec=int(sys.argv[4]))
    elif sys.argv[1] == 'sample':
        print(sample(sys.argv[2], prompt=sys.argv[3] if len(sys.argv) > 3 else '\n'))
