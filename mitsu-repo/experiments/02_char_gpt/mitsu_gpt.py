"""MITSU-GPT nano — a character-level GPT in pure numpy, hand-derived backprop.
The MLP activation is switchable: 'mitsu' (centered a=b=1: s(s-1)/(s+2), s=softplus)
or 'gelu'. Twin runs share init and batch schedule; ONLY the activation differs."""
import numpy as np, time, os

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
    def __init__(self, V=65, T=64, d=64, H=2, L=2, act='mitsu', dtype=np.float32):
        self.V,self.T,self.d,self.H,self.L,self.act,self.dtype = V,T,d,H,L,act,dtype
        self.hd = d//H; self.m = 4*d

def init_params(cfg, seed=0):
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

def ln_f(x, g, b):
    mu = x.mean(-1, keepdims=True); xc = x - mu
    var = (xc*xc).mean(-1, keepdims=True); inv = 1/np.sqrt(var + 1e-5)
    xh = xc*inv
    return g*xh + b, (xh, inv, g)
def ln_b(dy, c):
    xh, inv, g = c
    dg = (dy*xh).sum((0,1)); db = dy.sum((0,1))
    dxh = dy*g
    dx = (dxh - dxh.mean(-1, keepdims=True) - xh*(dxh*xh).mean(-1, keepdims=True))*inv
    return dx, dg, db

def forward(P, cfg, idx, targets=None):
    B, T = idx.shape; f = ACT[cfg.act]
    x = P['tok'][idx] + P['pos'][:T]
    cache = {'idx': idx, 'B': B, 'T': T, 'blocks': []}
    mask = np.triu(np.full((T, T), -np.inf, cfg.dtype), 1)
    for l in range(cfg.L):
        h, c1 = ln_f(x, P[f'ln1g{l}'], P[f'ln1b{l}'])
        qkv = h @ P[f'qkv{l}']
        q, k, v = [a.reshape(B, T, cfg.H, cfg.hd).transpose(0, 2, 1, 3) for a in np.split(qkv, 3, -1)]
        A = q @ k.transpose(0, 1, 3, 2)/np.sqrt(cfg.hd) + mask
        A = A - A.max(-1, keepdims=True)
        eA = np.exp(A); S = eA/eA.sum(-1, keepdims=True)
        o = (S @ v).transpose(0, 2, 1, 3).reshape(B, T, cfg.d)
        y = o @ P[f'proj{l}']
        x = x + y
        h2, c2 = ln_f(x, P[f'ln2g{l}'], P[f'ln2b{l}'])
        u = h2 @ P[f'fc{l}']; a = f[0](u); m = a @ P[f'out{l}']
        x = x + m
        cache['blocks'].append((h, q, k, v, S, o, h2, u, a, c1, c2))
    xf, cf = ln_f(x, P['lnfg'], P['lnfb'])
    logits = xf @ P['head']
    cache['xf'], cache['cf'] = xf, cf
    if targets is None: return logits, None, cache
    z = logits - logits.max(-1, keepdims=True)
    ez = np.exp(z); p = ez/ez.sum(-1, keepdims=True)
    loss = -np.log(np.maximum(p[np.arange(B)[:, None], np.arange(T)[None, :], targets], 1e-12)).mean()
    cache['p'] = p; cache['targets'] = targets
    return logits, float(loss), cache

def backward(P, cfg, cache):
    B, T = cache['B'], cache['T']; f = ACT[cfg.act]
    G = {k: np.zeros_like(w) for k, w in P.items()}
    dlog = cache['p'].copy()
    dlog[np.arange(B)[:, None], np.arange(T)[None, :], cache['targets']] -= 1
    dlog /= (B*T)
    G['head'] = cache['xf'].reshape(-1, cfg.d).T @ dlog.reshape(-1, cfg.V)
    dxf = dlog @ P['head'].T
    dx, G['lnfg'], G['lnfb'] = ln_b(dxf, cache['cf'])
    for l in range(cfg.L-1, -1, -1):
        h, q, k, v, S, o, h2, u, a, c1, c2 = cache['blocks'][l]
        # mlp branch
        dm = dx
        G[f'out{l}'] = a.reshape(-1, cfg.m).T @ dm.reshape(-1, cfg.d)
        da = dm @ P[f'out{l}'].T
        du = da*f[1](u)
        G[f'fc{l}'] = h2.reshape(-1, cfg.d).T @ du.reshape(-1, cfg.m)
        dh2 = du @ P[f'fc{l}'].T
        dx2, G[f'ln2g{l}'], G[f'ln2b{l}'] = ln_b(dh2, c2)
        dx = dx + dx2
        # attn branch
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
    G['pos'] = dx.sum(0)[:T]
    np.add.at(G['tok'], cache['idx'], dx)
    return G

def gradcheck(act='mitsu'):
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
    ix = rng.integers(0, len(tokens)-cfg.T-1, B)
    x = np.stack([tokens[i:i+cfg.T] for i in ix])
    y = np.stack([tokens[i+1:i+cfg.T+1] for i in ix])
    return x, y

def train(act, steps=2000, B=32, lr=2e-3, seed=0, max_sec=1200, tag=None):
    tag = tag or act
    text = open(os.path.join(os.path.dirname(__file__), 'shakespeare.txt')).read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    tokens = np.array([stoi[c] for c in text], np.int64)
    n = int(0.9*len(tokens)); tr, va = tokens[:n], tokens[n:]
    cfg = Cfg(act=act)
    ck = f'/home/claude/mitsu-gpt/ckpt_{tag}.npz'
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
    for _ in range(step0):  # replay schedule so resumes and twins stay aligned
        brng.integers(0, len(tr)-cfg.T-1, B)
    vrng = np.random.default_rng(123)
    vbat = [get_batch(va, cfg, B, vrng) for _ in range(8)]
    t0 = time.time()
    for step in range(step0+1, steps+1):
        x, y = get_batch(tr, cfg, B, brng)
        _, loss, cache = forward(P, cfg, x, y)
        if not np.isfinite(loss): print('DIVERGED at', step); break
        G = backward(P, cfg, cache)
        gn = np.sqrt(sum(float((g*g).sum()) for g in G.values()))
        sc = min(1.0, 1.0/max(gn, 1e-12))
        lr_t = lr*min(1.0, step/100)
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
    text = open('/home/claude/mitsu-gpt/shakespeare.txt').read()
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    cfg = Cfg(act='mitsu' if 'mitsu' in tag else 'gelu')
    z = np.load(f'/home/claude/mitsu-gpt/ckpt_{tag}.npz', allow_pickle=True)
    P = {k: z[k] for k in z.files if k not in ('m', 'v', 'step', 'hist')}
    rng = np.random.default_rng(seed)
    ids = [stoi[c] for c in prompt]
    for _ in range(n):
        ctx = np.array([ids[-cfg.T:]])
        logits, _, _ = forward(P, cfg, ctx)
        lg = logits[0, -1]/temp
        k_ = np.argsort(lg)[-topk:]
        pz = np.exp(lg[k_]-lg[k_].max()); pz /= pz.sum()
        ids.append(int(rng.choice(k_, p=pz)))
    return ''.join(chars[i] for i in ids)

if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'gradcheck':
        for a in ('mitsu', 'gelu'):
            print(a, 'gradcheck worst rel err:', f'{gradcheck(a):.2e}')
    elif sys.argv[1] == 'train':
        train(sys.argv[2], steps=int(sys.argv[3]), max_sec=int(sys.argv[4]))
    elif sys.argv[1] == 'sample':
        print(sample(sys.argv[2], prompt=sys.argv[3] if len(sys.argv) > 3 else '\n'))
