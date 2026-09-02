"""Nano scaling ladder — MitsuGLU vs SwiGLU, matched pairs, char-level WikiText-2.
Per rung: identical init (same shapes, same seed) and identical batch schedule;
ONLY the gate nonlinearity differs. Resumable job queue for chunked CPU runs."""
import numpy as np, json, os, sys, time
sys.path.insert(0, '/home/claude/mitsu-gpt')
from mitsu_gpt import softplus, sigm, mitsu, mitsu_d, ln_f, ln_b

DIR = '/home/claude/mitsu-8b'
RUNGS = [dict(d=32, L=1), dict(d=48, L=2), dict(d=72, L=2), dict(d=104, L=3)]
H = 2; T = 64; B = 32; LR = 2e-3; MULT = 15; CLIP = 1.0; VCAP = 256

def silu(u): return u*sigm(u)
def silu_d(u):
    s = sigm(u); return s*(1+u*(1-s))

class Cfg:
    def __init__(self, d, L, fam, V, GM, dtype=np.float32):
        self.d,self.L,self.fam,self.V,self.GM,self.dtype = d,L,fam,V,GM,dtype
        self.H, self.T, self.hd, self.m = H, T, d//H, 3*d

def gate(cfg, u):
    return cfg.GM*mitsu(u) if cfg.fam == 'mitsuglu' else silu(u)
def gate_d(cfg, u):
    return cfg.GM*mitsu_d(u) if cfg.fam == 'mitsuglu' else silu_d(u)

def init_params(cfg, seed):
    rng = np.random.default_rng(seed); dt = cfg.dtype; std = 0.06
    P = {'tok': (rng.standard_normal((cfg.V, cfg.d))*std).astype(dt),
         'pos': (rng.standard_normal((cfg.T, cfg.d))*std).astype(dt)}
    for l in range(cfg.L):
        P[f'ln1g{l}'] = np.ones(cfg.d, dt); P[f'ln1b{l}'] = np.zeros(cfg.d, dt)
        P[f'qkv{l}'] = (rng.standard_normal((cfg.d, 3*cfg.d))*std).astype(dt)
        P[f'proj{l}'] = (rng.standard_normal((cfg.d, cfg.d))*std/np.sqrt(2*cfg.L)).astype(dt)
        P[f'ln2g{l}'] = np.ones(cfg.d, dt); P[f'ln2b{l}'] = np.zeros(cfg.d, dt)
        P[f'w1{l}'] = (rng.standard_normal((cfg.d, cfg.m))*std).astype(dt)
        P[f'w3{l}'] = (rng.standard_normal((cfg.d, cfg.m))*std).astype(dt)
        P[f'w2{l}'] = (rng.standard_normal((cfg.m, cfg.d))*std/np.sqrt(2*cfg.L)).astype(dt)
    P['lnfg'] = np.ones(cfg.d, dt); P['lnfb'] = np.zeros(cfg.d, dt)
    P['head'] = (rng.standard_normal((cfg.d, cfg.V))*std).astype(dt)
    return P

def forward(P, cfg, idx, targets=None):
    Bz, Tz = idx.shape
    x = P['tok'][idx] + P['pos'][:Tz]
    cache = {'idx': idx, 'blocks': []}
    mask = np.triu(np.full((Tz, Tz), -np.inf, cfg.dtype), 1)
    for l in range(cfg.L):
        h, c1 = ln_f(x, P[f'ln1g{l}'], P[f'ln1b{l}'])
        qkv = h @ P[f'qkv{l}']
        q, k, v = [a.reshape(Bz, Tz, cfg.H, cfg.hd).transpose(0, 2, 1, 3) for a in np.split(qkv, 3, -1)]
        A = q @ k.transpose(0, 1, 3, 2)/np.sqrt(cfg.hd) + mask
        A -= A.max(-1, keepdims=True)
        eA = np.exp(A); S = eA/eA.sum(-1, keepdims=True)
        o = (S @ v).transpose(0, 2, 1, 3).reshape(Bz, Tz, cfg.d)
        x = x + o @ P[f'proj{l}']
        h2, c2 = ln_f(x, P[f'ln2g{l}'], P[f'ln2b{l}'])
        u1 = h2 @ P[f'w1{l}']; u3 = h2 @ P[f'w3{l}']
        g = gate(cfg, u1); a = g*u3
        x = x + a @ P[f'w2{l}']
        cache['blocks'].append((h, q, k, v, S, o, h2, u1, u3, g, a, c1, c2))
    xf, cf = ln_f(x, P['lnfg'], P['lnfb'])
    logits = xf @ P['head']
    cache['xf'], cache['cf'] = xf, cf
    if targets is None: return logits, None, cache
    z = logits - logits.max(-1, keepdims=True)
    ez = np.exp(z); p = ez/ez.sum(-1, keepdims=True)
    loss = -np.log(np.maximum(p[np.arange(Bz)[:, None], np.arange(Tz)[None, :], targets], 1e-12)).mean()
    cache['p'] = p; cache['targets'] = targets
    return logits, float(loss), cache

def backward(P, cfg, cache):
    Bz, Tz = cache['idx'].shape
    G = {k: np.zeros_like(w) for k, w in P.items()}
    dlog = cache['p'].copy()
    dlog[np.arange(Bz)[:, None], np.arange(Tz)[None, :], cache['targets']] -= 1
    dlog /= (Bz*Tz)
    G['head'] = cache['xf'].reshape(-1, cfg.d).T @ dlog.reshape(-1, cfg.V)
    dxf = dlog @ P['head'].T
    dx, G['lnfg'], G['lnfb'] = ln_b(dxf, cache['cf'])
    for l in range(cfg.L-1, -1, -1):
        h, q, k, v, S, o, h2, u1, u3, g, a, c1, c2 = cache['blocks'][l]
        dm = dx
        G[f'w2{l}'] = a.reshape(-1, cfg.m).T @ dm.reshape(-1, cfg.d)
        da = dm @ P[f'w2{l}'].T
        du1 = da*u3*gate_d(cfg, u1); du3 = da*g
        G[f'w1{l}'] = h2.reshape(-1, cfg.d).T @ du1.reshape(-1, cfg.m)
        G[f'w3{l}'] = h2.reshape(-1, cfg.d).T @ du3.reshape(-1, cfg.m)
        dh2 = du1 @ P[f'w1{l}'].T + du3 @ P[f'w3{l}'].T
        dx2, G[f'ln2g{l}'], G[f'ln2b{l}'] = ln_b(dh2, c2)
        dx = dx + dx2
        dy = dx
        G[f'proj{l}'] = o.reshape(-1, cfg.d).T @ dy.reshape(-1, cfg.d)
        do = (dy @ P[f'proj{l}'].T).reshape(Bz, Tz, cfg.H, cfg.hd).transpose(0, 2, 1, 3)
        dS = do @ v.transpose(0, 1, 3, 2)
        dv = S.transpose(0, 1, 3, 2) @ do
        dA = S*(dS - (dS*S).sum(-1, keepdims=True))
        dq = dA @ k/np.sqrt(cfg.hd); dk = dA.transpose(0, 1, 3, 2) @ q/np.sqrt(cfg.hd)
        dqkv = np.concatenate([gg.transpose(0, 2, 1, 3).reshape(Bz, Tz, cfg.d) for gg in (dq, dk, dv)], -1)
        G[f'qkv{l}'] = h.reshape(-1, cfg.d).T @ dqkv.reshape(-1, 3*cfg.d)
        dh = dqkv @ P[f'qkv{l}'].T
        dx1, G[f'ln1g{l}'], G[f'ln1b{l}'] = ln_b(dh, c1)
        dx = dx + dx1
    G['pos'][:Tz] += dx.sum(0)
    np.add.at(G['tok'], cache['idx'], dx)
    return G

def gradcheck(fam):
    cfg = Cfg(8, 1, fam, V=11, GM=2.0, dtype=np.float64)
    P = init_params(cfg, 3)
    rng = np.random.default_rng(4)
    idx = rng.integers(0, 11, (2, 5)); tgt = rng.integers(0, 11, (2, 5))
    _, _, cache = forward(P, cfg, idx, tgt)
    G = backward(P, cfg, cache)
    worst = 0.0
    for name, w in P.items():
        for _ in range(5):
            ix = tuple(rng.integers(0, s) for s in w.shape)
            eps = 1e-6; old = w[ix]
            w[ix] = old+eps; lp = forward(P, cfg, idx, tgt)[1]
            w[ix] = old-eps; lm = forward(P, cfg, idx, tgt)[1]
            w[ix] = old
            num = (lp-lm)/(2*eps)
            rel = abs(num-G[name][ix])/max(1e-8, abs(num)+abs(G[name][ix]))
            worst = max(worst, rel)
    return worst

def nparams(P): return sum(w.size for w in P.values())

def setup():
    tr = open(f'{DIR}/wiki_train.txt').read(); va = open(f'{DIR}/wiki_valid.txt').read()
    from collections import Counter
    cnt = Counter(tr)
    keep = [c for c, _ in cnt.most_common(VCAP-1)]
    stoi = {c: i+1 for i, c in enumerate(keep)}
    enc = lambda s: np.array([stoi.get(c, 0) for c in s], np.int64)
    np.savez_compressed(f'{DIR}/wiki_tokens.npz', tr=enc(tr), va=enc(va))
    rng = np.random.default_rng(0); z = rng.standard_normal(1_000_000)
    gm = float(np.std(silu(z))/np.std(mitsu(z)))
    jobs = []
    for r, rc in enumerate(RUNGS):
        cfg = Cfg(rc['d'], rc['L'], 'x', VCAP, gm)
        Np = nparams(init_params(cfg, 0))
        steps = int(np.ceil(MULT*Np/(B*T)))
        for fam in ('mitsuglu', 'swiglu'):
            jobs.append(dict(rung=r, fam=fam, d=rc['d'], L=rc['L'], N=Np, steps=steps, step=0, done=False, val=None))
        print(f"rung {r}: d={rc['d']} L={rc['L']}  params={Np:,}  tokens={MULT*Np:,}  steps={steps}")
    st = dict(GM=round(gm, 4), jobs=jobs)
    json.dump(st, open(f'{DIR}/ladder_state.json', 'w'), indent=1)
    print('gate_mult (frozen):', st['GM'], '| vocab:', VCAP, '| train tokens:', len(tr))

def run(budget):
    st = json.load(open(f'{DIR}/ladder_state.json'))
    z = np.load(f'{DIR}/wiki_tokens.npz'); tr, va = z['tr'], z['va']
    t0 = time.time()
    for jb in st['jobs']:
        if jb['done']: continue
        cfg = Cfg(jb['d'], jb['L'], jb['fam'], VCAP, st['GM'])
        ck = f"{DIR}/lad_{jb['rung']}_{jb['fam']}.npz"
        if os.path.exists(ck):
            zz = np.load(ck, allow_pickle=True)
            P = {k: zz[k] for k in zz.files if k not in ('m', 'v')}
            M = zz['m'].item(); V = zz['v'].item()
        else:
            P = init_params(cfg, seed=100+jb['rung'])
            M = {k: np.zeros_like(w) for k, w in P.items()}
            V = {k: np.zeros_like(w) for k, w in P.items()}
        brng = np.random.default_rng(5000+jb['rung'])
        for _ in range(jb['step']): brng.integers(0, len(tr)-T-1, B)
        step = jb['step']
        while step < jb['steps']:
            ix = brng.integers(0, len(tr)-T-1, B)
            x = np.stack([tr[i:i+T] for i in ix]); y = np.stack([tr[i+1:i+T+1] for i in ix])
            _, loss, cache = forward(P, cfg, x, y)
            if not np.isfinite(loss):
                jb['done'] = True; jb['val'] = 'DIVERGED'; break
            G = backward(P, cfg, cache)
            gn = np.sqrt(sum(float((g*g).sum()) for g in G.values()))
            sc = min(1.0, CLIP/max(gn, 1e-12)); step += 1
            lr_t = LR*min(1.0, step/100)
            b1c = 1-0.9**step; b2c = 1-0.999**step
            for kk in P:
                g = G[kk]*sc
                M[kk] = 0.9*M[kk]+0.1*g; V[kk] = 0.999*V[kk]+0.001*g*g
                P[kk] -= (lr_t*(M[kk]/b1c)/(np.sqrt(V[kk]/b2c)+1e-8)).astype(P[kk].dtype)
            if time.time()-t0 > budget: break
        jb['step'] = step
        if step >= jb['steps'] and not jb['done']:
            vrng = np.random.default_rng(777)
            vls = []
            for _ in range(10):
                ix = vrng.integers(0, len(va)-T-1, B)
                vx = np.stack([va[i:i+T] for i in ix]); vy = np.stack([va[i+1:i+T+1] for i in ix])
                vls.append(forward(P, cfg, vx, vy)[1])
            jb['val'] = float(np.mean(vls)); jb['done'] = True
            print(f"DONE rung {jb['rung']} {jb['fam']:8s} N={jb['N']:,}  val={jb['val']:.4f}")
        np.savez(ck, m=np.array(M, dtype=object), v=np.array(V, dtype=object), **P)
        json.dump(st, open(f'{DIR}/ladder_state.json', 'w'), indent=1)
        if time.time()-t0 > budget:
            rem = sum(j['steps']-j['step'] for j in st['jobs'] if not j['done'])
            print(f"...paused (queue: {rem} steps remain)"); return
    print('QUEUE COMPLETE')

def fit():
    from scipy.optimize import curve_fit
    st = json.load(open(f'{DIR}/ladder_state.json'))
    out = {}
    for fam in ('mitsuglu', 'swiglu'):
        pts = sorted([(j['N'], j['val']) for j in st['jobs'] if j['fam'] == fam and isinstance(j['val'], float)])
        N = np.array([p[0] for p in pts], float); Lv = np.array([p[1] for p in pts], float)
        f = lambda n, a, al, c: a*n**(-al) + c
        try:
            popt, _ = curve_fit(f, N, Lv, p0=[30, 0.3, 1.0], maxfev=20000)
            out[fam] = dict(points=pts, a=popt[0], alpha=popt[1], c=popt[2],
                            L8B=float(f(8e9, *popt)))
        except Exception as e:
            out[fam] = dict(points=pts, error=str(e))
    json.dump(out, open(f'{DIR}/ladder_fit.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == '__main__':
    if sys.argv[1] == 'gradcheck':
        for fam in ('mitsuglu', 'swiglu'):
            print(fam, 'gradcheck worst rel err:', f'{gradcheck(fam):.2e}')
    elif sys.argv[1] == 'setup': setup()
    elif sys.argv[1] == 'run': run(int(sys.argv[2]))
    elif sys.argv[1] == 'fit': fit()
