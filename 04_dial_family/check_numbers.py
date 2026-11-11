import json, numpy as np, collections, sys
R='results/'
def L(f): return [json.loads(l) for l in open(R+f)]
m, fa, fc, g = L('cls_mnist.jsonl'), L('cls_fashion.jsonl'), L('cls_fashion_confirm.jsonl'), L('gpt.jsonl')
print('counts', len(m), len(fa), len(fc), len(g))
for nm, rows, key in (('mnist',m,'cell'),('fashion',fa,'cell'),('confirm',fc,'cell'),('gpt',g,'variant')):
    c = collections.Counter(r[key] for r in rows); s = collections.Counter(r['seed'] for r in rows)
    print(nm, 'cells', len(c), 'per-cell', set(c.values()), 'seeds', dict(s), 'diverged', sorted({r[key] for r in rows if r['diverged']}), sum(r['diverged'] for r in rows))
# independent paired stats
def paired(rows, key, met, name, scale=1):
    d = {r['seed']: r[met] for r in rows if r[key]==name}; b = {r['seed']: r[met] for r in rows if r[key]=='mitsu'}
    ss = sorted(d); x = np.array([d[s]-b[s] for s in ss])*scale
    return x.mean(), x.mean()/(x.std(ddof=1)/np.sqrt(len(x))), np.mean([d[s] for s in ss])*scale
for nm in ['fixed(0.75,0.75)','fixed(0.5,0.5)','tri','log-learn01','gelu','silu','relu']:
    print('gpt', nm, ['%.4f'%v for v in paired(g,'variant','val',nm)])
for nm in ['inunit@64','tri@64','log-inunit@64','log-tri@64','inunit@22','tri@22','log-inunit@22','log-tri@22','fixed(0.5,0.5)','inscalar','log-inscalar','learn01','log-learn01','log-learn02']:
    print('mnist', nm, ['%.2f'%v for v in paired(m,'cell','acc',nm,100)], ' fashion', ['%.2f'%v for v in paired(fa,'cell','acc',nm,100)])
# P7 gap on GPT, light variants
light = ['mitsu','fixed(0.9,0.9)','fixed(0.75,1)','fixed(0.5,1)','fixed(0.75,0.75)','fixed(0.5,0.5)','fixed(1,1.25)','fixed(1.5,1.5)','learn01','learn02','inscalar']
def mean(rows,key,met,name):
    return np.mean([r[met] for r in rows if r[key]==name and not r['diverged']]) if any(r[key]==name and not r['diverged'] for r in rows) else float('nan')
gaps = []
for v in light:
    gp = mean(g,'variant','val','log-'+v)-mean(g,'variant','val',v); gaps.append(gp); print('P7 gpt', v, '%+.4f'%gp)
print('P7 mean gap gpt light', np.mean(gaps), ' excl (0.5,0.5):', np.mean([x for v,x in zip(light,gaps) if v!='fixed(0.5,0.5)']))
for v in ['inunit','tri']:
    print('P7 gpt heavy', v, '%+.4f'%(mean(g,'variant','val','log-'+v)-mean(g,'variant','val',v)))
for ds,rows in (('mnist',m),('fashion',fa)):
    gl=[]
    for v in light+['inunit@22','tri@22','inunit@64','tri@64']:
        a,b = mean(rows,'cell','acc','log-'+v), mean(rows,'cell','acc',v)
        if np.isfinite(a) and np.isfinite(b): gl.append((a-b)*100)
        print('P7', ds, v, '%+.2f'%((a-b)*100) if np.isfinite(a-b) else 'n/a (non-log diverged)')
    print('  mean log-minus-plain', ds, np.mean(gl), 'n', len(gl))
# P4 dials
for nm, rows, key in (('mnist',m,'cell'),('fashion',fa,'cell'),('gpt',g,'variant')):
    for v in ['learn01','learn02','log-learn01','log-learn02','inscalar','log-inscalar','inunit','log-inunit','inunit@22','inunit@64']:
        ds = [r['dials'] for r in rows if r[key]==v]
        if ds: print(nm, v, json.dumps(ds)[:400])
