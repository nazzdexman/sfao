"""Matched late-stage geometry experiment. All refiners use the same prefix."""
import json
import sys
from multiprocessing import Pool
import numpy as np
from sfao_final import sfao_final

def instance(kind, d):
    if kind == 'CEC2017 F4':
        from cec2017 import CEC2017
        return CEC2017(4,d)
    if kind == 'CEC2022 F2':
        from cec2022 import CEC2022
        return CEC2022(2,d)
    ident, geometry, position = kind.split('-')
    rng = np.random.default_rng(int(ident))
    q = np.linalg.qr(rng.normal(size=(d,d)))[0]
    shift = rng.uniform(-10,10,d)
    if position == 'edge':
        shift[0] = 95.0
    if geometry == 'axis':
        q = np.eye(d)
    def rosen(x):
        z = 1 + .02048*(q @ (x-shift))
        return float(np.sum(100*(z[1:]-z[:-1]**2)**2 + (z[:-1]-1)**2))
    return rosen

def run(job):
    kind,d,seed,budget,fraction,repair = job
    f = instance(kind,d)
    cut = int(budget*fraction)
    s = sfao_final(use_spin=False)
    best,_,prefix = s.optimize(f,d,-100,100,cut,seed=seed)
    assert len(prefix) == cut
    mem = s.revision_memory
    if len(mem) >= d:
        directions = mem / np.maximum(np.linalg.norm(mem,axis=1,keepdims=True),1e-300)
        m = directions.T @ directions / len(directions)
        eig,basis = np.linalg.eigh(m/(np.trace(m)/d+1e-300) + 1e-6*np.eye(d))
        scale = np.sqrt(np.maximum(eig,1e-12))
        scale /= np.sqrt(np.mean(scale**2))
    else:
        basis,scale = np.eye(d),np.ones(d)
    random_basis = np.linalg.qr(np.random.default_rng(seed+91873).normal(size=(d,d)))[0]
    def bounded(y):
        if repair == 'reflect':
            return 100 - np.abs((y+100) % 400 - 200)
        return np.clip(y,-100,100)
    out = []
    for method,b,w in [('learned',basis,scale),('isotropic',np.eye(d),np.ones(d)),
                       ('randomized',random_basis,scale)]:
        x,val,sigma = best.copy(),float(prefix[-1]),s.revision_reach
        rng = np.random.default_rng(seed+42000)
        hits = {str(t): next((i+1 for i,e in enumerate(prefix) if e<t),None) for t in (1.,.01,.001)}
        for k in range(cut+1,budget+1):
            y = bounded(x + sigma*(b @ (w*rng.standard_normal(d)))/np.sqrt(d))
            v = f(y)
            if v < val:
                x,val,sigma = y,v,sigma*1.5
            else:
                sigma *= .9
            sigma = max(sigma,2e-13)
            for t in hits:
                if hits[t] is None and val < float(t):
                    hits[t] = k
        out.append(dict(instance=kind,D=d,seed=seed,budget=budget,switch_fe=cut,repair=repair,
                        method=method,final_error=val,first_hit_fe=hits,prefix_error=float(prefix[-1])))
    return out

if __name__ == '__main__':
    jobs = [(k,d,s,b,p,r) for k,d in [('CEC2017 F4',30),('CEC2022 F2',10)] for s in range(10)
            for b in [20000] for p in [.65,.80] for r in ['clip']]
    jobs += [(f'{i}-{g}-{p}',10,s,12000,.80,r) for i in [3101,3102,3103]
             for g in ['axis','rot'] for p in ['center','edge'] for r in ['clip','reflect'] for s in range(5)]
    with Pool(6) as pool, open(sys.argv[1],'w') as f:
        for batch in pool.imap_unordered(run,jobs):
            for row in batch:
                print(json.dumps(row),file=f,flush=True)
