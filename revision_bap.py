"""Matched cap-only ablation on two Barcelona instances."""
import json
import sys
from multiprocessing import Pool
from bap import BAP
from sfao_final import sfao_final

def run(job):
    instance,seed,cap=job
    p=BAP(instance)
    optimizer=sfao_final(food_per_worm=None,cocoon_cap=cap)
    best,err,curve=optimizer.optimize(p.objective,p.dim,0,1,10000,seed=seed)
    assert optimizer.fe==len(curve)==10000
    start,pos=p.decode(best)
    p.check(start,pos)
    return dict(instance=instance,seed=seed,cap=cap,N0=min(18*p.dim,cap) if cap else 18*p.dim,
                budget=optimizer.fe,makespan=p.makespan(start)-1,bks=p.bks)

if __name__=='__main__':
    jobs=[(i,s,c) for i in ('bcn_24B_15','bcn_36A_15') for s in range(5) for c in (100,None)]
    with Pool(5) as pool, open(sys.argv[1],'w') as stream:
        for row in pool.imap_unordered(run,jobs):
            print(json.dumps(row),file=stream,flush=True)
