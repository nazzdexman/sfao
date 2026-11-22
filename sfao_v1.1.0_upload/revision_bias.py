"""Location, rotation and full-run repair comparison with matched seeds."""
import json
import sys
from multiprocessing import Pool
from revision_pilot import instance
from sfao_final import sfao_final
import numpy as np

def run(job):
    name,seed,repair = job
    d=10
    fe=12000
    s=sfao_final(revision_repair=repair)
    _,v,c=s.optimize(instance(name,d),d,-100,100,fe,seed=seed)
    assert s.fe==len(c)==fe
    hits={}
    for t in (1.,.01,.001):
        where=np.flatnonzero(c<t)
        hits[str(t)]=int(where[0])+1 if len(where) else None
    return dict(instance=name,seed=seed,repair=repair,budget=fe,D=d,final_error=v,first_hit_fe=hits)

if __name__=='__main__':
    jobs=[(f'{i}-{g}-{p}',s,r) for i in [3101,3102,3103] for g in ['axis','rot']
          for p in ['center','edge'] for s in range(5) for r in ['clip','reflect']]
    with Pool(6) as pool, open(sys.argv[1],'w') as stream:
        for row in pool.imap_unordered(run,jobs):
            print(json.dumps(row),file=stream,flush=True)
