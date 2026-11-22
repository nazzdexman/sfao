"""Instrumented reruns. Retain every best-so-far objective evaluation."""
import json
from multiprocessing import Pool
from pathlib import Path
import sys
import numpy as np
from revision_pilot import instance
from sfao_final import sfao_final

def run(job):
    name,d,seed,fe,both = job
    f = instance(name,d)
    s = sfao_final(**({'use_spin':False,'use_memory':False} if both else {}))
    _,v,curve=s.optimize(f,d,-100,100,fe,seed=seed)
    assert len(curve)==s.fe==fe
    return dict(name=name,D=d,seed=seed,budget=fe,both_disabled=both,
                first_M7_FE=s.revision_m7_first_fe,error=v),curve

if __name__=='__main__':
    out=Path(sys.argv[1])
    out.mkdir(exist_ok=True)
    jobs=[(n,d,s,f,False) for n,d,f in [('CEC2022 F2',10,200000),('CEC2017 F4',30,300000)]
          for s in range(3)]
    jobs += [('CEC2022 F2',10,s,200000,True) for s in range(30)]
    with Pool(8) as pool, (out/'trace_runs.jsonl').open('w') as stream:
        for row,curve in pool.imap_unordered(run,jobs):
            suffix='both-off' if row['both_disabled'] else 'full'
            name=f"{row['name'].replace(' ','_')}_{suffix}_{row['seed']:02}.npz"
            np.savez_compressed(out/name,curve=curve)
            print(json.dumps({**row,'file':name}),file=stream,flush=True)
