"""Exploratory matched-seed bootstrap for the CEC2022 F2 M7-by-M8 factorial."""
import gzip
import json
from pathlib import Path
import sys
import numpy as np

base='SFAO:cocoon:aspin'
def stored(filename,variant):
    with gzip.open(Path(__file__).parent/'results'/f'{filename}.jsonl.gz','rt') as f:
        return {r['seed']:r['err'] for r in map(json.loads,f) if r['f']==2 and r['D']==10 and r['algo']==variant}

with open(sys.argv[1]) as f:
    both={r['seed']:r['error'] for r in map(json.loads,f) if r['both_disabled']}
assert set(both)==set(range(30))
groups=[stored('cec30',base),stored('abl_pc',base+':nospin'),
        stored('abl_mac',base+':nomemory'),both]
x=np.array([[g[s] for s in range(30)] for g in groups])
x[x<1e-8]=0
coeff=np.array([[.5,-.5,.5,-.5],[.5,.5,-.5,-.5],[1,-1,-1,1]])
rng=np.random.default_rng(821)
index=rng.integers(0,30,(20000,30))
result={}
for target in [1e-6,1e-4,1e-3,1e-2,1e-1]:
    z=(x<target).astype(float)
    result[str(target)]={'counts':z.sum(1).astype(int).tolist()}
    samples=z[:,index].mean(2)
    result[str(target)]['contrasts']=np.column_stack([coeff @ z.mean(1),np.quantile(coeff @ samples,[.025,.975],axis=1).T]).tolist()
samples=x[:,index].mean(2)
result['final_error']={'means':x.mean(1).tolist(),'medians':np.median(x,axis=1).tolist(),
                       'contrasts':np.column_stack([coeff @ x.mean(1),np.quantile(coeff @ samples,[.025,.975],axis=1).T]).tolist()}
print(json.dumps(result,indent=2))
