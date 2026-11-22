"""Direct feasibility audit against the v0.0.1 relative-position model."""
import csv
import gzip
import itertools
import json
from pathlib import Path
import sys

archive=Path(__file__).resolve().parent
i='bcn_36A_15'
folder='hyb_dyn_fix_max-comp'
reference=Path(sys.argv[1])
src=reference/'instances'/folder/(i+'.json')
ref=reference/'bks'/folder/('bks-'+i+'.json')
for p in (src,ref):
    assert p.read_bytes()==(archive/'bap_data'/p.relative_to(reference)).read_bytes()
d=json.loads(src.read_text())
b=json.loads(ref.read_text())
q=d['n_berths']
horizon=d['n_periods']
n=d['n_ships']
arr,h,length=d['arrival_time'],d['handling_time'],d['ship_len']
assert all(x==1 for x in d['berth_len'])
def verify(name,t,p):
    assert len(t)==len(p)==n
    assert all(x==int(x) for x in t+p)
    c=[t[k]+h[k]-1 for k in range(n)]
    for k in range(n):
        assert arr[k]<=t[k]<=horizon-h[k]+1
        assert arr[k]+h[k]-1<=c[k]<=horizon
        assert 0<=p[k]<=q-length[k]
    for k,j in itertools.combinations(range(n),2):
        sigma_kj=int(t[j]>=t[k]+h[k])
        sigma_jk=int(t[k]>=t[j]+h[j])
        delta_kj=int(p[j]>=p[k]+length[k])
        delta_jk=int(p[k]>=p[j]+length[j])
        assert 1<=sigma_kj+sigma_jk+delta_kj+delta_jk
        assert sigma_kj+sigma_jk<=1 and delta_kj+delta_jk<=1
        for x,y,s,e in ((k,j,sigma_kj,delta_kj),(j,k,sigma_jk,delta_jk)):
            assert t[y]>=t[x]+h[x]+(s-1)*horizon
            assert p[y]>=p[x]+length[x]+(e-1)*q
    return [[name,k,arr[k],h[k],length[k],t[k],p[k],c[k]] for k in range(n)]
order=sorted(b['ships'],key=lambda x:x['data_ship_id'])
record=verify('reference',[x['mooring_time'] for x in order],[x['mooring_berth'] for x in order])
assert [x[-1] for x in record]==[s['completion_time'] for s in order]
assert max(x[-1] for x in record)==b['makespan']==188
with gzip.open(archive/'results/bap_discrete.jsonl.gz','rt') as stream:
    ga=next(r for r in map(json.loads,stream) if r['algo']=='GA' and r['inst']==i and r['seed']==0)
r=verify('GA seed 0',ga['start'],ga['pos'])
assert max(x[-1] for x in r)==186
with Path(sys.argv[2]).open('w') as stream:
    writer=csv.writer(stream)
    writer.writerow(['schedule','ship','arrival','handling','length','start','position','inclusive_completion'])
    writer.writerows(record+r)
