"""Regenerate endpoint-consistent Figure 2 and audit the archived run records."""
import collections
import gzip
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from scipy.stats import mannwhitneyu, ranksums
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_all as aa
import pv
from bap import BAP

root=Path(__file__).resolve().parent
out=root/'revision_output'
out.mkdir(exist_ok=True)
def rows(name):
    with gzip.open(root/'results'/f'{name}.jsonl.gz','rt') as stream:
        yield from map(json.loads,stream)

panels=[('cec2022',10,2),('cec2022',10,6),('cec2022',10,11),
        ('cec2017',30,4),('cec2017',30,21),('cec2017',30,13)]
algos=['SFAO','L-SRTDE','jSO','L-SHADE','CMA-ES','CLPSO']
by=collections.defaultdict(dict)
duplicates=collections.Counter()
for suite,files in aa.FILES.items():
    for name in files:
        for r in rows(name.removesuffix('.jsonl')):
            if r['suite']!=suite:
                continue
            a='SFAO' if r['algo']==aa.SF else ('SFAO-proto' if r['algo']=='SFAO' else r['algo'])
            key=suite,r['D'],r['f'],a
            if r['seed'] in by[key]:
                duplicates['repeat']+=1
                duplicates['different_end']+=by[key][r['seed']]['err']!=r['err']
            by[key][r['seed']]=r
data={'duplicate_counts':duplicates,'convergence':{},'f9':{},'f11':{},'pv':{}}
def errors(rs):
    return np.array([0 if r['err']<1e-8 else r['err'] for r in rs])
def group(k):
    rs=by[k]
    assert set(rs)==set(range(30)), (k,sorted(rs))
    return [rs[s] for s in range(30)]
def label(key):
    return ' '.join(map(str,key))

with gzip.open(out/'benchmark_run_audit.jsonl.gz','wt') as stream:
    for k,rs in sorted(by.items()):
        for r in rs.values():
            fe=200000 if r['suite']=='cec2022' else r['D']*10000
            e=float(errors([r])[0])
            stream.write(json.dumps({**r,'budget_fe':fe,'target_1e_3':e<1e-3,
                           'zeroed_final_error':e})+'\n')

COLORS={'SFAO':'#b32228','L-SRTDE':'#1f77b4','jSO':'#ff7f0e','L-SHADE':'#2ca02c','CMA-ES':'#17becf','CLPSO':'#9467bd'}
fig,axs=plt.subplots(2,3,figsize=(7.1,4.5))
for ax,p in zip(axs.flat,panels):
    suite,d,f=p
    budget=200000 if suite=='cec2022' else d*10000
    for a in algos:
        k=(*p,a)
        rs=group(k)
        grid=np.unique(np.concatenate([np.array(r['curve_fe']) for r in rs]+[[budget]]))
        grid=grid[grid>=max(r['curve_fe'][0] for r in rs)]
        yy=[]
        for r in rs:
            x=np.array(r['curve_fe'])
            y=np.array(r['curve'])
            assert np.all(np.diff(x)>0)
            v=y[np.searchsorted(x,grid,side='right')-1].copy()
            yy.append(v)
        e=errors(rs)
        last=np.array([r['curve'][-1] for r in rs])
        med=np.median(yy,axis=0)
        med[-1]=np.median(e)
        assert 0<=med[-1]<=2*np.mean(e)+1e-15
        at=np.flatnonzero(grid<=.8*budget)[-1]
        data['convergence'][label(k)]=dict(budget=budget,stored_last_fes=sorted({r['curve_fe'][-1] for r in rs}),
          stored_median=float(np.median(last)),final_median=float(np.median(e)),mean=float(e.mean()),
          unequal_last=int(np.count_nonzero(e!=last)),pre_fe=int(grid[at]),pre_median=float(med[at]),
          next_fe=int(grid[at+1]),next_median=float(med[at+1]),
          zero=int(np.count_nonzero(e==0)))
        line,=ax.plot(grid[:-1],np.maximum(med[:-1],1e-8),lw=1.7 if a=='SFAO' else .9,
                     color=COLORS[a],label=a)
        ax.plot(grid[-2:],np.maximum(med[-2:],1e-8),ls=':',lw=.8,color=line.get_color())
        ax.plot(budget,max(med[-1],1e-8),'o',ms=2.6,color=line.get_color())
    ax.set(xscale='log',yscale='log',title=f'{suite.upper()} F{f} (D={d}; {budget:,} FE)')
    ax.title.set_fontsize(8)
    ax.tick_params(which='both',labelsize=7)
    ax.grid(alpha=.2)
for ax in axs[-1]: ax.set_xlabel('Function evaluations',fontsize=8)
for ax in axs[:,0]: ax.set_ylabel('Median final/best-so-far error',fontsize=8)
axs[0,0].legend(fontsize=6.2,frameon=False,loc='lower left')
fig.tight_layout()
fig.savefig(out/'Figure2.png',dpi=330)
plt.close(fig)
x=errors(group(('cec2022',10,9,'SFAO')))
for a in ['SFAO']+aa.CHAMP+['CLPSO','ABC']:
    y=errors(group(('cec2022',10,9,a)))
    u=mannwhitneyu(x,y,method='asymptotic')
    delta=y[None,:]-x[:,None]
    data['f9'][a]=dict(mean=float(y.mean()),median=float(np.median(y)),min=float(y.min()),max=float(y.max()),
        original_p=float(ranksums(x,y).pvalue),allclose=bool(np.allclose(x,y)),tie_corrected_p=float(u.pvalue),
        signed_effect=float((np.sign(delta)).mean()),
        signed_effect_tolerances={str(t):float(np.sign(delta[np.abs(delta)>t]).sum()/delta.size)
                                  for t in (1e-8,1e-6,1e-3)})
    data['f11'][a]=int(np.count_nonzero(errors(group(('cec2022',10,11,a)))==0))
def current_rmse(params,model):
    if model=='SDM':
        ph,i1,rs,rsh,n1=params
        diodes=[(i1,n1)]
    else:
        ph,i1,i2,rs,rsh,n1,n2=params
        diodes=[(i1,n1),(i2,n2)]
    def predict(v):
        def equation(i):
            vd=v+i*rs
            return i-ph+sum(a*np.expm1(vd/(n*pv.Vt)) for a,n in diodes)+vd/max(rsh,1e-12)
        return brentq(equation,-10,10,xtol=1e-14)
    return float(np.sqrt(np.mean((np.array([predict(v) for v in pv.V])-pv.I)**2)))
g=collections.defaultdict(list)
for r in rows('pv_results'): g[r['model'],r['algo']].append(r)
for (model,a),rs in sorted(g.items()):
    e=np.array([r['rmse'] for r in rs])
    best=min(rs,key=lambda r:r['rmse'])
    params=best['params']
    assert np.isclose(pv.PROBLEMS[model](params),best['rmse'],atol=1e-12)
    ref=pv.BEST_KNOWN[model]
    data['pv'][label((model,a))]=dict(seed=best['seed'],params=params,residual=best['rmse'],
        prediction=current_rmse(params,model),median=float(np.median(e)),reference=ref,
        successes=int(np.count_nonzero(e<=ref*1.0001)))
data['bap_186']=[]
for r in rows('bap_discrete'):
    if r['algo']=='GA' and r['inst']=='bcn_36A_15' and r['makespan']==187:
        p=BAP(r['inst'])
        start,pos=np.array(r['start']),np.array(r['pos'])
        p.check(start,pos)
        data['bap_186'].append(dict(seed=r['seed'],exclusive=p.makespan(start),inclusive=p.makespan(start)-1,
                                    bks=p.bks,dual=p.dual))
with open(out/'audit.json','w') as stream:
    json.dump(data,stream,indent=2)
