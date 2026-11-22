"""Export Tables 6–11 and a tie-aware per-function sensitivity audit."""
import csv
from pathlib import Path
import json
import numpy as np
from scipy.stats import mannwhitneyu
import analyze_all as aa

stats=json.load(open('output/stats.json'))
archive=json.load(open('summary_all.json'))
out=Path('output/revision_tables')
out.mkdir(parents=True,exist_ok=True)
def export(name,header,rows):
    with (out/f'{name}.csv').open('w') as stream:
        w=csv.writer(stream)
        w.writerow(header)
        w.writerows(rows)

suites=[('CEC2022 D10','cec2022',10),('CEC2017 D10','cec2017',10),('CEC2017 D30','cec2017',30)]
export('table6_ranks_holm',
       ['setting','method','mean_rank','holm_p','nemenyi_cd'],
       [[k,r['algo'],r['rank'],r['p_holm'],stats[k]['CD']] for k,_,_ in suites for r in stats[k]['holm']]+
       [[k,'SFAO',stats[k]['ranks']['SFAO'],'',stats[k]['CD']] for k,_,_ in suites])
export('table7_original_wtl',['setting','comparator','win','no_detected_difference','loss'],
       [[k,a,*v.split('/')] for k,_,_ in suites for a,v in archive[k]['wtl'].items()])
export('table8_cec2022',['function','algorithm','mean','sample_sd','median'],
       [[i,a,m,stats['CEC2022 D10']['sd'][a][i],stats['CEC2022 D10']['median'][a][i]]
        for a,fs in stats['CEC2022 D10']['means'].items() for i,m in fs.items()])
export('table9_f2_success',['algorithm','count','n','wilson_low','wilson_high'],
       [[a,r['s'],30,*r['ci']] for a,r in stats['CEC2022 D10']['F2_success'].items()])
export('table10_a12',['setting','comparator','large_for_sfao','large_against_sfao','negligible'],
       [[k,a,v['large_for_SFAO'],v['large_against'],v['negligible']] for k,_,_ in suites
        for a,v in stats[k]['A12_summary'].items()])
export('table11_ablations',['mechanism','full_better','no_detected_difference','full_worse','f2_success'],
       [[k,r['full_better'],r['nd'],r['full_worse'],r['F2_success']] for k,r in stats['ablation'].items()])
rows=[]
for setting,suite,d in suites:
    data=aa.load(suite,d)
    for a in archive[setting]['wtl']:
        for f in map(int,stats[setting]['means'][a]):
            x=np.array([data['SFAO',f][s] for s in range(30)])
            y=np.array([data[a,f][s] for s in range(30)])
            diff=y[None,:]-x[:,None]
            u=mannwhitneyu(x,y,method='asymptotic')
            signed=2*(1-u.statistic/diff.size)-1
            vals=[float(np.sign(diff[np.abs(diff)>tol]).sum()/diff.size) for tol in [1e-8,1e-6,1e-3]]
            rows.append([setting,f,a,aa.wtl(x,y),u.pvalue,signed,*vals,
                         float(np.median(x)-np.median(y)),float(np.mean(x)-np.mean(y))])
export('tie_aware_sensitivity',['setting','function','comparator','historical_class',
       'two_sided_tie_corrected_p','signed_pairwise_effect','effect_at_1e-8','effect_at_1e-6','effect_at_1e-3',
       'median_difference_sfao_minus_other','mean_difference_sfao_minus_other'],rows)
