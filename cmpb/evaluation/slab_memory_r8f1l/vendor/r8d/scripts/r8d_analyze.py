"""D4: join baseline/profile/metadata by subject and candidate, never pool reps.

python -B scripts/r8d_analyze.py --run results/server_RUN
Refuses partial runs. Fits only an exploratory held-out-subject predictor,
and reports no rule if rank, error or regret requirements fail.
"""
import argparse
import csv
import math
import statistics as st
from pathlib import Path
from r8d_common import readj, writej, seal
from r8d_validate import validate_run
from r8d_cost_model import features,loso


def analyze(d):
    validation=validate_run(d)
    if validation['status']!='PASS':
        return {'status':'NOT EXECUTED / INCOMPLETE','assessment':'INCONCLUSIVE','validation':validation,'rows':[],
                'policy_model':{'status':'NOT IDENTIFIED'},'note':'No statistics promoted from partial, stub, smoke or unvalidated evidence.'}
    groups={}
    for p in sorted(d.glob('workers/*/record.json')):
        r=readj(p)
        if r['purpose']=='smoke': continue
        groups.setdefault((r['subject'],r['copies'],r['K']),[]).append(r)
    rows=[]
    for (sid,m,K),rr in sorted(groups.items()):
        timing=[r['algorithm_wall_s'] for r in rr if r['purpose']=='timing']; med=st.median(timing)
        meta=next(r for r in rr if r['purpose']=='metadata'); profile=next(r for r in rr if r['purpose']=='profile')
        rows.append({'subject':sid,'copies':m,'K':K,'vol_shape':meta['vol_shape'],'time_median_s':med,
            'time_repetitions_s':timing,'repetition_sd_s':st.stdev(timing),'R_star':meta.get('R_star'),
            'R_act_total':meta['activation']['R_act_total'],'activation_cell_visits':meta['activation']['activation_cell_visits'],
            'n_solves':meta['activation']['n_solves'],'memory':meta['memory'],'profile':profile['profile'],
            'profile_over_baseline':profile['algorithm_wall_s']/med,'metadata_over_baseline':meta['algorithm_wall_s']/med,
            'profile_activation_equal_metadata':profile['activation']['R_act_total']==meta['activation']['R_act_total'],
            'out_sha256':meta['out_sha256'],'geometry':features(meta['vol_shape'],K,rounds=meta.get('R_star'))})
    policies=[]
    for sid in sorted(set(r['subject'] for r in rows)):
        for m in (1,2,4):
            rr=[r for r in rows if r['subject']==sid and r['copies']==m]
            base=next(r for r in rr if r['K']==1); dd=[r for r in rr if r['K']>1]; fastest=min(dd,key=lambda r:r['time_median_s'])
            for policy,K in [('fixed16',16),('depth_scaled',16*m),('observed_candidate_min',fastest['K'])]:
                r=next(r for r in dd if r['K']==K)
                policies.append({'subject':sid,'copies':m,'policy':policy,'K':K,'dd_over_gpu':r['time_median_s']/base['time_median_s'],
                    'candidate_regret':r['time_median_s']/fastest['time_median_s'],
                    'live_memory_dd_over_gpu':r['memory']['pool_live_peak_bytes']/base['memory']['pool_live_peak_bytes'],
                    'reserved_memory_dd_over_gpu':r['memory']['pool_reserved_peak_bytes']/base['memory']['pool_reserved_peak_bytes']})
    return {'status':'ANALYZED','assessment':'INCONCLUSIVE','validation':validation,'rows':rows,'policy_comparison':policies,
            'policy_model':loso(rows),'interpretation':'All minima are among tested feasible candidates; 3 deliberately selected subjects; mechanism timings intrusive; no population or causal claim.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--run',type=Path,required=True); a=ap.parse_args()
    result=analyze(a.run); writej(a.run/'analysis.json',result)
    if result['rows']:
        with (a.run/'candidate_times.csv').open('w',newline='',encoding='utf-8') as f:
            fields=['subject','copies','K','time_median_s','R_act_total','n_solves','profile_over_baseline','metadata_over_baseline']
            w=csv.DictWriter(f,fields,extrasaction='ignore'); w.writeheader(); w.writerows(result['rows'])
    seal(a.run); print(result['status']); return 0 if result['status']=='ANALYZED' else 2


if __name__=='__main__': raise SystemExit(main())
