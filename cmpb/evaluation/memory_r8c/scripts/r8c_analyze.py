"""Subject-level analysis; incomplete/smoke data cannot support scaling claims.

python -B scripts/r8c_analyze.py --run results/server_RUN
Repetitions are summarized within subject first. GPU event envelopes are not
misreported as kernel times. m=1 K=16m is an explicit alias, never extra data.
"""
import argparse
import math
import statistics as st
from pathlib import Path
from r8c_common import readj, writej, sha, seal
from r8c_validate import validate_run


def slope(x,y):
    xb=st.mean(x); yb=st.mean(y)
    return sum((a-xb)*(b-yb) for a,b in zip(x,y))/sum((a-xb)**2 for a in x)


def analyze(directory):
    d=Path(directory); validation=validate_run(d); rows=[]; grouped={}
    if validation['status']!='PASS':
        return {'schema':'R8C-1','status':'NOT MEASURED / INCOMPLETE','scientific_assessment':'INCONCLUSIVE',
                'validation':validation,'rows':[],'output_differences':[],'paired_ratios':[],'subject_slopes':[],'bootstrap':{},
                'note':'Invalid, mixed or incomplete records are preserved in validation; no scientific statistics computed.'}
    for p in sorted(d.glob('workers/*/record.json')):
        r=readj(p)
        if r.get('purpose') not in ('timing','metadata') or r.get('status')!='COMPLETED': continue
        grouped.setdefault((r['subject'],r['copies'],r['method']),[]).append((r,p))
    for (sid,m,method),rr in sorted(grouped.items()):
        times=[r['t_algorithm_s'] for r,p in rr if r['purpose']=='timing']; metas=[(r,p) for r,p in rr if r['purpose']=='metadata']
        if len(times)!=3 or len(metas)!=1: continue
        r,p=metas[0]; mem=r['memory']
        row={'subject':sid,'copies':m,'method':method,'K':r['K'],'t_median_s':st.median(times),'t_mean_s':st.mean(times),
             't_sd_repetitions_s':st.stdev(times),'pool_reserved_mib':mem['pool_total_peak_bytes']/2**20,
             'pool_live_peak_mib':mem['pool_used_peak_bytes']/2**20,'sampled_device_delta_mib':mem['sampled_device_delta_peak_bytes']/2**20,
             'host_process_hwm_mib':mem['host_hwm_bytes']/2**20,'hwm_reset':mem['hwm_reset'],'host_hwm_scope':mem['host_hwm_scope'],
             'algorithm_sampled_rss_peak_mib':mem['sampled_host_rss_peak_bytes']/2**20,'pool_peak_scope':mem['pool_peak_scope'],
             'R_star':r.get('R_star'),'R_act_total':r['metadata']['R_act_total'],'n_solves':r['metadata']['n_solves'],
             'global_violations':r['verification']['global_violations'],'out_sha256':r['out_sha256'],
             'source':str(p.relative_to(d)),'source_sha256':sha(p)}
        if r.get('output_npy'):
            row['output_npy_path']=str(p.parent/r['output_npy'])
        rows.append(row)
    # Compare the saved metadata outputs with bounded mmap chunks.
    differences=[]
    if rows:
        import numpy as np
        for r in rows:
            if r['method']=='gpu_k1': continue
            k=next((q for q in rows if q['subject']==r['subject'] and q['copies']==r['copies'] and q['method']=='gpu_k1'),None)
            if not k or not k.get('output_npy_path') or not r.get('output_npy_path'): continue
            a=np.load(k['output_npy_path'],mmap_mode='r',allow_pickle=False); b=np.load(r['output_npy_path'],mmap_mode='r',allow_pickle=False)
            if a.shape!=b.shape: raise ValueError('saved comparison shape mismatch')
            n=0; maxdiff=0.0
            for z in range(0,a.shape[2],16):
                aa=a[:,:,z:z+16]; bb=b[:,:,z:z+16]
                n+=int(np.count_nonzero(aa!=bb)); maxdiff=max(maxdiff,float(np.abs(aa-bb).max()))
            differences.append({'subject':r['subject'],'copies':r['copies'],'method':r['method'],'different_cells':n,
                                'fraction_different':n/a.size,'max_abs_diff':maxdiff})
    ratios=[]; subjects=sorted({r['subject'] for r in rows}); sl=[]
    for s in subjects:
        for method in ('ddfp_k16','ddfp_k16m'):
            rr=[]
            for m in (1,2,4):
                k=next((r for r in rows if (r['subject'],r['copies'],r['method'])==(s,m,'gpu_k1')),None)
                dd=next((r for r in rows if (r['subject'],r['copies'],r['method'])==(s,m,'ddfp_k16' if m==1 else method)),None)
                if k and dd:
                    q={'subject':s,'copies':m,'method':method,'time_dd_over_k1':dd['t_median_s']/k['t_median_s'],
                       'reserved_k1_over_dd':k['pool_reserved_mib']/dd['pool_reserved_mib'],
                       'live_k1_over_dd':k['pool_live_peak_mib']/dd['pool_live_peak_mib']}
                    rr.append(q); ratios.append(q)
            if len(rr)==3: sl.append({'subject':s,'method':method,'slope_log_time_ratio_log_m':slope([math.log(q['copies']) for q in rr],[math.log(q['time_dd_over_k1']) for q in rr])})
    bootstrap={}
    if validation['status']=='PASS':
        import numpy as np
        rng=np.random.default_rng(20261016)
        for method in ('ddfp_k16','ddfp_k16m'):
            v=np.array([r['slope_log_time_ratio_log_m'] for r in sl if r['method']==method])
            if len(v)==3:
                b=v[rng.integers(0,3,(10000,3))].mean(axis=1); lo,hi=map(float,np.quantile(b,[.025,.975]))
                bootstrap[method]={'mean_slope':float(v.mean()),'bootstrap_percentile_range_95':[lo,hi], 'n_subjects':3,
                                   'trend':'DECREASING' if hi<0 else 'INCREASING' if lo>0 else 'INCONCLUSIVE',
                                   'scope':'descriptive R8C V1 sensitivity selection; not population inference; frozen R8-A verdicts unchanged'}
    return {'schema':'R8C-1','status':'ANALYZED' if validation['status']=='PASS' else 'NOT MEASURED / INCOMPLETE',
            'scientific_assessment':'INCONCLUSIVE','validation':validation,'rows':rows,'output_differences':differences,
            'paired_ratios':ratios,'subject_slopes':sl,'bootstrap':bootstrap,
            'note':'3 deliberately selected subjects support sensitivity analysis only. No causal attribution from trend alone.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--run',type=Path,required=True); a=ap.parse_args()
    out=analyze(a.run); writej(a.run/'analysis.json',out)
    lines=['# R8-C scaling results','',f'Status: **{out["status"]}**. Scientific assessment: **{out["scientific_assessment"]}**.','',
           '|Subject|m|Method|Time median s|Live pool MiB|Reserved pool MiB|R*|Σ R_act|Solves|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in out['rows']: lines.append(f'|{r["subject"]}|{r["copies"]}|{r["method"]}|{r["t_median_s"]:.4f}|{r["pool_live_peak_mib"]:.1f}|{r["pool_reserved_mib"]:.1f}|{r["R_star"]}|{r["R_act_total"]}|{r["n_solves"]}|')
    if not out['rows']: lines+=['','Primary statistics withheld because the run is incomplete or invalid; see validation.json.']
    lines+=['',out['note'],'','See analysis.json for provenance, paired ratios, full validation and descriptive bootstrap resampling ranges.']
    (a.run/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8'); seal(a.run); print(out['status'])


if __name__=='__main__': main()
