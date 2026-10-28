from __future__ import annotations
import argparse, csv, glob, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np


def parse_configs(s): return [tuple(map(int,x.split(':'))) for x in s.split(',') if x.strip()]
def read_csvs(pattern):
    rows=[]; files=sorted(glob.glob(pattern))
    if not files: raise SystemExit(f'No CSVs match {pattern}')
    for p in files:
        with open(p, newline='', encoding='utf-8') as f: rows.extend(csv.DictReader(f))
    return files, rows

def wilson(k,n,z=1.96):
    if n==0:return [None,None]
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    m=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [100*max(0,c-m),100*min(1,c+m)]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv-glob', required=True)
    ap.add_argument('--failure-glob', required=True)
    ap.add_argument('--configs', default='16:1')
    ap.add_argument('--expected-subjects', type=int, default=1251)
    ap.add_argument('--output', required=True)
    ap.add_argument('--summary', required=True)
    a=ap.parse_args(); configs=parse_configs(a.configs)
    files,rows=read_csvs(a.csv_glob)
    keys=[(r['subject_id'],int(r['K']),int(r['delta'])) for r in rows]
    if len(keys)!=len(set(keys)): raise SystemExit('Duplicate subject/K/delta keys across shards')
    expected=a.expected_subjects*len(configs)
    if len(rows)!=expected: raise SystemExit(f'Row count mismatch: {len(rows)} != {expected}')
    subjects={r['subject_id'] for r in rows}
    if len(subjects)!=a.expected_subjects: raise SystemExit(f'Subject count mismatch: {len(subjects)}')
    failures=[]
    for p in sorted(glob.glob(a.failure_glob)):
        with open(p,encoding='utf-8') as f:
            for i,line in enumerate(f):
                if i==0 and line.lower().startswith('subject_id'): continue
                if line.strip(): failures.append((p,line.strip()))
    if failures: raise SystemExit(f'Nonempty failure files: {len(failures)} records')
    rows.sort(key=lambda r:(r['subject_id'],int(r['K']),int(r['delta'])))
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    grp=defaultdict(list)
    for r in rows: grp[(int(r['K']),int(r['delta']))].append(r)
    summary={'expected_subjects':a.expected_subjects,'actual_subjects':len(subjects),'expected_rows':expected,'actual_rows':len(rows),'source_csvs':files,'timing_policy':'time_s retained for operational provenance only; do not use for TIPS/runtime benchmark','configs':[]}
    for cfg in configs:
        rr=grp[cfg]
        R=np.array([int(r['R_star']) for r in rr]); B=np.array([int(r['bdry_viol']) for r in rr]); F=np.array([int(r['chi_sign_flip']) for r in rr]); REF=np.array([int(r['ref_chi']) for r in rr]); CHI=np.array([int(r['ddfp_chi']) for r in rr]); n=len(rr); p=int((B==0).sum())
        formula=int(sum(int(r['chi_sign_flip']) != int((int(r['ref_chi']) != 0) and (int(r['ddfp_chi'])*int(r['ref_chi']) < 0)) for r in rr))
        x={'K':cfg[0],'delta':cfg[1],'n':n,'dwc_pass':p,'dwc_fail':n-p,'dwc_pass_pct':100*p/n,'dwc_wilson95_pct':wilson(p,n),'R_eq_2':int((R==2).sum()),'R_eq_2_pct':100*float((R==2).sum())/n,'R_le_2':int((R<=2).sum()),'R_gt_2':int((R>2).sum()),'R_mean':float(R.mean()),'R_median':float(np.median(R)),'R_min':int(R.min()),'R_max':int(R.max()),'bdry_viol_sum':int(B.sum()),'bdry_viol_max':int(B.max()),'chi_flip_n':int(F.sum()),'chi_flip_pct':100*float(F.sum())/n,'ref_chi_zero_n':int((REF==0).sum()),'ddfp_chi_min':int(CHI.min()),'ddfp_chi_max':int(CHI.max()),'chi_formula_mismatch':formula}
        summary['configs'].append(x)
    Path(a.summary).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print('P6_K16_CHIFLIP_MERGE_PASS')
if __name__=='__main__': main()
