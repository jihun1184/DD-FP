"""Code-derived features and exploratory grouped predictive validation.

For a solve with padded cells P and activation rounds A, frozen activate and
expand each launch ceil(P/256)*256 threads per activation. A is observed, not
available a priori. Prospective models use only input geometry and K; the
mechanism fit may use A and must never masquerade as a pre-run selector.
"""
import argparse
import math
import statistics as st
from r8d_common import ROOT, readj, writej, model, slab_shapes, extent


def features(shape,K,delta=1,rounds=None):
    slabs=slab_shapes(tuple(shape),K,delta)
    P=[math.prod(tuple(2*x+1 for x in s)) for s in slabs]
    N=math.prod(extent(tuple(shape)))
    depths=[s[2] for s in slabs]; d=shape[2]
    return {'N':N,'D':d,'K':K,'core_depth_min':d//K,'core_depth_max':math.ceil(d/K),
        'halo_depth_sum':sum(depths)-d,'slab_padded_cells_sum':sum(P),'slab_padded_cells_max':max(P),
        'padded_overlap_ratio':sum(P)/math.prod(tuple(2*x+1 for x in shape)),
        'solves_if_rounds_known':None if rounds is None else K*(1+rounds),
        'all_solve_padded_cells_if_rounds_known':None if rounds is None else (1+rounds)*sum(P)}


def prospective_x(row):
    f=features(row['vol_shape'],row['K'])
    # Independent columns embody global arrays, local depth, and K/halo cost.
    return [1.0,f['N']/1e8,f['N']*f['D']/f['K']/1e9,f['K']/64]


def loso(rows):
    import numpy as np
    subjects=sorted({r['subject'] for r in rows})
    if len(subjects)<3: return {'status':'NOT IDENTIFIED','reason':'at least 3 subjects needed'}
    folds=[]
    for subject in subjects:
        train=[r for r in rows if r['subject']!=subject and r['K']>1]
        test=[r for r in rows if r['subject']==subject and r['K']>1]
        X=np.array([prospective_x(r) for r in train]); y=np.array([r['time_median_s'] for r in train])
        if len(train)<8 or np.linalg.matrix_rank(X)<4:
            return {'status':'NOT IDENTIFIED','reason':'insufficient geometry variation or rank'}
        beta=np.linalg.lstsq(X,y,rcond=None)[0]; preds=X@beta
        for m in sorted({r['copies'] for r in test}):
            group=[r for r in test if r['copies']==m]
            estimates=[float(np.array(prospective_x(r))@beta) for r in group]
            if any(p<=0 for p in estimates): return {'status':'NOT IDENTIFIED','reason':'nonphysical predictions'}
            best=group[min(range(len(group)),key=lambda i:estimates[i])]
            fastest=min(r['time_median_s'] for r in group)
            folds.append({'held_out_subject':subject,'copies':m,'selected_candidate_K':best['K'],
                'regret_ratio':best['time_median_s']/fastest,'candidate_K':[r['K'] for r in group],
                'predicted_s':estimates,'observed_s':[r['time_median_s'] for r in group],
                'train_subjects':sorted(set(r['subject'] for r in train)),'coefficients':beta.tolist()})
    errors=[abs(p/o-1) for f in folds for p,o in zip(f['predicted_s'],f['observed_s'])]
    acceptable=bool(errors) and st.median(errors)<=.15 and max(f['regret_ratio'] for f in folds)<=1.10
    return {'status':'EXPLORATORY_CANDIDATE_RULE' if acceptable else 'NO_RULE_RECOMMENDED',
        'folds':folds,'median_absolute_relative_prediction_error':st.median(errors),
        'exploratory_acceptance':'median absolute relative error <=15%, max candidate regret <=10%; registered before D3',
        'scope':'held-out subject; tested candidate set only, memory/correctness feasibility checked separately; no population significance'}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--audit',type=str,default=str(ROOT/'results/d0_audit.json'))
    ap.add_argument('--out',default=str(ROOT/'results/cost_features.json')); a=ap.parse_args()
    audit=readj(a.audit); rows=[]
    for r in audit['rows']:
        # Shapes are read from the archived record, never inferred from a paper.
        rec=readj(ROOT/r['source']); shape=rec['vol_shape']
        rows.append({**r,'vol_shape':shape,**features(shape,r['K'],rounds=r['R_star'])})
    writej(a.out,{'kind':'R8C_RECORD_REANALYSIS','rows':rows,
        'prospective_rule_status':'NOT IDENTIFIED: D3 controlled candidate sweep not executed',
        'mechanism_scope':'activation_cell_visits is measured-metadata full-grid sweep proxy, not a hardware instruction count'})
    print('wrote',a.out)


if __name__=='__main__': main()
