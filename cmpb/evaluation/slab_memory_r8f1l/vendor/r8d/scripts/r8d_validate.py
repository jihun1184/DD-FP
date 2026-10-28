"""Fail-closed validation of new R8-D workers and full candidate coverage.

Run evidence and output validity validation requires actual CUDA, complete separate-process passes,
saved metadata outputs, environment/config/source provenance, full intervals,
global DWC and separate seam checks. Different K outputs need not be equal.
"""
import argparse
import math
import re
from pathlib import Path
from r8d_common import ROOT, KS, C, arr_sha, obj_sha, readj, sha, writej, extent, verifier_plan
from r8c_validate import check_array


def profile_errors(profile,activation):
    why=[]; scopes=profile.get('nested_inclusive_spans',{})
    if profile.get('profile_intrusive') is not True: why.append('profile missing')
    for label in ('kernel:activate_level','kernel:expand_frontier'):
        if scopes.get(label,{}).get('calls')!=activation.get('R_act_total'):
            why.append(label+' activation-count mismatch')
    if scopes.get('fp_api',{}).get('calls')!=activation.get('n_solves'): why.append('FP profile coverage mismatch')
    if not scopes.get('build_ispan_gpu',{}).get('calls'): why.append('build profile missing')
    return why


def record(r,folder):
    why=[]
    if r.get('schema')!='R8D-1' or r.get('status')!='COMPLETED': why.append('not completed R8D')
    if r.get('backend')!='cupy' or r.get('cuda_executed') is not True: why.append('not real CUDA')
    K=r.get('K'); purpose=r.get('purpose'); m=r.get('copies')
    if purpose not in ('timing','profile','metadata','smoke'): why.append('unknown purpose')
    if type(K) is not int or K<1 or r.get('method')!=('gpu_k1' if K==1 else 'ddfp'): why.append('algorithm/K mismatch')
    if purpose!='smoke' and (m not in (1,2,4) or K not in [1]+KS.get(str(m),[])): why.append('unregistered candidate')
    for k in ('out_sha256','input_sha256','vol_sha256','config_sha256'):
        if not re.fullmatch('[a-f0-9]{64}',str(r.get(k,''))): why.append('invalid '+k)
    if r.get('sources_unchanged') is not True: why.append('source preservation missing')
    try:
        es=extent(tuple(r['vol_shape']))
        if list(es)!=r.get('out_shape') or not verifier_plan(es)['ok']: why.append('shape/int32 unsafe')
    except Exception: why.append('invalid shape')
    if r.get('array_check',{}).get('ok') is not True: why.append('interval/original values unconfirmed')
    v=r.get('verification',{})
    if not (v.get('verify_status')=='DONE' and v.get('global_dwc_ok') is True and v.get('global_violations')==0 and v.get('nan_skipped')==0): why.append('global DWC missing/failed')
    if K!=1:
        if v.get('seam_violations')!=0: why.append('seam missing/failed')
        if not (type(r.get('R_star')) is int and 1<=r['R_star']<r.get('max_rounds',0)): why.append('convergence strictly before cap unconfirmed')
    for k in ('algorithm_wall_s','warmup_wall_s'):
        if type(r.get(k)) not in (int,float) or not math.isfinite(r[k]) or r[k]<=0: why.append('invalid time '+k)
    trace=r.get('activation',{})
    if purpose in ('profile','metadata','smoke'):
        expected=1 if K==1 else K*(1+r.get('R_star',0))
        if trace.get('n_solves')!=expected or len(trace.get('solves',[]))!=expected: why.append('solve coverage mismatch')
        if any(type(s.get('R_act')) is not int or s['R_act']<0 or s.get('nan_count')!=0 for s in trace.get('solves',[])): why.append('invalid activation trace')
        solves=trace.get('solves',[])
        if trace.get('R_act_total')!=sum(s.get('R_act',0) for s in solves): why.append('aggregate activation mismatch')
        if trace.get('activation_cell_visits')!=sum(s.get('R_act',0)*s.get('padded_cells',0) for s in solves): why.append('aggregate sweep proxy mismatch')
        from r8c_memory_model import slab_shapes
        try:
            shapes=slab_shapes(tuple(r['vol_shape']),K,1) if K!=1 else [r['vol_shape']]
            padded=[[2*x+1 for x in sh] for sh in shapes]
            for i,s in enumerate(solves):
                p=padded[i%len(padded)]
                if s.get('padded_shape')!=p or s.get('padded_cells')!=math.prod(p): why.append('solve geometry mismatch'); break
        except Exception: why.append('solve geometry unreadable')
        if K!=1:
            changes=trace.get('round_changes',[])
            if [q.get('round') for q in changes]!=list(range(1,r.get('R_star',0)+1)) or not changes or not 0<=changes[-1].get('max_change',-1)<.5:
                why.append('stop predicate missing/failed')
    if purpose in ('metadata','smoke'):
        mem=r.get('memory',{})
        if mem.get('errors')!=[] or mem.get('samples',0)<1: why.append('memory monitor failed')
        for k in ('pool_live_peak_bytes','pool_reserved_peak_bytes','sampled_host_rss_peak_bytes'):
            if type(mem.get(k)) not in (int,float) or not math.isfinite(mem[k]) or mem[k]<=0: why.append('missing/nonfinite memory '+k)
        if mem.get('pool_reserved_peak_bytes',0)<mem.get('pool_live_peak_bytes',0): why.append('reserved smaller than live')
    if purpose=='profile' or purpose=='smoke' and r.get('smoke_profile'):
        why.extend(profile_errors(r.get('profile',{}),trace))
        if any(s.get('original_slab_constraints',{}).get('ok') is not True for s in trace.get('solves',[])):
            why.append('individual FP original voxel/interval contract missing/failed')
    if purpose in ('profile','metadata','smoke'):
        if any(s.get('seeded') and (s.get('seed_raw_bits_preserved') is not True or not isinstance(s.get('seed_count'),int) or s['seed_count']<1) for s in trace.get('solves',[])):
            why.append('seed raw preservation unconfirmed')
    if purpose=='smoke':
        if not r.get('cpu_reference_equal') or r.get('cpu_dwc_violations')!=0 or not r.get('negative_control_detected'): why.append('CPU smoke/negative control failed')
    reg=r.get('regression',{})
    if reg.get('expected') and reg.get('ok') is not True: why.append('known R7 output regression')
    if purpose!='smoke' and m==1 and K in (1,16):
        if not re.fullmatch('[a-f0-9]{64}',str(reg.get('expected',''))) or reg.get('ok') is not True or reg.get('expected')!=r.get('out_sha256'):
            why.append('mandatory m1 known-output regression missing/mismatched')
    for filename,key in [('environment.json','environment_sha256'),('admission.json','admission_sha256')]:
        p=Path(folder)/filename
        if not p.is_file() or sha(p)!=r.get(key): why.append('missing/failing provenance '+filename)
        elif readj(p).get('ok') is not True: why.append('gate refused '+filename)
    if purpose=='metadata':
        p=Path(folder)/'output.npy'
        if not p.is_file() or sha(p)!=r.get('output_file_sha256'): why.append('saved output missing/hash')
        else:
            import numpy as np
            u=np.load(p,mmap_mode='r',allow_pickle=False)
            if list(u.shape)!=r.get('out_shape') or u.dtype!=np.float32 or arr_sha(u)!=r['out_sha256']: why.append('array payload hash/shape/dtype')
    return {'ok':not why,'reasons':why}


def validate_run(d):
    d=Path(d)
    if not (d/'run_config.json').exists(): return {'status':'NOT EXECUTED','reasons':['run manifest absent']}
    manifest=readj(d/'run_config.json'); errors=[]; checks=[]; observed=[]; groups={}; smokes=[]
    c=manifest['config']
    if obj_sha(c)!=manifest['config_sha256']: errors.append('manifest hash mismatch')
    subjects=manifest.get('subjects',[])
    registered=['BraTS2021_01363','BraTS2021_00158','BraTS2021_01628']
    if subjects!=registered: errors.append('subject selection differs from frozen E6 rule')
    if c.get('k_candidates')!=KS or c.get('copies')!=[1,2,4] or c.get('reps')!=3 or c.get('delta')!=1 or c.get('max_rounds')!=8:
        errors.append('registered design differs')
    expected={(s,m,K,purpose,rep) for s in registered for m in (1,2,4) for K in [1]+KS[str(m)]
              for purpose,rep in [('timing',i) for i in range(3)]+[('profile',0),('metadata',0)]}
    planned=[(j.get('subject'),j.get('copies'),j.get('K'),j.get('purpose'),j.get('rep')) for j in manifest.get('jobs',[])]
    if len(planned)!=len(expected) or set(planned)!=expected: errors.append('manifest plan differs from independent 225-job design')
    for p in sorted(d.glob('workers/*/record.json')):
        try:
            r=readj(p); check=record(r,p.parent)
            env=readj(p.parent/'environment.json')
            if env.get('sources')!=manifest.get('sources') or r.get('config_sha256')!=manifest['config_sha256']:
                check['ok']=False; check['reasons'].append('manifest source/config mismatch')
            if env.get('config_sha256')!=manifest['config_sha256'] or r.get('max_rounds')!=c.get('max_rounds'):
                check['ok']=False; check['reasons'].append('environment/round-cap fingerprint mismatch')
            if r['purpose']=='smoke': smokes.append(r)
            else:
                entry=env.get('inventory',{}).get(r.get('subject'),{})
                shape=entry.get('vol_shape',[])
                if entry.get('sha256')!=r.get('input_sha256') or len(shape)!=3 or r.get('vol_shape')!=[shape[0],shape[1],shape[2]*r.get('copies',0)]:
                    check['ok']=False; check['reasons'].append('input SHA/tiling shape provenance mismatch')
                key=(r['subject'],r['copies'],r['K'],r['purpose'],r['rep']); observed.append(key)
                groups.setdefault(key[:3],set()).add(r['out_sha256'])
            checks.append({'path':p.relative_to(d).as_posix(),**check})
        except Exception as e: checks.append({'path':str(p),'ok':False,'reasons':[str(e)]})
    if not expected or set(observed)!=expected or len(observed)!=len(expected): errors.append('primary coverage incomplete/duplicate')
    if any(len(v)!=1 for v in groups.values()): errors.append('output nondeterminism across independent passes')
    smoke_expected={(K,rep) for K in (1,8,16,32,64,128) for rep in (0,1)}
    if len(smokes)!=12 or {(r['K'],r['rep']) for r in smokes}!=smoke_expected: errors.append('GPU/CPU smoke coverage incomplete')
    for K in (1,8,16,32,64,128):
        if len({r['out_sha256'] for r in smokes if r['K']==K})!=1: errors.append('smoke nondeterministic')
        if not any(r.get('smoke_profile') is True and r['K']==K for r in smokes): errors.append('profile smoke missing')
    if any(not q['ok'] for q in checks): errors.append('worker failed validation')
    if (d/'STOPPED.json').exists(): errors.append('run stopped')
    return {'status':'PASS' if not errors else 'FAIL_OR_INCOMPLETE','reasons':errors,'checks':checks,
            'n_expected':len(expected),'n_observed':len(observed),'missing':[list(k) for k in sorted(expected-set(observed))]}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--run',type=Path,required=True); a=ap.parse_args()
    result=validate_run(a.run); writej(a.run/'validation.json',result); print(result['status'])
    return 0 if result['status']=='PASS' else 2


if __name__=='__main__': raise SystemExit(main())
