"""Validate R8-C records and arrays, including negative controls in tests.

python -B scripts/r8c_validate.py --run results/server_RUN
Missing, refused, partial or synthetic data can never become a scientific PASS.
"""
import argparse
import hashlib
import math
import re
from pathlib import Path
from r8c_common import readj, writej, sha, obj_sha
from r8c_memory_model import extent, verifier_plan


def safe_readj(path):
    try:
        value=readj(path)
        return value if isinstance(value,dict) else {'_read_error':'JSON object required'}
    except (OSError,ValueError) as e: return {'_read_error':f'{type(e).__name__}: {e}'}


def check_array(u, vol, chunk=16):
    """Independent, bounded host check of all Ispan intervals from corner voxels.

    Eight parity classes cover every expanded cell exactly. For each class,
    min/max over its 1/2/4/8 original corner voxels defines the original interval.
    No full expanded endpoint arrays are built. Original voxels are a separate check.
    """
    import numpy as np
    if tuple(u.shape) != extent(tuple(int(x) for x in vol.shape)) or u.dtype != np.float32:
        return {'ok': False, 'reason': 'shape/dtype mismatch'}
    finite = True; violations = 0; original_diff = 0
    for z in range(0, vol.shape[2], chunk):
        original_diff += int(np.count_nonzero(u[::2, ::2, 2*z:2*min(vol.shape[2], z+chunk):2] != vol[:, :, z:z+chunk]))
    for px in (0,1):
        for py in (0,1):
            for pz in (0,1):
                nx, ny, nz = (vol.shape[i]-p for i,p in enumerate((px,py,pz)))
                for z in range(0, nz, chunk):
                    e = min(nz, z+chunk)
                    lo = vol[:nx, :ny, z:e].copy(); hi = lo.copy()
                    for dx in range(px+1):
                        for dy in range(py+1):
                            for dz in range(pz+1):
                                corner = vol[dx:dx+nx, dy:dy+ny, z+dz:e+dz]
                                np.minimum(lo, corner, out=lo); np.maximum(hi, corner, out=hi)
                    part = u[px:2*nx:2, py:2*ny:2, 2*z+pz:2*e:2]
                    finite = finite and bool(np.isfinite(part).all())
                    violations += int(np.count_nonzero((part < lo) | (part > hi)))
    return {'ok': finite and violations == 0 and original_diff == 0, 'finite': finite,
            'original_voxel_differences': original_diff, 'original_interval_violations': violations,
            'scope': 'all expanded cells, independently recomputed original corner intervals'}


def validate_record(r, root=None):
    reasons = []
    if not isinstance(r,dict): return {'ok':False,'reasons':['record must be a JSON object']}
    if any(k in r and not isinstance(r[k],dict) for k in ('array_check','verification','regression','metadata','memory','verifier_negative_control')):
        return {'ok':False,'reasons':['malformed nested record object']}
    if r.get('schema') != 'R8C-1' or r.get('status') != 'COMPLETED': reasons.append('not a completed R8C-1 record')
    if r.get('backend') != 'cupy' or r.get('cuda_executed') is not True: reasons.append('not real CUDA evidence')
    if r.get('purpose') not in ('timing', 'metadata', 'smoke'): reasons.append('unknown purpose')
    copies=r.get('copies'); method=r.get('method')
    expected_k=1 if method=='gpu_k1' else 16 if method=='ddfp_k16' else 16*copies if method=='ddfp_k16m' and type(copies) is int else None
    if type(copies) is not int or copies not in (1,2,4) or expected_k is None or r.get('K')!=expected_k or type(r.get('rep')) is not int or r['rep'] not in (0,1,2): reasons.append('invalid method/K/copies/rep correspondence')
    if r.get('sources_unchanged') is not True: reasons.append('source preservation missing/failed')
    shape = r.get('vol_shape')
    try:
        es = extent(tuple(shape)); vp = verifier_plan(es)
        if list(es) != r.get('out_shape'): reasons.append('wrong output shape')
    except (ValueError, TypeError): vp = {'ok': False}; reasons.append('malformed volume shape')
    if not isinstance(r.get('out_sha256'), str) or not re.fullmatch('[0-9a-f]{64}', r['out_sha256']): reasons.append('invalid output SHA256')
    for key in ('input_sha256', 'vol_sha256', 'config_sha256'):
        if not re.fullmatch('[0-9a-f]{64}', str(r.get(key, ''))): reasons.append('missing/invalid '+key)
    if r.get('array_check', {}).get('ok') is not True: reasons.append('array constraints failed or missing')
    v = r.get('verification', {})
    if not vp['ok']:
        if v.get('verify_status') != 'REFUSED_INT32' or any(k in v for k in ('global_dwc_ok','global_violations')): reasons.append('unsafe global certificate')
        reasons.append('global certificate unavailable')
    elif not (v.get('verify_status') == 'DONE' and v.get('global_dwc_ok') is True and v.get('global_violations') == 0 and v.get('nan_skipped') == 0):
        reasons.append('global DWC failed or missing')
    if r.get('method') != 'gpu_k1':
        if v.get('seam_violations') != 0: reasons.append('seam failed or missing')
        rr, mr = r.get('R_star'), r.get('max_rounds')
        if type(rr) is not int or type(mr) is not int or not 1 <= rr < mr: reasons.append('termination unconfirmed at round cap')
        if r.get('purpose') in ('metadata','smoke'):
            if r.get('metadata', {}).get('n_solves') != r.get('K',0)*(1+(rr or 0)): reasons.append('solve-count mismatch')
            changes=r.get('metadata',{}).get('round_changes',[])
            if not isinstance(changes,list) or not changes or any(not isinstance(q,dict) for q in changes):
                reasons.append('round change history missing')
            elif [q.get('round') for q in changes]!=list(range(1,(rr or 0)+1)) or type(changes[-1].get('max_change')) not in (float,int) or not 0<=changes[-1]['max_change']<0.5:
                reasons.append('recorded stopping condition not confirmed')
    if copies==1 or r.get('purpose')=='smoke':
        if r.get('regression', {}).get('ok') is not True: reasons.append('regression comparison failed or missing')
    elif r.get('regression', {}).get('status')!='NOT_AVAILABLE': reasons.append('larger-size regression status missing')
    if r.get('purpose') == 'smoke' and r.get('cpu_independent_dwc_violations') != 0: reasons.append('independent CPU smoke DWC failed/missing')
    if r.get('purpose') == 'smoke' and r.get('verifier_negative_control',{}).get('detected') is not True: reasons.append('smoke verifier negative control failed/missing')
    for k in ('t_algorithm_s','wall_s'):
        if type(r.get(k)) not in (float,int) or not math.isfinite(r[k]) or r[k] <= 0: reasons.append('invalid time '+k)
    if r.get('purpose') in ('metadata', 'smoke'):
        m = r.get('memory', {})
        if m.get('sampler_errors') != [] or type(m.get('n_samples')) is not int or m.get('n_samples',0) < 1: reasons.append('memory/resource monitor failed')
        for k in ('pool_used_peak_bytes','pool_total_peak_bytes','host_hwm_bytes','baseline_device_used_bytes','sampled_host_rss_peak_bytes','sampled_device_delta_peak_bytes'):
            if type(m.get(k)) not in (float,int) or not math.isfinite(m[k]) or m[k] < 0: reasons.append('missing memory metric '+k)
        if any(type(m.get(k)) not in (int,float) or not math.isfinite(m[k]) or m[k]<=0 for k in ('pool_total_peak_bytes','pool_used_peak_bytes')): reasons.append('nonpositive pool metric')
        if type(m.get('hwm_reset')) is not bool or not isinstance(m.get('host_hwm_scope'),str) or not isinstance(m.get('pool_peak_scope'),str): reasons.append('memory scope missing')
        meta=r.get('metadata',{})
        if type(meta.get('n_solves')) is not int or meta.get('n_solves',0)<1 or type(meta.get('R_act_total')) is not int or meta.get('R_act_total',-1)<0:
            reasons.append('activation metadata missing')
        if method=='gpu_k1' and meta.get('n_solves')!=1: reasons.append('K1 solve-count mismatch')
    if r.get('purpose')=='metadata' and (r.get('output_npy')!='output.npy' or not re.fullmatch('[0-9a-f]{64}',str(r.get('output_file_sha256','')))):
        reasons.append('metadata saved output/hash required')
    if root:
        envp=Path(root)/'environment.json'
        if not envp.is_file() or sha(envp)!=r.get('environment_sha256'): reasons.append('environment provenance missing/mismatch')
        elif safe_readj(envp).get('ok') is not True or safe_readj(envp).get('config_sha256')!=r.get('config_sha256'): reasons.append('environment gate/config did not pass')
        if r.get('output_npy') == 'output.npy':
            p = Path(root) / r['output_npy']
            if not p.is_file() or sha(p) != r.get('output_file_sha256'): reasons.append('saved output hash mismatch')
            else:
                import numpy as np
                try:
                    u=np.load(p,mmap_mode='r',allow_pickle=False)
                    if u.dtype!=np.float32 or list(u.shape)!=r.get('out_shape') or not u.flags.c_contiguous:
                        reasons.append('saved output dtype/shape/order mismatch')
                    elif hashlib.sha256(memoryview(u).cast('B')).hexdigest()!=r['out_sha256']: reasons.append('saved array data hash mismatch')
                except (ValueError,OSError,TypeError) as e: reasons.append('saved NPY unreadable: '+str(e))
    return {'ok': not reasons, 'reasons': reasons}


def validate_run(directory):
    d = Path(directory); files = sorted(d.glob('workers/*/record.json')); rr = [safe_readj(p) for p in files]
    checks = [{'file': str(p.relative_to(d)), **validate_record(r, p.parent)} for p,r in zip(files,rr)]
    bad = [c for c in checks if not c['ok']]
    if bad:
        return {'status':'FAIL_OR_INCOMPLETE','n_records':len(files),'n_primary_records':sum(r.get('purpose') in ('timing','metadata') for r in rr),
                'n_expected':96,'complete':False,'deterministic':False,'smoke_gate_ok':False,'stopped':(d/'STOPPED.json').exists(),
                'provenance_errors':['one or more records failed validation; aggregate evaluation withheld'],'missing':[],'checks':checks}
    # Smoke records are required for execution but do not count towards primary coverage.
    primary = [r for r in rr if r.get('purpose') in ('timing','metadata')]
    expected = set(); observed = set(); deterministic = True
    manifest = safe_readj(d/'run_config.json') if (d/'run_config.json').is_file() else {}
    provenance=[]
    subjects=manifest.get('subjects',[])
    if not isinstance(subjects,list) or any(not isinstance(s,str) or not s for s in subjects): subjects=[]
    if len(set(subjects))!=3: provenance.append('expected three selected subjects')
    config=manifest.get('config',{})
    if not config or obj_sha(config)!=manifest.get('config_sha256'): provenance.append('manifest config hash mismatch')
    for r,p in zip(rr,files):
        if r.get('config_sha256')!=manifest.get('config_sha256'): provenance.append('worker/config mismatch: '+p.parent.name)
        if 'max_rounds' in config and r.get('max_rounds')!=config['max_rounds']: provenance.append('worker/round cap mismatch: '+p.parent.name)
        ep=p.parent/'environment.json'
        if ep.is_file():
            env=safe_readj(ep)
            if env.get('sources')!=manifest.get('source_hashes'): provenance.append('worker source fingerprint mismatch: '+p.parent.name)
            if r.get('purpose')!='smoke':
                inv=env.get('inventory',{}).get(r.get('subject'),{})
                if inv.get('sha256')!=r.get('input_sha256'): provenance.append('worker input fingerprint mismatch: '+p.parent.name)
                base_shape=inv.get('vol_shape',[])
                if len(base_shape)!=3 or r.get('vol_shape')!=[base_shape[0],base_shape[1],base_shape[2]*r['copies']]:
                    provenance.append('worker tiling shape mismatch: '+p.parent.name)
    smoke=[r for r in rr if r.get('purpose')=='smoke']
    smoke_keys={(r.get('method'),r.get('rep')) for r in smoke}
    smoke_expected={(m,i) for m in ('gpu_k1','ddfp_k16') for i in (0,1)}
    smoke_ok=len(smoke)==4 and smoke_keys==smoke_expected and all(len({r['out_sha256'] for r in smoke if r['method']==m})==1 for m in ('gpu_k1','ddfp_k16'))
    sg=safe_readj(d/'smoke_gate.json') if (d/'smoke_gate.json').is_file() else {}
    smoke_records=sorted([[str(r.get('method')),r.get('rep') if type(r.get('rep')) is int else -1,str(r.get('out_sha256'))] for r in smoke])
    smoke_ok=smoke_ok and sg.get('status')=='PASS' and sorted(sg.get('records',[]))==smoke_records
    for s in subjects:
        for m in (1,2,4):
            for method in ('gpu_k1','ddfp_k16','ddfp_k16m'):
                if m == 1 and method == 'ddfp_k16m': continue
                expected.add((s,m,method,'metadata',0))
                for rep in range(3): expected.add((s,m,method,'timing',rep))
    hashgroups = {}
    for r in primary:
        key = (r.get('subject'),r.get('copies'),r.get('method'),r.get('purpose'),r.get('rep'))
        observed.add(key); hashgroups.setdefault(key[:3],set()).add(r.get('out_sha256'))
    deterministic = all(len(h) == 1 for h in hashgroups.values())
    complete = bool(expected) and expected == observed and len(primary) == len(observed)
    stop = (d/'STOPPED.json').exists()
    return {'status': 'PASS' if files and not bad and complete and deterministic and not stop and smoke_ok and not provenance else 'INCONCLUSIVE' if not files else 'FAIL_OR_INCOMPLETE',
            'n_records': len(files), 'n_primary_records': len(primary), 'n_expected': len(expected), 'complete': complete,
            'deterministic': deterministic, 'smoke_gate_ok':smoke_ok, 'provenance_errors':provenance,
            'stopped': stop, 'missing': sorted(expected-observed), 'checks': checks}


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--run',type=Path,required=True); a=ap.parse_args()
    out=validate_run(a.run); writej(a.run/'validation.json',out); print(out['status'])
    return 0 if out['status']=='PASS' else 2


if __name__=='__main__': raise SystemExit(main())
