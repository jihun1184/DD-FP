"""Controller: preflight -> independent GPU smoke -> primary scaling workers.

python -B scripts/r8c_scaling_runner.py --config config/server.json --out results/server_RUN --preflight
python -B scripts/r8c_scaling_runner.py --config config/server.json --out results/server_RUN --execute [--resume]
Every (subject,size,method,pass,rep) uses a fresh subprocess. Failure stops ALL
later jobs; restart from a STOPPED run requires a new output directory.
"""
import argparse
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from r8c_common import HERE, load_config, output_path, gate, selection, readj, writej, sha, obj_sha, smi, admit, seal
from r8c_validate import validate_record, validate_run


def execute_child(c, config_path, out, job):
    """Only terminate our own spawned child on timeout/interrupt/contamination."""
    folder=out/'workers'/job['id']
    if folder.exists():
        rec=folder/'record.json'
        if rec.is_file():
            previous=readj(rec)
            if validate_record(previous,folder)['ok'] and previous['config_sha256']==obj_sha(c) and all(previous.get(k)==job[k] for k in ('subject','copies','method','purpose','rep')):
                return previous
        raise RuntimeError(f'incomplete/invalid existing worker; use new run directory: {folder}')
    command=[c['python'],'-B',str(HERE/'scripts/r8c_gpu_memory_probe.py'),'--config',str(config_path),'--out',str(folder),
             '--purpose',job['purpose'],'--method',job['method'],'--subject',job['subject'],'--copies',str(job['copies']),'--rep',str(job['rep'])]
    logs=out/'logs'; logs.mkdir(exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             CUDA_VISIBLE_DEVICES=str(c['gpu_index']), MPLCONFIGDIR=str(out/'mplconfig'),CUPY_CACHE_DIR=str(out/'cupy_cache'))
    with (logs/(job['id']+'.log')).open('x',encoding='utf-8') as log:
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        started=time.monotonic(); why=None
        try:
            while p.poll() is None:
                if time.monotonic()-started>c['worker_timeout_s']: why='WORKER_TIMEOUT'; break
                if (out/'STOP_REQUESTED').exists(): why='USER_STOP_REQUESTED'; break
                # Controller samples only resource inventory, never algorithm arrays.
                # The worker is expected on this GPU; any OTHER compute pid aborts.
                observed=smi(c)
                if any(q['pid']!=p.pid for q in observed['processes']): why='FOREIGN_GPU_PROCESS'; break
                time.sleep(.5)
            if why: raise RuntimeError(why)
        except BaseException:
            if p.poll() is None:
                p.terminate()
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired: p.kill(); p.wait()
            if folder.exists() and not (folder/'record.json').exists():
                writej(folder/'record.json',{'schema':'R8C-1','status':'INTERRUPTED','reason':why or 'controller interrupted',**job})
            raise
    if p.returncode: raise RuntimeError(f'worker {job["id"]} failed rc={p.returncode}; see logs')
    r=readj(folder/'record.json')
    if not validate_record(r,folder)['ok'] or any(r.get(k)!=job[k] for k in ('subject','copies','method','purpose','rep')):
        raise RuntimeError(f'worker validation/job identity failed: {job["id"]}')
    return r


def plan_jobs(c, subjects):
    rng=random.Random(c['order_seed']); jobs=[]
    for copies in c['copies']:
        combinations=[(s,m) for s in subjects for m in ('gpu_k1','ddfp_k16','ddfp_k16m') if not(copies==1 and m=='ddfp_k16m')]
        rng.shuffle(combinations)
        for s,method in combinations:
            for purpose,rep in [('timing',i) for i in range(c['reps'])]+[('metadata',0)]:
                jobs.append({'id':f'{s}_m{copies}_{method}_{purpose}_{rep}','subject':s,'copies':copies,'method':method,'purpose':purpose,'rep':rep})
    return jobs


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--out',type=Path,required=True)
    action=ap.add_mutually_exclusive_group(required=True); action.add_argument('--preflight',action='store_true'); action.add_argument('--execute',action='store_true')
    ap.add_argument('--resume',action='store_true'); a=ap.parse_args()
    c=load_config(a.config); out=output_path(a.out,c); out.mkdir(parents=True,exist_ok=True)
    if (out/'STOPPED.json').exists(): raise SystemExit('STOPPED run is immutable: correct cause and choose a new run directory')
    try:
        env=gate(c); writej(out/'preflight.json',env)
        if not env['ok']:
            print('NOT EXECUTED: '+'; '.join(env['reasons'])); return 2
        subjects=selection(c); jobs=plan_jobs(c,subjects)
        # Estimate retained .npy disk budget for ALL primary metadata passes.
        ne=lambda sh: (2*sh[0]-1)*(2*sh[1]-1)*(2*sh[2]-1)
        disk_need=sum(4*ne([*env['inventory'][j['subject']]['vol_shape'][:2],env['inventory'][j['subject']]['vol_shape'][2]*j['copies']]) for j in jobs if j['purpose']=='metadata')+2**31
        if shutil.disk_usage(out).free<disk_need: raise RuntimeError(f'insufficient free disk: need {disk_need/2**30:.2f} GiB')
        manifest={'schema':'R8C-1','config':c,'config_sha256':obj_sha(c),'subjects':subjects,'jobs':jobs,'source_hashes':env['sources'],
                  'e6_rows_sha256':sha(c['e6_rows']),'metadata_sha256':sha(c['metadata']),'disk_budget_bytes':disk_need}
        if (out/'run_config.json').exists():
            old=readj(out/'run_config.json')
            if old!=manifest: raise RuntimeError('resume fingerprint differs (config/source/inputs/plan)')
            if a.execute and not a.resume: raise RuntimeError('run exists: use --resume after preflight or interruption')
        else: writej(out/'run_config.json',manifest,True)
        if a.preflight:
            print(f'PREFLIGHT PASS: {len(jobs)} primary workers; disk budget {disk_need/2**30:.2f} GiB; no solver executed'); return 0
        # Each smoke path is independently checked against CPU and original IBI assembly.
        smoke=[]
        for method in ('gpu_k1','ddfp_k16'):
            for rep in (0,1):
                job={'id':f'SMOKE_{method}_{rep}','subject':'SMOKE','copies':1,'method':method,'purpose':'smoke','rep':rep}
                smoke.append(execute_child(c,a.config.resolve(),out,job))
            if smoke[-1]['out_sha256']!=smoke[-2]['out_sha256']: raise RuntimeError('smoke nondeterministic output')
        writej(out/'smoke_gate.json',{'status':'PASS','records':[(r['method'],r['rep'],r['out_sha256']) for r in smoke]})
        hashes={}; completed=[]
        for i,job in enumerate(jobs):
            shape=env['inventory'][job['subject']]['vol_shape'].copy(); shape[2]*=job['copies']
            K=1 if job['method']=='gpu_k1' else 16 if job['method']=='ddfp_k16' else 16*job['copies']
            admission=admit(c,tuple(shape),K)
            if not admission['ok']: raise RuntimeError('ADMISSION_REFUSED: '+'; '.join(admission['reasons']))
            # Previous observed reservation and RSS provide an additional admission floor.
            for prev in completed:
                if prev['method']!=job['method'] or prev['purpose']!='metadata': continue
                scale=job['copies']/prev['copies']
                if scale<1: continue
                if 1.15*prev['memory']['pool_total_peak_bytes']*scale/2**20>admission['vram_limit_mib']:
                    raise RuntimeError('observed-pool extrapolation exceeds GPU budget')
                if 1.15*prev['memory']['host_hwm_bytes']*scale>c['host_fraction']*admission['host']['MemAvailable']:
                    raise RuntimeError('observed-RSS extrapolation exceeds host budget')
            print(f'[{i+1}/{len(jobs)}] {job["id"]}',flush=True)
            r=execute_child(c,a.config.resolve(),out,job); completed.append(r)
            key=(r['subject'],r['copies'],r['method']); hashes.setdefault(key,set()).add(r['out_sha256'])
            if len(hashes[key])!=1: raise RuntimeError('output nondeterminism across fresh timing/metadata processes')
        final=validate_run(out); writej(out/'validation.json',final)
        if final['status']!='PASS': raise RuntimeError('final run incomplete/invalid')
        writej(out/'COMPLETED.json',{'status':'COMPLETED','n_primary_workers':len(completed),'config_sha256':obj_sha(c)})
        seal(out); print('COMPLETED; run r8c_analyze.py for scientific assessment'); return 0
    except BaseException as e:
        writej(out/'STOPPED.json',{'status':'NOT EXECUTED' if not (out/'workers').exists() else 'STOPPED','reason':f'{type(e).__name__}: {e}',
                                  'action':'Preserve partial evidence. Diagnose and start a new directory.'})
        seal(out); print(f'STOPPED: {e}',file=sys.stderr); return 3


if __name__=='__main__': raise SystemExit(main())
