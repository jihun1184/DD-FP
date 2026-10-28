"""D3 controller: preflight, CPU-checked GPU smoke, randomized fresh workers.

Every subject/m/K has 3 uninstrumented timings, 1 profile, 1 metadata worker.
No GPU-process inventory is queried. Existing nonterminal runs resume only
with identical config/source/input fingerprints and validated completed jobs.
"""
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from r8d_common import ROOT, C, config, gate, hashes, obj_sha, output, parser, readj, seal, writej
from r8d_validate import record, validate_run


def plan(c,subjects):
    rng=random.Random(c['order_seed']); jobs=[]
    for m in (1,2,4):
        pairs=[(s,K) for s in subjects for K in [1]+c['k_candidates'][str(m)]]
        # Replicate blocks prevent all 3 timings from sharing one order position.
        for purpose,rep in [('timing',i) for i in range(3)]+[('profile',0),('metadata',0)]:
            rng.shuffle(pairs)
            for s,K in pairs:
                jobs.append({'id':f'{s}_m{m}_K{K}_{purpose}_{rep}','subject':s,'copies':m,'K':K,'purpose':purpose,'rep':rep})
    return jobs


def child(c,config_path,out,job):
    folder=out/'workers'/job['id']
    if folder.exists():
        if (folder/'record.json').is_file():
            r=readj(folder/'record.json')
            if record(r,folder)['ok'] and r['config_sha256']==obj_sha(c) and all(r[k]==job[k] for k in ('subject','copies','K','purpose','rep')):
                return r
        raise RuntimeError('incomplete worker preserved; choose new run directory: '+str(folder))
    args=[c['python'],'-B',str(ROOT/'scripts/r8d_worker.py'),'--config',str(config_path),'--out',str(folder)]
    for k in ('subject','copies','K','purpose','rep'): args += ['--'+k,str(job[k])]
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1',CUDA_VISIBLE_DEVICES=str(c['gpu_index']),
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUPY_CACHE_DIR=str(out/'cupy_cache'))
    (out/'logs').mkdir(exist_ok=True)
    with (out/'logs'/(job['id']+'.log')).open('x',encoding='utf-8') as f:
        process=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,env=env); start=time.monotonic()
        try:
            while process.poll() is None:
                if (out/'STOP_REQUESTED').exists(): raise RuntimeError('USER_STOP_REQUESTED')
                if time.monotonic()-start>c['worker_timeout_s']: raise TimeoutError('worker timeout')
                if (folder/'stage.json').is_file():
                    stage=readj(folder/'stage.json')
                    if time.time()-stage['started_unix_s']>c['stage_timeout_s']: raise TimeoutError('stage timeout: '+stage['stage'])
                time.sleep(.5)
        except BaseException as e:
            if process.poll() is None:
                process.terminate()
                try: process.wait(10)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            writej(folder/'controller_interruption.json',{'reason':str(e),'job':job}); raise
    if process.returncode: raise RuntimeError('worker failed: '+job['id'])
    r=readj(folder/'record.json')
    if not record(r,folder)['ok']: raise RuntimeError('invalid worker: '+job['id'])
    return r


def main():
    p=parser(__doc__); mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight',action='store_true'); mode.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true'); a=p.parse_args(); c=config(a.config); out=output(a.out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'STOPPED.json').exists() or (out/'COMPLETED.json').exists(): raise SystemExit('terminal run preserved; use a new output directory')
    if any(out.iterdir()) and not a.resume: raise SystemExit('existing run: use --resume or a new output directory')
    if (out/'run_config.json').exists():
        # Refuse drift before adding a preflight/STOPPED file to old evidence.
        old=readj(out/'run_config.json')
        try: unchanged=old.get('config_sha256')==obj_sha(c) and old.get('sources')==hashes(c)
        except Exception: unchanged=False
        if not unchanged:
            print('EXISTING RUN PRESERVED: fingerprint changed; use new directory',file=sys.stderr); return 2
    try:
        env=gate(c); name='preflight.json' if not (out/'preflight.json').exists() else f'preflight_{time.time_ns()}.json'
        writej(out/name,env)
        if not env['ok']: print('NOT EXECUTED: '+'; '.join(env['reasons'])); seal(out); return 2
        subjects=C.selection(c); jobs=plan(c,subjects)
        manifest={'schema':'R8D-1','config':c,'config_sha256':obj_sha(c),'subjects':subjects,'sources':hashes(c),'jobs':jobs}
        if (out/'run_config.json').exists():
            if readj(out/'run_config.json')!=manifest: raise RuntimeError('resume fingerprint changed')
        else: writej(out/'run_config.json',manifest)
        disk=sum(4*__import__('math').prod(C_extent(env['inventory'][j['subject']]['vol_shape'],j['copies'])) for j in jobs if j['purpose']=='metadata')+2**31
        if shutil.disk_usage(out).free<disk: raise RuntimeError(f'disk budget refused: need {disk} bytes')
        writej(out/'resource_plan.json',{'saved_output_budget_bytes':disk,'primary_workers':len(jobs),'smoke_workers':12})
        if a.preflight: print(f'PREFLIGHT PASS: {len(jobs)} primary workers + 12 smoke'); seal(out); return 0
        # Every candidate path receives two independent CPU-reference smoke checks.
        for K in (1,8,16,32,64,128):
            outputs=[]
            for rep in (0,1):
                job={'id':f'SMOKE_K{K}_{rep}','subject':'SMOKE','copies':1,'K':K,'purpose':'smoke','rep':rep}
                outputs.append(child(c,a.config.resolve(),out,job)['out_sha256'])
            if len(set(outputs))!=1: raise RuntimeError('smoke nondeterminism')
        seen={}
        for i,j in enumerate(jobs):
            shape=list(env['inventory'][j['subject']]['vol_shape']); shape[2]*=j['copies']
            admission=C.admit(c,tuple(shape),j['K'])
            if not admission['ok']: raise RuntimeError('ADMISSION REFUSED '+j['id']+': '+'; '.join(admission['reasons']))
            print(f'[{i+1}/{len(jobs)}] {j["id"]}',flush=True)
            r=child(c,a.config.resolve(),out,j); key=(r['subject'],r['copies'],r['K'])
            seen.setdefault(key,set()).add(r['out_sha256'])
            if len(seen[key])!=1: raise RuntimeError('timing/profile/metadata output mismatch')
        check=validate_run(out); writej(out/'validation.json',check)
        if check['status']!='PASS': raise RuntimeError('full validation failed')
        writej(out/'COMPLETED.json',{'status':'COMPLETED','primary_workers':len(jobs)})
        seal(out); print('COMPLETED; run r8d_analyze.py'); return 0
    except BaseException as e:
        writej(out/'STOPPED.json',{'reason':f'{type(e).__name__}: {e}','action':'partial evidence preserved; new directory after diagnosis'})
        seal(out); print(str(e),file=sys.stderr); return 3


def C_extent(shape,m): return [2*shape[0]-1,2*shape[1]-1,2*shape[2]*m-1]


if __name__=='__main__': raise SystemExit(main())
