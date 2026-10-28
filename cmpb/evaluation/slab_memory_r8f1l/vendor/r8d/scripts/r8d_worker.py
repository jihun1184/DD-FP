"""One fresh CUDA process. Stage markers enable external timeout and recovery.

Never changes production files. Failure record and flushed profile JSONL are
preserved. Timing, profile, metadata have deliberately different overheads.
"""
import argparse
import contextlib
import gc
import os
import time
import traceback
from pathlib import Path
from r8d_common import ROOT, C, arr_sha, config, gate, hashes, load_r7, obj_sha, output, readj, seal, sha, writej
from r8d_timing_probe import Profiler, algorithm
from r8d_activation_trace import ActivationTrace
from r8d_memory_trace import MemoryTrace
from r8d_validate import check_array, record
from r8c_gpu_memory_probe import verify


def worker(a):
    c=config(a.config); out=output(a.out); out.mkdir(parents=True,exist_ok=False)
    row={'schema':'R8D-1','status':'STARTED','backend':'cupy','cuda_executed':False,
         'subject':a.subject,'copies':a.copies,'K':a.K,'method':'gpu_k1' if a.K==1 else 'ddfp',
         'purpose':a.purpose,'rep':a.rep,'config_sha256':obj_sha(c),'max_rounds':c['max_rounds']}
    def stage(name): writej(out/'stage.json',{'stage':name,'started_unix_s':time.time()})
    writej(out/'partial.json',row)
    prof=None
    try:
        stage('environment'); env=gate(c,require_data=a.purpose!='smoke'); writej(out/'environment.json',env)
        if not env['ok']: raise RuntimeError('; '.join(env['reasons']))
        import numpy as np
        R7=load_r7(c); be=R7.CupyBackend(c['repo'],str(Path(c['r7_runner'])/'lib')); cp=be.cp
        stage('input')
        if a.purpose=='smoke':
            vol=np.random.default_rng(20261016).integers(0,256,tuple(c['smoke_shape']),dtype=np.uint8); input_hash=arr_sha(vol)
        else:
            import nibabel as nib
            entry=env['inventory'][a.subject]; input_hash=entry['sha256']
            base=be.preprocess(nib.load(entry['path']).get_fdata(dtype=np.float32))
            if arr_sha(base)!=entry['vol_sha256']: raise RuntimeError('preprocessed hash mismatch')
            vol=np.ascontiguousarray(np.concatenate([base if i%2==0 else base[:,:,::-1] for i in range(a.copies)],axis=2)); del base
        stage('admission')
        admission=C.admit(c,tuple(int(v) for v in vol.shape),a.K); writej(out/'admission.json',admission)
        if not admission['ok']: raise RuntimeError('ADMISSION REFUSED: '+'; '.join(admission['reasons']))
        stage('warmup'); t=time.perf_counter()
        warm=np.random.default_rng(1).integers(0,256,tuple(c['smoke_shape']),dtype=np.uint8)
        wu,_=algorithm(be,warm,a.K,c['max_rounds']); del wu,warm
        tiny=cp.zeros((9,9,9),dtype=cp.float32); be.global_check(tiny); del tiny
        be.sync(); row['warmup_wall_s']=time.perf_counter()-t
        row['warmup_scope']='CUDA context + exact K route + JIT/cache loading + tiny global verifier; outside algorithm'
        gc.collect(); be.free(); cp.get_default_pinned_memory_pool().free_all_blocks(); be.sync()
        admission=C.admit(c,tuple(int(v) for v in vol.shape),a.K); writej(out/'admission.json',admission)
        if not admission['ok']: raise RuntimeError('resources changed during warmup')
        cp.get_default_memory_pool().set_limit(size=int(admission['vram_limit_mib']*2**20))
        trace=None; memory=None
        row['smoke_profile']=a.purpose=='smoke' and a.rep==1
        if a.purpose=='profile' or row['smoke_profile']: prof=Profiler(cp,out/'profile.jsonl',c['stage_timeout_s'])
        tracing=ActivationTrace(be,prof) if a.purpose!='timing' else contextlib.nullcontext()
        monitoring=MemoryTrace(cp) if a.purpose in ('metadata','smoke') else contextlib.nullcontext()
        stage('algorithm')
        with monitoring as memory, tracing as trace:
            be.sync(); e0,e1=cp.cuda.Event(),cp.cuda.Event(); e0.record(); t=time.perf_counter()
            with prof.span('algorithm_profile') if prof else contextlib.nullcontext():
                u,extra=algorithm(be,vol,a.K,c['max_rounds'],prof)
            be.sync()
            row['algorithm_wall_s']=time.perf_counter()-t; e1.record(); e1.synchronize()
            row['cuda_event_envelope_s']=cp.cuda.get_elapsed_time(e0,e1)/1000
        if prof:
            row['profile']=prof.result(); prof.close(); prof=None
        if trace: row['activation']=trace.result()
        if memory: row['memory']=memory.result()
        row.update(extra, cuda_executed=True,vol_shape=list(vol.shape),out_shape=list(u.shape),input_sha256=input_hash,
                   vol_sha256=arr_sha(vol),out_sha256=arr_sha(u))
        stage('verification')
        row['verification']=verify(be,u,extra); row['array_check']=check_array(u,vol)
        row['regression']={'status':'NOT AVAILABLE for this (m,K)'}
        if a.purpose=='smoke':
            row['cpu_dwc_violations']=int(sum(be.cpu_check(u).values()))
            bad=np.zeros((3,3,3),np.float32); bad[0,0,0]=bad[1,1,1]=1
            row['negative_control_detected']=verify(be,bad,{})['global_violations']>0 and sum(be.cpu_check(bad).values())>0
            if a.K==1:
                from cpu_fp import build_ispan_cpu
                from seedref import seed_aware_cpu_reference
                lo,hi,li=build_ispan_cpu(vol)
                ref=seed_aware_cpu_reference(lo,hi,li,None,'lower')[0][1:-1,1:-1,1:-1]
            else:
                import ibi_cpu
                ref=ibi_cpu.ibi(vol,a.K,1,c['max_rounds'],tie='lower')['u']
            row['cpu_reference_equal']=bool(np.array_equal(u,ref))
        elif a.copies==1 and a.K in (1,16):
            e1=readj(Path(c['e1_subjects'])/(a.subject+'.json'))
            expected=e1['hashes']['gpu_k1_out'] if a.K==1 else e1['ddfp']['out_sha256']
            row['regression']={'expected':expected,'ok':row['out_sha256']==expected}
        stage('save')
        if a.purpose=='metadata':
            np.save(out/'output.npy',u,allow_pickle=False); row['output_file_sha256']=sha(out/'output.npy')
        row['environment_sha256']=sha(out/'environment.json'); row['admission_sha256']=sha(out/'admission.json')
        row['sources_unchanged']=hashes(c)==env['sources']
        row['status']='COMPLETED'; row['validation']=record(row,out)
        if not row['validation']['ok']: row['status']='VALIDATION_FAILED'
        writej(out/'record.json',row); stage('done'); seal(out)
        return 0 if row['validation']['ok'] else 3
    except BaseException as e:
        row.update(status='INTERRUPTED' if isinstance(e,KeyboardInterrupt) else 'OOM' if 'memory' in str(e).lower() and 'out' in str(e).lower() else 'ERROR',
                   error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
        if prof: prof.close()
        writej(out/'record.json',row); seal(out); print(row['error'],flush=True); return 3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','out','subject','purpose'): p.add_argument('--'+name,required=True)
    for name in ('copies','K','rep'): p.add_argument('--'+name,type=int,required=True)
    a=p.parse_args()
    if a.K<1 or a.purpose not in ('smoke','timing','profile','metadata'): p.error('invalid K/purpose')
    return worker(a)


if __name__=='__main__': raise SystemExit(main())
