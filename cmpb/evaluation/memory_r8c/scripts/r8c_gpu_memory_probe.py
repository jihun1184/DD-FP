"""Real GPU worker, called in an independent subprocess by r8c_scaling_runner.

--purpose timing: synchronized wall phases, no tracing/allocator wrapper/sampler.
--purpose metadata: allocation-event + 10-ms sampling, source-stage snapshots,
R_act collection and non-timed verification. Never pool-purge during a solve.
CUDA-event elapsed time brackets the whole algorithm including host waits and
transfers; it is explicitly NOT a sum of pure kernel times.
"""
import argparse
import ast
import contextlib
import copy
import gc
import hashlib
import importlib
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from r8c_common import HERE, load_config, output_path, gate, admit, readj, writej, sha, obj_sha, smi, seal
from r8c_memory_model import verifier_plan
from r8c_validate import check_array, validate_record


def rss():
    q={}
    for line in Path('/proc/self/status').read_text().splitlines():
        v=line.split()
        if v[0] in ('VmRSS:','VmHWM:'): q[v[0][:-1]]=int(v[1])*1024
    return q


def instrument_function(fn, lines, checkpoint, round_callback=None, overrides=None):
    """Compile an instrumented copy of the HASH-CHECKED function, not a new solver.

    Insert scalar callbacks only. Never inspect frame.f_locals: its snapshot can
    keep old tensor references alive and corrupt the lifetime measurement.
    The original source file and original function remain unchanged.
    """
    tree=ast.parse(Path(fn.__code__.co_filename).read_text(encoding='utf-8'))
    node=copy.deepcopy(next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==fn.__name__))
    class Inject(ast.NodeTransformer):
        def visit(self,n):
            n=super().visit(n)
            before=[]; after=[]
            if isinstance(n,ast.stmt) and getattr(n,'lineno',None) in lines:
                call=ast.Expr(ast.Call(ast.Name('__r8c_checkpoint',ast.Load()),[ast.Constant(f'{fn.__name__}_before_line_{n.lineno}')],[]))
                before.append(ast.copy_location(call,n))
            if round_callback is not None and isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='max_change' for t in n.targets):
                call=ast.Expr(ast.Call(ast.Name('__r8c_round',ast.Load()),[ast.Name('r',ast.Load()),ast.Name('max_change',ast.Load())],[]))
                after.append(ast.copy_location(call,n))
            return before+[n]+after if before or after else n
    node=Inject().visit(node); module=ast.fix_missing_locations(ast.Module(body=node if isinstance(node,list) else [node],type_ignores=[]))
    namespace=dict(fn.__globals__); namespace.update(overrides or {})
    namespace.update(__r8c_checkpoint=checkpoint,__r8c_round=round_callback)
    exec(compile(module,fn.__code__.co_filename,'exec'),namespace)
    return namespace[fn.__name__]


class Monitor(contextlib.AbstractContextManager):
    """Allocator endpoints capture pool live/reserved highs; device/RSS are sampled.

    The allocator wrapper observes the SAME default MemoryPool.malloc result.
    It does not change tensor lifetimes. Scalar checkpoints synchronize only the
    metadata pass; measured metadata execution time is not used for comparisons.
    """
    def __init__(self, be, c, path):
        self.be=be; self.cp,self.c,self.path=be.cp,c,path; self.stop=threading.Event(); self.lock=threading.RLock()
        self.errors=[]; self.foreign=[]; self.samples=0; self.peak_used=0; self.peak_total=0; self.peak_dev=0; self.peak_rss=0
        self.stage='algorithm'; self.stages={}; self.solves=[]; self.round_changes=[]; self.transfers={}; self.restore=[]
        self.process_samples=[]; self.last_smi=0

    def observe(self, label=None, event=False):
        cp=self.cp; p=cp.get_default_memory_pool(); free,total=cp.cuda.runtime.memGetInfo(); host=rss()
        rec={'seconds':time.perf_counter()-self.start,'stage':label or self.stage,'pool_used_bytes':p.used_bytes(),
             'pool_total_bytes':p.total_bytes(),'device_used_bytes':total-free,'rss_bytes':host.get('VmRSS',0)}
        with self.lock:
            self.samples+=1; self.peak_used=max(self.peak_used,rec['pool_used_bytes']); self.peak_total=max(self.peak_total,rec['pool_total_bytes'])
            self.peak_dev=max(self.peak_dev,rec['device_used_bytes']); self.peak_rss=max(self.peak_rss,rec['rss_bytes'])
            bucket=self.stages.setdefault(rec['stage'],{})
            for k in ('pool_used_bytes','pool_total_bytes','device_used_bytes','rss_bytes'): bucket[k]=max(bucket.get(k,0),rec[k])
            if event: self.log.write(__import__('json').dumps(rec)+'\n'); self.log.flush()

    def loop(self):
        try:
            self.cp.cuda.Device(0).use()
            while not self.stop.wait(.01):
                self.observe()
                if time.monotonic()-self.last_smi >= .5:
                    g=smi(self.c); self.last_smi=time.monotonic()
                    self.process_samples.append({'seconds':time.perf_counter()-self.start,**g})
                    self.foreign.extend(g['foreign'])
        except Exception as e: self.errors.append(f'{type(e).__name__}: {e}')

    def checkpoint(self,label):
        self.cp.cuda.Stream.null.synchronize(); self.stage=label; self.observe(event=True)

    def __enter__(self):
        self.start=time.perf_counter(); self.log=self.path.open('x',encoding='utf-8')
        self.base_device=self.cp.cuda.runtime.memGetInfo(); self.base_host=rss(); self.observe('start',True)
        self.old_alloc=self.cp.cuda.get_allocator(); monitor=self
        def allocator(size):
            ptr=monitor.old_alloc(size); monitor.observe(); return ptr
        self.cp.cuda.set_allocator(allocator)
        old_build=self.be._build; old_ibi=self.be._run_ibi; old_fp=self.be._fp
        traced_build=instrument_function(old_build,{57,59,60,85,89,93,95,96,107,108,110},self.checkpoint)
        traced_fp=instrument_function(old_fp,{589,597,608},self.checkpoint)
        self.be._build=traced_build; self.be._fp=traced_fp
        def round_callback(r,change): self.round_changes.append({'round':int(r),'max_change':float(change)})
        self.be._run_ibi=instrument_function(old_ibi,{234,235,241,242,251,266,267,274},self.checkpoint,round_callback,
                                            {'build_ispan_gpu':traced_build,'front_propagation_gpu':traced_fp})
        self.restore.extend([lambda:setattr(self.be,'_build',old_build),lambda:setattr(self.be,'_run_ibi',old_ibi),lambda:setattr(self.be,'_fp',old_fp)])
        # Patch only Python attributes in this worker; disk sources remain frozen.
        mod=importlib.import_module('src.ddfp.gpu_seed_aware_worklist'); original=mod.seed_aware_boutry_bucket_gpu
        def solve(*a,**k):
            old=monitor.stage; monitor.stage='fp_seeded' if k.get('boundary_seeds') is not None or len(a)>4 and a[4] is not None else 'fp_unseeded'
            monitor.observe(event=True); start=time.perf_counter(); result=original(*a,**k)
            monitor.solves.append({**{key:result[1].get(key) for key in ('total_activation_rounds','scheduler_transition_count','equidistant_next_level_tie_events','nan_count')},
                                   'padded_shape':list(a[1].shape),'padded_cells':int(a[1].size),
                                   'seeded':k.get('boundary_seeds') is not None or len(a)>4 and a[4] is not None,
                                   'instrumented_seconds':time.perf_counter()-start})
            monitor.observe(event=True); monitor.stage=old; return result
        mod.seed_aware_boutry_bucket_gpu=solve; self.restore.append(lambda:setattr(mod,'seed_aware_boutry_bucket_gpu',original))
        # Observe transfer API calls (logical payload). Scalars and pinned staging are retained separately by scope.
        for api in ('asnumpy','asarray'):
            orig=getattr(self.cp,api)
            def transfer(*args,_name=api,_fn=orig,**kw):
                t=time.perf_counter(); result=_fn(*args,**kw)
                cross=(_name=='asnumpy' and args and isinstance(args[0],monitor.cp.ndarray)) or (_name=='asarray' and args and not isinstance(args[0],monitor.cp.ndarray))
                if cross:
                    totals=monitor.transfers.setdefault(_name,{'calls':0,'payload_bytes':0,'host_api_seconds':0.0})
                    totals['calls']+=1; totals['payload_bytes']+=int(getattr(result,'nbytes',0)); totals['host_api_seconds']+=time.perf_counter()-t
                return result
            setattr(self.cp,api,transfer); self.restore.append(lambda n=api,fn=orig:setattr(self.cp,n,fn))
        self.thread=threading.Thread(target=self.loop,daemon=True); self.thread.start(); return self

    def __exit__(self,*exc):
        self.stop.set(); self.thread.join(15)
        if self.thread.is_alive(): self.errors.append('sampler thread did not stop')
        self.cp.cuda.set_allocator(self.old_alloc)
        for fn in reversed(self.restore): fn()
        self.observe('end',True); self.end_host=rss(); self.log.close()
        return False

    def result(self):
        free,total=self.base_device
        return {'pool_used_peak_bytes':self.peak_used,'pool_total_peak_bytes':self.peak_total,'sampled_device_peak_bytes':self.peak_dev,
                'baseline_device_used_bytes':total-free,'sampled_device_delta_peak_bytes':self.peak_dev-(total-free),
                'sampled_host_rss_peak_bytes':self.peak_rss,'host_hwm_bytes':self.end_host.get('VmHWM'), 'baseline_host':self.base_host,
                'n_samples':self.samples,'sampler_errors':self.errors,'foreign_observed':self.foreign,'stage_peaks':self.stages,
                'process_gpu_samples':self.process_samples,'scope':'algorithm only; excludes verification and output save',
                'pool_peak_scope':'allocation endpoints plus stage snapshots and 10-ms samples; non-pool peaks remain sampled lower bounds',
                'transfers':self.transfers}


def algorithm(be,vol,K,rounds):
    """Use original R7 functions with original return/transfer behavior."""
    phases={}; cp=be.cp
    if K==1:
        cp.cuda.Stream.null.synchronize(); t=time.perf_counter(); lo,hi,li=be.prep(vol); be.sync(); phases['build_s']=time.perf_counter()-t
        t=time.perf_counter(); g=be._fp(lo,hi,li,verbose=False); be.sync(); phases['fp_including_host_bridge_s']=time.perf_counter()-t
        t=time.perf_counter(); u=cp.asnumpy(g[1:-1,1:-1,1:-1]); be.sync(); phases['output_d2h_s']=time.perf_counter()-t
        del lo,hi,g; extra={}
    else:
        be.sync(); t=time.perf_counter(); r=be.ibi(vol,K,1,rounds); be.sync(); phases['ibi_s']=time.perf_counter()-t
        u=r['u_dd']; extra={'R_star':r['R_star'],'boundary_z_orig':r['boundary_z_orig']}
    return u,phases,extra


def verify(be,u,extra):
    plan=verifier_plan(tuple(int(x) for x in u.shape)); out={'plan':plan}
    if not plan['ok']: return {**out,'verify_status':'REFUSED_INT32'}
    be.sync(); t=time.perf_counter(); ug=be.h2d(u); be.sync(); h2d=time.perf_counter()-t
    t=time.perf_counter(); v=be.global_check(ug); be.sync(); seconds=time.perf_counter()-t; del ug
    out.update(verify_status='DONE',global_dwc_ok=bool(v.get('dwc_ok')),global_violations=int(v['violations']),
               nan_skipped=int(v.get('nan_skipped',-1)),t_global_s=seconds,t_h2d_s=h2d)
    if extra.get('boundary_z_orig') is not None:
        t=time.perf_counter(); out['seam_violations']=be.seam(u,extra['boundary_z_orig']); out['t_seam_s']=time.perf_counter()-t
    return out


def worker(a):
    c=load_config(a.config); out=output_path(a.out,c); out.mkdir(parents=True,exist_ok=False)
    row={'schema':'R8C-1','status':'STARTED','backend':'cupy','cuda_executed':False,'purpose':a.purpose,'method':a.method,
         'subject':a.subject,'copies':a.copies,'rep':a.rep,'config_sha256':obj_sha(c),'pid':os.getpid(),'max_rounds':c['max_rounds']}
    writej(out/'partial.json',row,True)
    try:
        env=gate(c,require_data=a.purpose!='smoke'); writej(out/'environment.json',env,True)
        if not env['ok']: raise RuntimeError('; '.join(env['reasons']))
        import numpy as np
        sys.path.insert(0,str(Path(c['r7_runner'])/'r7')); import r7_common as C
        be=C.CupyBackend(c['repo'],str(Path(c['r7_runner'])/'lib')); cp=be.cp
        if a.purpose=='smoke':
            vol=np.random.default_rng(20261016).integers(0,256,tuple(c['smoke_shape']),dtype=np.uint8)
            inp=hashlib.sha256(vol.tobytes()).hexdigest()
        else:
            import nibabel as nib
            entry=env['inventory'][a.subject]; inp=entry['sha256']
            base=be.preprocess(nib.load(entry['path']).get_fdata(dtype=np.float32))
            if C.arr_sha(base)!=entry['vol_sha256'] or list(base.shape)!=entry['vol_shape']: raise RuntimeError('preprocessing regression mismatch')
            vol=np.ascontiguousarray(np.concatenate([base if i%2==0 else base[:,:,::-1] for i in range(a.copies)],axis=2)); del base
        K=1 if a.method=='gpu_k1' else 16 if a.method=='ddfp_k16' else 16*a.copies
        if K>vol.shape[2]: raise RuntimeError('K exceeds input depth')
        admission=admit(c,tuple(int(x) for x in vol.shape),K); writej(out/'admission.json',admission,True)
        if not admission['ok']: raise RuntimeError('ADMISSION_REFUSED: '+'; '.join(admission['reasons']))
        # Exact production-path compilation on tiny data, outside all measurement scopes.
        warm=np.random.default_rng(1).integers(0,256,(9,9,48),dtype=np.uint8)
        wu,_,_=algorithm(be,warm,1 if K==1 else 16,c['max_rounds']); del wu,warm
        tiny=cp.zeros((9,9,9),dtype=cp.float32); be.global_check(tiny); del tiny
        gc.collect(); be.free(); cp.get_default_pinned_memory_pool().free_all_blocks(); be.sync()
        try: Path('/proc/self/clear_refs').write_text('5'); hwm_reset=True
        except OSError: hwm_reset=False
        before=smi(c)
        if before['foreign']: raise RuntimeError('foreign GPU process before solve')
        # Clamp pool capacity below the reserved headroom; allocator errors stop the entire controller.
        cp.get_default_memory_pool().set_limit(size=int(admission['vram_limit_mib']*2**20))
        monitor=Monitor(be,c,out/'memory_timeline.jsonl') if a.purpose!='timing' else contextlib.nullcontext()
        with monitor as mon:
            be.sync(); ev0,ev1=cp.cuda.Event(),cp.cuda.Event(); ev0.record(); t=time.perf_counter()
            u,ph,extra=algorithm(be,vol,K,c['max_rounds']); be.sync(); wall=time.perf_counter()-t; ev1.record(); ev1.synchronize()
        row.update(status='COMPLETED',cuda_executed=True,K=K,vol_shape=list(vol.shape),out_shape=list(u.shape),input_sha256=inp,
                   vol_sha256=C.arr_sha(vol),out_sha256=C.arr_sha(u),t_algorithm_s=sum(ph.values()),wall_s=wall,phases=ph,
                   cuda_event_envelope_s=cp.cuda.get_elapsed_time(ev0,ev1)/1000,
                   kernel_execution_time='NOT MEASURED: CUDA event envelope includes host idle and transfers',**extra)
        if mon is not None:
            row['memory']=mon.result(); row['metadata']={'solves':mon.solves,'n_solves':len(mon.solves),
                'R_act_total':sum(s['total_activation_rounds'] for s in mon.solves),'round_changes':mon.round_changes}
            row['memory']['hwm_reset']=hwm_reset
            row['memory']['host_hwm_scope']='algorithm-scoped high-water' if hwm_reset else 'process high-water; warmup may contribute; use sampled_host_rss_peak_bytes for algorithm samples'
        row['verification']=verify(be,u,extra); row['array_check']=check_array(u,vol)
        if a.purpose=='smoke':
            row['cpu_independent_dwc_violations']=int(sum(be.cpu_check(u).values()))
            bad=np.zeros((3,3,3),np.float32); bad[0,0,0]=bad[1,1,1]=1
            bad_gpu=verify(be,bad,{})['global_violations']; bad_cpu=int(sum(be.cpu_check(bad).values()))
            row['verifier_negative_control']={'detected':bad_gpu>0 and bad_cpu>0,'gpu_violations':bad_gpu,'cpu_violations':bad_cpu,
                                             'description':'binary opposite-corner obstruction in one 2x2x2 cube'}
            if K==1:
                from cpu_fp import build_ispan_cpu
                from seedref import seed_aware_cpu_reference
                lo,hi,li=build_ispan_cpu(vol); ref=seed_aware_cpu_reference(lo,hi,li,None,'lower')[0][1:-1,1:-1,1:-1]
                row['regression']={'ok':bool(np.array_equal(u,ref)),'scope':'frozen unseeded CPU worklist reference with matching lower tie semantics'}
            else:
                import ibi_cpu
                ref=ibi_cpu.ibi(vol,K,1,c['max_rounds'],tie='lower')['u']
                row['regression']={'ok':bool(np.array_equal(u,ref)),'scope':'independent frozen CPU IBI port on tiny input'}
        elif a.copies==1:
            e1=readj(Path(c['e1_subjects'])/(a.subject+'.json'))
            expected=e1['hashes']['gpu_k1_out'] if K==1 else e1['ddfp']['out_sha256']
            row['regression']={'ok':row['out_sha256']==expected,'expected_sha256':expected,'scope':'R7 E1 full-output hash m=1'}
        else: row['regression']={'status':'NOT_AVAILABLE','scope':'no frozen larger-size output; independent full interval + DWC checks reported separately'}
        if a.purpose=='metadata':
            np.save(out/'output.npy',u,allow_pickle=False); row.update(output_npy='output.npy',output_file_sha256=sha(out/'output.npy'))
        row['environment_sha256']=sha(out/'environment.json')
        row['foreign_gpu_procs']=smi(c)['foreign']; row['sources_unchanged']=all(Path(p).is_file() and sha(p)==h for p,h in env['sources'].items())
        if not row['sources_unchanged']: raise RuntimeError('source changed during run')
        validation=validate_record(row,out); row['validation']=validation
        if not validation['ok']: row['status']='VALIDATION_FAILED'
        writej(out/'record.json',row,True); seal(out)
        return 0 if validation['ok'] else 3
    except BaseException as e:
        row.update(status='INTERRUPTED' if isinstance(e,KeyboardInterrupt) else 'OOM_UNEXPECTED' if 'OutOfMemory' in type(e).__name__ or 'out of memory' in str(e).lower() else 'ERROR',
                   error=f'{type(e).__name__}: {e}',traceback=traceback.format_exc())
        writej(out/'record.json',row); seal(out); print(row['error'],file=sys.stderr); return 3


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--config',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--purpose',choices=['timing','metadata','smoke'],required=True); ap.add_argument('--method',choices=['gpu_k1','ddfp_k16','ddfp_k16m'],required=True)
    ap.add_argument('--subject',default='SMOKE'); ap.add_argument('--copies',type=int,default=1); ap.add_argument('--rep',type=int,default=0)
    a=ap.parse_args()
    if a.copies not in (1,2,4) or a.rep not in (0,1,2): ap.error('primary V1 supports copies=1,2,4 and rep=0,1,2')
    return worker(a)


if __name__=='__main__': raise SystemExit(main())
