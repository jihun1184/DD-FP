"""D2: nested wall/CUDA-stream observations, scalar-only records.

The baseline never calls Profiler. Profile mode intentionally serializes
observed spans to make their attribution inspectable. Stream spans include
submission/idle gaps; they are not pure device execution times. Nested spans
must not be added. Unobserved implicit .item()/index transfers are declared.
"""
import contextlib
import time


class Forward:
    def __init__(self,obj,**overrides): self._obj=obj; self.__dict__.update(overrides)
    def __getattr__(self,n): return getattr(self._obj,n)


class Profiler:
    def __init__(self,cp,path,timeout_s):
        import json
        self.cp=cp; self.file=path.open('x',encoding='utf-8'); self.json=json
        self.timeout_s=timeout_s; self.stack=[]; self.n=0; self.totals={}; self.syncs={}; self.transfers={}
        self.start=time.perf_counter(); self.deadline=self.start+timeout_s
        profiler=self
        def synchronization(fn,label):
            def run(*a,**k):
                profiler.check(); t=time.perf_counter(); out=fn(*a,**k)
                bucket=profiler.syncs.setdefault(label,{'calls':0,'host_wait_s':0.0})
                bucket['calls']+=1; bucket['host_wait_s']+=time.perf_counter()-t
                return out
            return run
        null=Forward(cp.cuda.Stream.null,synchronize=synchronization(cp.cuda.Stream.null.synchronize,'Stream.null.synchronize'))
        # Frozen production uses the null stream. Any drift is hash-rejected.
        cuda=Forward(cp.cuda,Stream=Forward(cp.cuda.Stream,null=null),
            runtime=Forward(cp.cuda.runtime,deviceSynchronize=synchronization(cp.cuda.runtime.deviceSynchronize,'deviceSynchronize')))
        def rawmodule(*a,**k):
            m=cp.RawModule(*a,**k)
            def get_function(name):
                kernel=m.get_function(name)
                def launch(*args,**kw):
                    with profiler.span('kernel:'+name): return kernel(*args,**kw)
                return launch
            return Forward(m,get_function=get_function)
        def transfer(name):
            fn=getattr(cp,name)
            def run(*a,**k):
                cross=bool(a) and ((name=='asnumpy' and isinstance(a[0],cp.ndarray)) or (name in ('asarray','array') and not isinstance(a[0],cp.ndarray)))
                if not cross: return fn(*a,**k)
                direction='D2H' if name=='asnumpy' else 'H2D'
                with profiler.span('transfer:'+direction): result=fn(*a,**k)
                payload=int(getattr(result,'nbytes',0)); b=profiler.transfers.setdefault(direction,{'calls':0,'logical_payload_bytes':0})
                b['calls']+=1; b['logical_payload_bytes']+=payload
                return result
            return run
        self.cp_proxy=Forward(cp,cuda=cuda,RawModule=rawmodule,asarray=transfer('asarray'),asnumpy=transfer('asnumpy'),array=transfer('array'))

    def check(self):
        if time.perf_counter()>self.deadline: raise TimeoutError('profile stage timeout')

    def round_change(self,*a): pass

    @contextlib.contextmanager
    def span(self,label):
        self.check(); ident=self.n; self.n+=1; parent=self.stack[-1] if self.stack else None
        a,b=self.cp.cuda.Event(),self.cp.cuda.Event()
        a.record(); t=time.perf_counter(); self.stack.append(ident); failed=False
        try: yield
        except BaseException:
            failed=True; raise
        finally:
            b.record(); b.synchronize(); wall=time.perf_counter()-t; self.stack.pop()
            event=self.cp.cuda.get_elapsed_time(a,b)/1000
            rec={'id':ident,'parent':parent,'label':label,'start_s':t-self.start,'wall_s':wall,
                 'cuda_stream_span_s':event,'failed':failed}
            # Per-launch numbers are reduced online. Writing a million JSON
            # lines per solve would dominate timing and storage. Structural
            # scopes remain in JSONL; kernel totals retain exact call counts.
            if not label.startswith('kernel:'):
                self.file.write(self.json.dumps(rec)+'\n')
            if ident%256==0: self.file.flush()
            v=self.totals.setdefault(label,{'calls':0,'inclusive_wall_s':0.0,'inclusive_cuda_stream_span_s':0.0})
            v['calls']+=1; v['inclusive_wall_s']+=wall; v['inclusive_cuda_stream_span_s']+=event

    def result(self):
        return {'nested_inclusive_spans':self.totals,'observed_explicit_sync':self.syncs,'transfers':self.transfers,
            'transfer_scope':'asnumpy/asarray/array logical API payload only; excludes ndarray.item/index implicit transfers, internal staging, memcpy and device copies',
            'sync_scope':'explicit production null-stream and runtime calls only; excludes implicit scalar waits and profiler-added Event.synchronize',
            'cuda_scope':'serialized default-stream event brackets include submission gaps; nested inclusive spans are not additive',
            'profile_intrusive':True,'span_count':self.n}

    def close(self): self.file.close()


def algorithm(be,vol,K,rounds,profile=None):
    """Same algorithm boundary/host return behavior as R8-C; no profile hooks."""
    cp=profile.cp_proxy if profile else be.cp
    if K==1:
        lo,hi,li=be.prep(vol)
        u=be._fp(lo,hi,li,verbose=False)
        u=cp.asnumpy(u[1:-1,1:-1,1:-1])
        return u,{}
    r=be.ibi(vol,K,1,rounds)
    return r['u_dd'],{'R_star':r['R_star'],'boundary_z_orig':r['boundary_z_orig']}
