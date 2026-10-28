"""Allocation endpoint + 10 ms resource observations on an independent pass.

Pool live is used_bytes; reserved is total_bytes. Neither is total process
VRAM. Non-pool VRAM and host RSS are sampled lower bounds. A sampler error
invalidates the metadata pass. No tensor references are retained by callbacks.
"""
import contextlib
import threading
import time
from pathlib import Path


def rss():
    q={}
    for line in Path('/proc/self/status').read_text().splitlines():
        v=line.split()
        if v[0] in ('VmRSS:','VmHWM:'): q[v[0][:-1]]=int(v[1])*1024
    return q


class MemoryTrace(contextlib.AbstractContextManager):
    def __init__(self,cp):
        self.cp=cp; self.stop=threading.Event(); self.lock=threading.RLock()
        self.peaks={}; self.n=0; self.errors=[]
    def observe(self):
        p=self.cp.get_default_memory_pool(); free,total=self.cp.cuda.runtime.memGetInfo()
        rec={'pool_live_peak_bytes':p.used_bytes(),'pool_reserved_peak_bytes':p.total_bytes(),
             'sampled_device_used_peak_bytes':total-free,'sampled_host_rss_peak_bytes':rss()['VmRSS']}
        with self.lock:
            self.n+=1
            for k,v in rec.items(): self.peaks[k]=max(v,self.peaks.get(k,0))
    def loop(self):
        try:
            self.cp.cuda.Device(0).use()
            while not self.stop.wait(.01): self.observe()
        except Exception as e: self.errors.append(str(e))
    def __enter__(self):
        self.base_rss=rss(); self.observe(); self.old=self.cp.cuda.get_allocator()
        def allocator(size):
            ptr=self.old(size); self.observe(); return ptr
        self.cp.cuda.set_allocator(allocator)
        self.thread=threading.Thread(target=self.loop,daemon=True); self.thread.start(); return self
    def __exit__(self,*exc):
        self.stop.set(); self.thread.join(10); self.cp.cuda.set_allocator(self.old)
        if self.thread.is_alive(): self.errors.append('sampler did not stop')
        self.observe(); self.end_rss=rss(); return False
    def result(self):
        return {**self.peaks,'samples':self.n,'errors':self.errors,'process_hwm_bytes':self.end_rss['VmHWM'],
                'scope':'algorithm allocation endpoints + 10ms samples; pool peaks observed, non-pool and RSS sampled lower bounds; HWM includes warm-up'}
