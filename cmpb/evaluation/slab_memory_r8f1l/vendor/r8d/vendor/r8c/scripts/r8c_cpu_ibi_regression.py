"""CPU-only production-assembly regression with explicit, fail-closed import shim.

This does NOT test CUDA. The SciPy gaussian_filter import is unused by these
random-array cases; if SciPy is absent, calling that placeholder raises.
Original R7 CPU solver, assembly, independent CPU IBI port and mutation checks
are reused byte-for-byte. No scientific record is emitted by this script.
"""
import argparse
import ast
import importlib.util
import json
import sys
import types
from typing import Tuple
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--v1',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    import numpy as np
    missing_scipy=False
    try: import scipy.ndimage
    except ImportError:
        missing_scipy=True; sp=types.ModuleType('scipy'); nd=types.ModuleType('scipy.ndimage')
        def forbidden(*args,**kwargs): raise RuntimeError('Gaussian filtering excluded from this CPU assembly test')
        nd.gaussian_filter=forbidden; sp.ndimage=nd; sys.modules.update({'scipy':sp,'scipy.ndimage':nd})
    p=a.v1/'DDFP_R7_GPU_SERVER_RUNNER_R0/r3_scripts/r3_diff_ibi.py'
    spec=importlib.util.spec_from_file_location('original_r3_differential',p); D=importlib.util.module_from_spec(spec); spec.loader.exec_module(D)
    cpu,seed,ibi=D.load_lib(a.v1/'DDFP_R7_GPU_SERVER_RUNNER_R0/lib')
    import r1_check53
    negative=np.zeros((3,3,3),np.float32); negative[0,0,0]=negative[1,1,1]=1
    bad_count,bad_oracle=r1_check53.check(negative,levels=[1.0]); good_count,_=r1_check53.check(np.zeros_like(negative))
    dwc_negative_ok=sum(bad_count.values())>0 and bad_count==bad_oracle and sum(good_count.values())==0
    prod=D.load_production(a.v1/'DD-FP-R7','shim',cpu,seed,'lower'); rng=np.random.default_rng(20261016)
    cases=[]
    for shape in [(3,3,9),(5,5,12),(6,6,15)]:
        for K in (2,3):
            for delta in (0,1,2):
                cases.append({'id':f'{shape}_K{K}_d{delta}','vol':rng.integers(0,256,shape,dtype=np.uint8),'K':K,'delta':delta})
    rows=D.run_suite(cases,prod,ibi,'lower'); mutations={}
    # New R8-C metadata instrumentation must preserve the original solver values.
    from r8c_gpu_memory_probe import instrument_function
    # Exercise the original GPU-build expression graph with NumPy operations,
    # including annotation compilation, without pretending this launches CUDA.
    build_source=a.v1/'DD-FP-R7/src/ddfp/gpu_immersion.py'
    function=next(n for n in ast.parse(build_source.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='build_ispan_gpu')
    ns={'cp':np,'np':np,'Tuple':Tuple}; exec(compile(ast.Module(body=[function],type_ignores=[]),str(build_source),'exec'),ns)
    raw_build=ns['build_ispan_gpu']; build_checkpoints=[]
    instrumented_build=instrument_function(raw_build,{57,59,60,85,89,93,95,96,107,108,110},build_checkpoints.append)
    bv=rng.integers(0,256,(5,6,7),dtype=np.uint8); bb=instrumented_build(bv); br=cpu.build_ispan_cpu(bv)
    build_numpy_shim_ok=all(np.array_equal(x,y) for x,y in zip(bb,br)) and bool(build_checkpoints)
    checkpoints=[]; round_changes=[]; original_ibi=prod.run_ibi_v10
    instrumented=instrument_function(original_ibi,{234,235,241,242,251,266,267,274},checkpoints.append,
                                    lambda r,c:round_changes.append((r,c)))
    prod.run_ibi_v10=instrumented
    try: instrumented_rows=D.run_suite(cases,prod,ibi,'lower')
    finally: prod.run_ibi_v10=original_ibi
    for name,(attr,fn,original) in D.mutants(prod).items():
        setattr(prod,attr,fn)
        try: mutations[name]=sum(not r.get('identical',False) for r in D.run_suite(cases,prod,ibi,'lower'))
        finally: setattr(prod,attr,original)
    ok=all(r.get('identical') is True for r in rows+instrumented_rows) and bool(round_changes) and bool(checkpoints) and all(v>0 for v in mutations.values()) and dwc_negative_ok and build_numpy_shim_ok
    out={'kind':'CPU_SHIM_TEST_NOT_GPU_EVIDENCE','ok':ok,'n_cases':len(rows),'cases':rows,'mutation_cases_detected':mutations,
         'instrumented_cases_identical':sum(r.get('identical') is True for r in instrumented_rows),
         'instrumented_round_callbacks':len(round_changes),'instrumented_checkpoints':len(checkpoints),
         'build_numpy_expression_shim_ok':build_numpy_shim_ok,'cpu_dwc_negative_control_ok':dwc_negative_ok,'cpu_dwc_negative_counts':bad_count,
         'unused_scipy_import_placeholder':missing_scipy,'scope':'original and R8C callback-instrumented assembly vs independent CPU IBI; CUDA and Gaussian preprocessing not exercised'}
    a.out.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8'); print(json.dumps({k:v for k,v in out.items() if k!='cases'},indent=2))
    return 0 if ok else 1


if __name__=='__main__': raise SystemExit(main())
