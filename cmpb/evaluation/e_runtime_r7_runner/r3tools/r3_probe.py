#!/usr/bin/env python3
"""Tiny wiring probe (a few seconds, < 200 MB GPU memory): imports exactly what scripts A/B import, runs the single-pass FP on a 12^3 volume,
compares it bit-for-bit with the CPU reference, and runs the production global face+cube check on it.
It starts nothing long-running and must only be launched on a GPU that the guard found idle.
--backend cpu is a plumbing self-test with CPU stand-ins (its result is never GPU evidence).
Exit: 0 pass | 1 mismatch / violation | 2 import or environment problem | 3 internal error
"""
import argparse, hashlib, json, sys, time, traceback
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--lib", type=Path, required=True)
    ap.add_argument("--backend", choices=["cupy", "cpu"], default="cupy"); ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import numpy as np
    rec = dict(script="r3_probe.py", backend=a.backend, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), steps={})
    sys.path.insert(0, str(a.lib)); sys.path.insert(0, str(a.repo))
    try:
        import r1_check53 as r1
        if a.backend == "cupy":
            import cupy as cp
            from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu, verify_dwc_gpu          # same imports as r3_global_check_timing.py
            from src.ddfp.cpu_fp import build_ispan_cpu, fp_cpu
            from scripts.ddfp.experiment_DDFP_all import preprocess, run_ibi_v10, count_boundary_violations
            import nibabel                                                                                     # noqa: F401
            rec["steps"]["imports"] = "ok"; rec["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            rec["free_gib_after_context"] = round(cp.cuda.runtime.memGetInfo()[0] / 2**30, 2)
        else:
            import cpu_fp as _c
            build_ispan_cpu, fp_cpu = _c.build_ispan_cpu, _c.fp_cpu
            build_ispan_gpu = build_ispan_cpu; front_propagation_gpu = lambda lo, hi, l, verbose=False: fp_cpu(lo, hi, l)
            class cp:                                             # noqa: N801 -- minimal stand-in
                asarray = staticmethod(np.asarray); asnumpy = staticmethod(np.asarray)
            def verify_dwc_gpu(u, verbose=False):
                v = sum(r1.check(np.ascontiguousarray(u, np.float32), slab=16)[0].values()); return dict(dwc_ok=v == 0, violations=v, nan_skipped=0, total_blocks=-1)
            rec["steps"]["imports"] = "cpu-standins"; rec["gpu"] = "CPU_STANDIN"
    except Exception as e:
        rec["steps"]["imports"] = f"{type(e).__name__}: {e}"; rec["verdict"] = "ERROR_IMPORT"
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1)); return 2
    try:
        vol = np.random.default_rng(0).integers(0, 256, (12, 12, 12), dtype=np.uint8)
        t0 = time.perf_counter(); lo, hi, l_inf = build_ispan_gpu(vol); g = front_propagation_gpu(lo, hi, l_inf, verbose=False)[1:-1, 1:-1, 1:-1]
        g = np.ascontiguousarray(cp.asnumpy(g), np.float32); rec["gpu_fp_seconds_incl_compile"] = round(time.perf_counter() - t0, 3)
        lo2, hi2, l2 = build_ispan_cpu(vol); c = np.ascontiguousarray(fp_cpu(lo2, hi2, l2)[1:-1, 1:-1, 1:-1], np.float32)
        same = g.shape == c.shape and bool(np.array_equal(g.view(np.uint32), c.view(np.uint32)))
        rec["steps"]["fp_bitwise_equal_cpu"] = same; rec["out_sha256"] = hashlib.sha256(g.tobytes()).hexdigest()
        v = verify_dwc_gpu(cp.asarray(g), verbose=False); rec["steps"]["global_check"] = {k: (x if isinstance(x, (int, float, bool)) else str(x)) for k, x in v.items()}
        ok = same and v["violations"] == 0 and v.get("dwc_ok", False)
        rec["verdict"] = "PROBE_PASS" if ok else "PROBE_FAIL"
    except Exception as e:
        rec["verdict"] = "ERROR_RUN"; rec["error"] = f"{type(e).__name__}: {e}"; rec["trace"] = traceback.format_exc()[-1500:]; ok = None
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(rec, indent=1)); print(json.dumps(rec, indent=1))
    return 0 if ok else (3 if ok is None else 1)


if __name__ == "__main__":
    sys.exit(main())
