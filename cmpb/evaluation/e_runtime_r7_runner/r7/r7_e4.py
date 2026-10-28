#!/usr/bin/env python3
"""R7-E4  Safe stepwise memory scaling of CUDA-native K=1 and DD-FP (K=16, delta=1).  NO out-of-memory run is ever provoked on purpose.

Input: one recorded cohort subject, tiled along z (m copies, every second copy mirrored so the volume stays continuous) -> expanded cell count grows linearly with m.
Before EVERY run the step is admitted only if a conservative prediction fits the limits (otherwise it is recorded as SKIPPED_BY_SAFETY_RULE and that method stops):
  predicted(m) = 1.15 * max( v_last * m / m_last ,  linear extrapolation through the last two points )      (no intercept discount; power-law term for super-linear growth; a margin, not a guarantee)
  VRAM : predicted peak-delta <= min( safety_fraction * total - baseline , total - baseline - headroom )
  RAM  : predicted RSS        <= host_ram_fraction * available RAM at start
  time : cumulative + predicted run time <= max_seconds
The first step (m = first value of --copies, normally 1) is the size of the recorded benchmark volumes that already ran on 24 GB.
The largest COMPLETED step is a lower bound of the processable size, NOT the true limit.  Out-of-memory happening anyway is recorded as OOM_UNEXPECTED and stops the method.
Exit: 0 finished | 2 environment | 3 internal | 4 GPU busy / foreign process appeared.
"""
import argparse, gc, json, math, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_common as C


def tile_z(vol, m):
    return np.ascontiguousarray(np.concatenate([vol if i % 2 == 0 else vol[:, :, ::-1] for i in range(m)], axis=2))


def predict(points, m):
    """points: [(m_i, value_i)] ascending in m_i, at least one.  Conservative prediction for size m: 1.15 x max(proportional, linear extrapolation, power-law extrapolation)
    where the power-law exponent is fitted through the last two points and floored at 1.  Growth that accelerates beyond that fit is NOT covered by the formula;
    the safety fraction and the absolute VRAM headroom in `lim` are what absorb it (see module docstring) -- this is a safety margin, not a guarantee."""
    m1, v1 = points[-1]; est = v1 * m / m1
    if len(points) >= 2:
        m0, v0 = points[-2]; est = max(est, v1 + (v1 - v0) / (m1 - m0) * (m - m1))
        if v0 > 0 and v1 > 0 and m1 > m0: est = max(est, v1 * (m / m1) ** max(1.0, math.log(v1 / v0) / math.log(m1 / m0)))
    return 1.15 * est


def decide(points_vram, points_rss, points_sec, m, lim):
    """Pure function.  Returns (admit: bool, reason: str, predictions: dict)."""
    if not points_vram: return True, "first step (recorded benchmark size)", {}
    p = dict(vram_mib=predict(points_vram, m), rss_mib=predict(points_rss, m), seconds=predict(points_sec, m))
    if p["vram_mib"] > lim["vram_mib"]: return False, f"predicted VRAM {p['vram_mib']:.0f} MiB > limit {lim['vram_mib']:.0f} MiB", p
    if p["rss_mib"] > lim["rss_mib"]: return False, f"predicted RSS {p['rss_mib']:.0f} MiB > limit {lim['rss_mib']:.0f} MiB", p
    if lim["seconds_used"] + p["seconds"] > lim["seconds_max"]: return False, f"predicted time {p['seconds']:.0f} s exceeds the remaining budget", p
    return True, "within limits", p


def run(be, m_name, vol, A):
    if m_name == "gpu_k1":
        lo, hi, l = be.prep(vol); g = be.fp_k1_dev(lo, hi, l); be.sync(); u = be.to_host(g[1:-1, 1:-1, 1:-1]); del lo, hi, g
    else:
        r = be.ibi(vol, 16, 1, A.max_rounds); be.sync(); u = r["u_dd"]
    return u


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--lib", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", choices=["cupy", "stub"], default="cupy"); ap.add_argument("--brats-root", type=Path); ap.add_argument("--metadata", type=Path)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--subject-index", type=int, default=0)
    ap.add_argument("--copies", default="1,2,3,4,6,8"); ap.add_argument("--methods", default="gpu_k1,ddfp_k16_d1"); ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--safety-fraction", type=float, default=0.75); ap.add_argument("--headroom-gib", type=float, default=3.0); ap.add_argument("--host-ram-fraction", type=float, default=0.6)
    ap.add_argument("--max-seconds", type=float, default=5400)
    ap.add_argument("--synthetic", type=int, default=0); ap.add_argument("--synthetic-size", type=int, default=6)
    ap.add_argument("--stub-total-mib", type=float, default=None, help="stub only: pretend this device size so the admission rule can be exercised")
    ap.add_argument("--gpu-index", type=int, default=None); ap.add_argument("--min-free-gib", type=float, default=8); ap.add_argument("--allow-shared", action="store_true")
    A = ap.parse_args(); A.out.mkdir(parents=True, exist_ok=True); methods = [m for m in A.methods.split(",") if m]; steps = [int(x) for x in A.copies.split(",")]
    be = C.make_backend(A.backend, A.repo, A.lib); guard = C.gpu_guard(A.backend, A.gpu_index, A.min_free_gib, A.allow_shared)
    if be.cuda: cp = be.cp; be.fp_k1_dev = lambda lo, hi, l: be._fp(lo, hi, l, verbose=False); be.to_host = lambda g: C.f32(cp.asnumpy(g))
    else: be.fp_k1_dev = lambda lo, hi, l: be.seedref.seed_aware_cpu_reference(lo, hi, l, None, "lower")[0]; be.to_host = lambda g: C.f32(g)
    hf = C.host_facts()
    if A.synthetic: base_vol = C.synthetic_cohort(1, A.synthetic_size)[0][2]; sid, meta = "synthetic0", None
    else:
        if not A.brats_root: raise C.Die(2, "--brats-root required")
        cl, meta = C.cohort(A.brats_root, A.metadata, A.n, A.seed); sid, path = cl[A.subject_index]
        import nibabel as nib
        base_vol = be.preprocess(nib.load(path).get_fdata(dtype=np.float32))
    be.free(); be.sync(); ms = be.memstats(); total = ms.get("device_total_mib", A.stub_total_mib or 24000.0); base = ms.get("device_used_mib", 0.0)
    lim = dict(vram_mib=min(A.safety_fraction * total - base, total - base - A.headroom_gib * 1024), rss_mib=A.host_ram_fraction * hf.get("mem_available_gib", 8) * 1024, seconds_max=A.max_seconds, seconds_used=0.0)
    C.jdump(A.out / "run_meta.json", dict(script="r7_e4.py", r7_version=C.R7_VERSION, script_sha256=C.sha_f(__file__), common_sha256=C.sha_f(C.__file__), started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              device=be.device_info(), guard=guard, host=hf, args={k: str(v) for k, v in vars(A).items()}, subject=sid, base_vol_shape=list(base_vol.shape), inputs_verified_against_metadata=bool(meta),
              limits=lim, device_total_mib=total, device_baseline_used_mib=base, note="no out-of-memory run is provoked on purpose; see module docstring"))
    pts = {m: dict(v=[], r=[], s=[]) for m in methods}; stopped = {}; rows_f = A.out / "steps.jsonl"
    if rows_f.exists(): raise C.Die(2, f"{rows_f} exists: E4 results are never overwritten; use a new attempt directory")
    wv = np.random.default_rng(1).integers(0, 256, (9, 9, 48 if be.cuda else 12), dtype=np.uint8)
    for m in methods: run(be, m, wv, A)
    be.free(); be.sync()
    for k in steps:
        vol = tile_z(base_vol, k); cells = int(np.prod([2 * d - 1 for d in vol.shape]))
        for m in methods:
            if m in stopped: continue
            ok, why, pred = decide([(a, b) for a, b in pts[m]["v"]], pts[m]["r"], pts[m]["s"], k, lim)
            row = dict(copies=k, vol_shape=list(vol.shape), expanded_cells=cells, method=m, backend=be.name, cuda_executed=be.cuda, admission=why, predicted=pred, limits={x: round(y, 1) for x, y in lim.items()})
            if not ok:
                row["status"] = "SKIPPED_BY_SAFETY_RULE"; stopped[m] = k; C.append_jsonl(rows_f, row); print(f"[m={k}] {m}: SKIPPED_BY_SAFETY_RULE ({why})", flush=True); continue
            try:
                gc.collect(); be.free(); be.sync(); hw = C.hwm_reset(); t = time.time()
                with C.Sampler(be) as s: u = run(be, m, vol, A)
                dt = time.time() - t; r = s.result(); ms2 = be.memstats()
                row.update(status="COMPLETED", seconds=round(dt, 2), **r, **{f"mem_{a}": b for a, b in ms2.items()}, vmhwm_mib=C.rss_mib().get("VmHWM"), hwm_reset=hw, out_shape=list(u.shape),
                           out_finite=bool(np.isfinite(u).all()), out_sha256=C.arr_sha(u), expected_out_shape=[2 * d - 1 for d in vol.shape])
                del u
                pts[m]["v"].append((k, max(r["peak_vram_delta_mib"], 1.0))); pts[m]["r"].append((k, max(r["peak_rss_mib"], C.rss_mib().get("VmHWM", 0.0)))); pts[m]["s"].append((k, dt)); lim["seconds_used"] += dt
            except BaseException as e:
                if isinstance(e, KeyboardInterrupt): raise
                oom = "OutOfMemory" in type(e).__name__ or "out of memory" in str(e).lower()
                row.update(status="OOM_UNEXPECTED" if oom else "ERROR", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:]); stopped[m] = k
                try: be.free()
                except Exception: pass
            row["foreign_gpu_procs_after"] = C.foreign_procs(A.gpu_index) if be.cuda else []
            C.append_jsonl(rows_f, row)
            print(f"[m={k}] {m}: {row['status']}" + (f" peakVRAMdelta={row['peak_vram_delta_mib']:.0f} MiB rss={row['peak_rss_mib']:.0f} MiB {row['seconds']}s" if row["status"] == "COMPLETED" else ""), flush=True)
            if row["foreign_gpu_procs_after"] and not A.allow_shared: print("foreign GPU process appeared; stopping (own job only)", file=sys.stderr); return 4
            if row["status"] in ("OOM_UNEXPECTED", "ERROR"): print(f"{m} stopped: {row.get('error')}", file=sys.stderr)
    C.jdump(A.out / "summary.json", dict(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), stopped=stopped, backend=be.name, cuda_executed=be.cuda, steps=steps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
