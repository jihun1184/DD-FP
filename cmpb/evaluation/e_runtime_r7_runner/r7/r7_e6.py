#!/usr/bin/env python3
"""R7-E6 / E3-data  Same-server benchmark: CUDA-native K=1, DD-FP (K=16, delta=1), Sequential FP on the SAME server CPU, on the same cohort.

Protocol (fixed before running; derived from the recorded 2026-09-05 benchmark, repeats reduced 5 -> 3, extendable with --reps-start/--reps):
  * cohort = recorded 20 subjects (seed 42, inputs hash-verified); preprocessing (load + percentile scaling) is NOT timed, as in the recorded benchmark
  * untimed warm-up: every GPU code path once (tiny volume) before the first subject
  * per (subject, repetition): the three methods in a pseudo-random order (rng(order_seed, subject_index, rep)); every run starts with
    GPU pool release + sync, VmHWM reset, and ends with sync before the clock stops
  * phases (all synchronised):  K1     t_prep = build_ispan_gpu | t_fp = front_propagation_gpu (includes the production kernel's own D2H/H2D of the result) | t_d2h = slice + asnumpy
                                DD-FP  t_ibi = run_ibi_v10 (whole IBI) ; t_seam = production count_boundary_violations (CPU numpy) timed separately
                                Seq    t_prep = build_ispan_cpu | t_fp = fp_cpu | t_out = interior slice
    t_algorithm = sum of the timed algorithm phases (K1: prep+fp+d2h, DD-FP: ibi, Seq: prep+fp+out); t_recorded_equiv adds t_seam for DD-FP (what the recorded 14.85 s contained)
  * AFTER the clock stopped (never inside it): output hashes, GPU temperature/power, foreign-process check, optional verification costs for E3
    (H2D + production global face+cube check; seam scan for DD-FP)
  * memory: zero-perturbation metrics in every run (VmHWM after reset, CuPy pool bytes, device used); polling sampler only in the separate, non-timed
    memory pass (--mem-pass): peak device memory delta and RSS for GPU methods
Resumable at row level.  Exit: 0 finished | 2 environment | 3 internal | 4 GPU/CPU busy.
"""
import argparse, gc, json, os, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_common as C


def order_for(seed, subj_i, rep, methods):
    rng = np.random.default_rng([seed, subj_i, rep]); return [methods[i] for i in rng.permutation(len(methods))]


def run_method(be, cpu_fp, m, vol, A):
    """Returns (u_interior_host, phases dict, extras dict).  Everything inside is the timed region; sync() brackets every phase."""
    ph, ex, pc = {}, {}, time.perf_counter
    if m == "gpu_k1":
        be.sync(); t = pc(); lo, hi, l = be.prep(vol); be.sync(); ph["t_prep"] = pc() - t
        t = pc(); g = be.fp_k1_dev(lo, hi, l); be.sync(); ph["t_fp"] = pc() - t
        t = pc(); u = be.to_host(g[1:-1, 1:-1, 1:-1]); ph["t_d2h"] = pc() - t
        del lo, hi, g
    elif m == "ddfp_k16_d1":
        be.sync(); t = pc(); r = be.ibi(vol, 16, 1, A.max_rounds); be.sync(); ph["t_ibi"] = pc() - t
        u = r["u_dd"]; ex = dict(R_star=r["R_star"], boundary_z_orig=r["boundary_z_orig"])
    else:
        t = pc(); lo, hi, l = cpu_fp.build_ispan_cpu(vol); ph["t_prep"] = pc() - t
        t = pc(); s = cpu_fp.fp_cpu(lo, hi, l); ph["t_fp"] = pc() - t
        t = pc(); u = np.ascontiguousarray(s[1:-1, 1:-1, 1:-1]); ph["t_out"] = pc() - t
        del lo, hi, s
    return u, ph, ex


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--lib", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", choices=["cupy", "stub"], default="cupy"); ap.add_argument("--brats-root", type=Path); ap.add_argument("--metadata", type=Path)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--order-seed", type=int, default=20260904)
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--reps-start", type=int, default=0, help="extend an earlier run: first repetition index to execute")
    ap.add_argument("--methods", default=",".join(C.METHODS)); ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--no-verify-costs", dest="verify_costs", action="store_false"); ap.add_argument("--no-mem-pass", dest="mem_pass", action="store_false")
    ap.add_argument("--synthetic", type=int, default=0); ap.add_argument("--synthetic-size", type=int, default=6)
    ap.add_argument("--gpu-index", type=int, default=None); ap.add_argument("--min-free-gib", type=float, default=8); ap.add_argument("--allow-shared", action="store_true")
    ap.add_argument("--max-load1", type=float, default=None, help="refuse to start when the 1-min load average exceeds this (default 0.5 x CPU count)")
    A = ap.parse_args(); A.out.mkdir(parents=True, exist_ok=True); methods = [m for m in A.methods.split(",") if m]
    be = C.make_backend(A.backend, A.repo, A.lib); guard = C.gpu_guard(A.backend, A.gpu_index, A.min_free_gib, A.allow_shared)
    if be.cuda:       # adapter methods that keep device arrays alive between the timed phases
        cp = be.cp; be.fp_k1_dev = lambda lo, hi, l: be._fp(lo, hi, l, verbose=False); be.to_host = lambda g: C.f32(cp.asnumpy(g))
    else:
        be.fp_k1_dev = lambda lo, hi, l: be.seedref.seed_aware_cpu_reference(lo, hi, l, None, "lower")[0]; be.to_host = lambda g: C.f32(g)
    hf = C.host_facts(); lim = A.max_load1 if A.max_load1 is not None else 0.5 * (hf.get("cpu_count") or 1)
    if be.cuda and hf.get("loadavg", [0])[0] > lim: raise C.Die(4, f"1-min load average {hf['loadavg'][0]:.2f} > {lim:.2f}: other CPU work would contaminate the Sequential timing (use --max-load1 to override knowingly)")
    cpu_fp, _ = __import__("r7_e1").cpu_modules(A.lib)
    if A.synthetic: subs = [(s, None, v) for s, _, v in C.synthetic_cohort(A.n, A.synthetic_size)]; meta = None
    else:
        if not A.brats_root: raise C.Die(2, "--brats-root required")
        cl, meta = C.cohort(A.brats_root, A.metadata, A.n, A.seed); subs = [(s, p, None) for s, p in cl]
    C.jdump(A.out / "run_meta.json", dict(script="r7_e6.py", r7_version=C.R7_VERSION, script_sha256=C.sha_f(__file__), common_sha256=C.sha_f(C.__file__), started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              device=be.device_info(), guard=guard, host=hf, args={k: str(v) for k, v in vars(A).items()}, hwm_reset_supported=C.hwm_reset(), inputs_verified_against_metadata=bool(meta),
              protocol=dict(reps=A.reps, reps_start=A.reps_start, order_seed=A.order_seed, methods=methods, warmup="1 per GPU path, tiny volume, untimed", preprocessing_timed=False, K=16, delta=1,
                            sequential_baseline="frozen fp_cpu, single thread (OMP/MKL=1), same server CPU as the GPU host")))
    wv = np.random.default_rng(1).integers(0, 256, (9, 9, 48 if be.cuda else 12), dtype=np.uint8)
    for m in methods:                                   # untimed warm-up of every GPU path (Sequential FP needs none)
        if m != "seq_fp_cpu": run_method(be, cpu_fp, m, wv, A)
    be.global_check(be.h2d(np.zeros((9, 9, 9), np.float32))); be.free()
    rows_f = A.out / "rows.jsonl"; have = set()
    if rows_f.exists():
        for ln in rows_f.read_text().splitlines():
            if ln.strip(): r = json.loads(ln); have.add((r["subject"], r["rep"], r["method"]))
    reps = range(A.reps_start, A.reps_start + A.reps)
    for si, (sid, path, sv) in enumerate(subs):
        todo = [(rep, m) for rep in reps for m in methods if (sid, rep, m) not in have]
        if not todo and (not A.mem_pass or (sid, -1, "mem") in have): continue
        t0 = time.perf_counter()
        if sv is not None: vol = sv
        else:
            import nibabel as nib
            vol = be.preprocess(nib.load(path).get_fdata(dtype=np.float32))
        t_load = round(time.perf_counter() - t0, 2); inp = C.sha_f(path) if path else None; vsha = C.arr_sha(vol)
        for rep in reps:
            for pos, m in enumerate(order_for(A.order_seed, si, rep, methods)):
                if (sid, rep, m) in have: continue
                try:
                    gc.collect(); be.free(); be.sync(); hw = C.hwm_reset(); load0 = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
                    wall0 = time.time(); u, ph, ex = run_method(be, cpu_fp, m, vol, A); be.sync(); wall = time.time() - wall0
                    row = dict(subject=sid, rep=rep, method=m, order_position=pos, backend=be.name, cuda_executed=be.cuda, input_sha256=inp, vol_sha256=vsha, t_load_untimed_s=t_load,
                               **{k: round(v, 6) for k, v in ph.items()}, t_algorithm_s=round(sum(ph.values()), 6), wall_s=round(wall, 3), load1_before=load0, hwm_reset=hw,
                               out_sha256=C.arr_sha(u), out_sample_sha256=C.sample_hash(u), out_shape=list(u.shape), **{f"mem_{k}": v for k, v in be.memstats().items()}, **{k.lower(): v for k, v in C.rss_mib().items()})
                    if m == "ddfp_k16_d1": row.update(R_star=ex["R_star"])
                    if be.cuda: row["gpu_snapshot"] = C.gpu_snapshot(A.gpu_index)
                    # ---- E3 verification costs: outside the timed region ----
                    if A.verify_costs and m != "seq_fp_cpu":
                        if m == "ddfp_k16_d1":
                            t = time.perf_counter(); row["seam_violations"] = be.seam(u, ex["boundary_z_orig"]); row["t_seam_s"] = round(time.perf_counter() - t, 6)
                            row["t_recorded_equiv_s"] = round(row["t_algorithm_s"] + row["t_seam_s"], 6)
                        be.sync(); t = time.perf_counter(); ug = be.h2d(u); be.sync(); row["t_h2d_s"] = round(time.perf_counter() - t, 6)
                        t = time.perf_counter(); g = be.global_check(ug); be.sync(); row["t_global_s"] = round(time.perf_counter() - t, 6)
                        row["global_violations"], row["global_dwc_ok"] = int(g["violations"]), bool(g.get("dwc_ok")); del ug
                    elif m == "gpu_k1": row["t_recorded_equiv_s"] = row["t_algorithm_s"]
                    row["foreign_gpu_procs_after"] = C.foreign_procs(A.gpu_index) if be.cuda else []
                    C.append_jsonl(rows_f, row); have.add((sid, rep, m)); del u
                except C.Die: raise
                except BaseException as e:
                    C.jdump(A.out / f"error_{sid}_{rep}_{m}.json", dict(subject=sid, rep=rep, method=m, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:])); print("INTERNAL ERROR:", e, file=sys.stderr); return 3
                print(f"[{si+1}/{len(subs)}] {sid} rep{rep} {m:<12} t_alg={row['t_algorithm_s']:.2f}s" + (f" +seam {row['t_seam_s']:.2f}s" if 't_seam_s' in row else "") + (f" global {row['t_global_s']:.3f}s" if 't_global_s' in row else ""), flush=True)
                if row["foreign_gpu_procs_after"] and not A.allow_shared: print("foreign GPU process appeared; stopping (own job only):", row["foreign_gpu_procs_after"], file=sys.stderr); return 4
        if A.mem_pass and be.cuda and (sid, -1, "mem") not in have:
            mem = dict(subject=sid, rep=-1, method="mem", backend=be.name, cuda_executed=True)
            for m in [x for x in methods if x != "seq_fp_cpu"]:
                gc.collect(); be.free(); be.sync()
                with C.Sampler(be) as s: u, ph, ex = run_method(be, cpu_fp, m, vol, A); be.sync()
                mem[m] = s.result() | dict(phases={k: round(v, 3) for k, v in ph.items()}); del u
            C.append_jsonl(rows_f, mem); have.add((sid, -1, "mem")); print(f"[{si+1}/{len(subs)}] {sid} memory pass done", flush=True)
    C.jdump(A.out / "summary.json", dict(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), rows=len(have), backend=be.name, cuda_executed=be.cuda, reps=list(reps)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
