#!/usr/bin/env python3
"""R7-E1  BraTS cohort: CUDA-native K=1 full output vs same-tie CPU state machine (parity), vs Sequential FP (semantics), global DWC (independent).

Three INDEPENDENT judgements are recorded per subject (nothing is derived from another):
  (1) IMPLEMENTATION PARITY   GPU(tie=lower) == CPU state machine(tie=lower),  GPU(tie=higher) == CPU state machine(tie=higher)
                              bitwise on the whole padded array + equal tie-event counts + equal level-transition lists.
  (2) OUTPUT EQUIVALENCE      frozen Sequential FP (`fp_cpu`, insertion-order ties) vs GPU lower / GPU higher / CPU lower / CPU higher.
                              A difference is NOT classified as a CUDA error here; tie semantics may explain it (verifier classifies).
  (3) DWC                     production global face+cube check on every output (+ independent numpy checker on the first --cpu-check-n subjects).
DD-FP (K=16, delta=1) is run once and compared separately (recorded, never judged as an error).
Resumable: subjects/<id>.json that already exist are skipped.  Arrays are saved only for mismatches (--save-arrays mismatch).
Exit: 0 finished | 2 environment/integrity | 3 internal error (stops at the first failing subject) | 4 GPU busy / foreign process appeared.
"""
import argparse, json, platform, sys, time, traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_common as C


def cpu_modules(lib: Path):
    sys.path.insert(0, str(lib)); sys.path.insert(0, str(Path(__file__).resolve().parent))
    if C.sha_f(lib / "cpu_fp.py") != C.EXPECTED["src/ddfp/cpu_fp.py"]: raise C.Die(2, "lib/cpu_fp.py hash mismatch")
    import cpu_fp, r7_cpu_sm
    return cpu_fp, r7_cpu_sm


def short(c):    # compact comparison record
    return {k: c.get(k) for k in ("shape_equal", "n_diff_bits", "n_diff_value", "max_abs_diff", "equal", "n_nan_a", "n_nan_b")} | ({"first_diffs": c["first_diffs"][:5]} if c.get("first_diffs") else {})


def dwc(be, u, cpu_check):
    ug = be.h2d(np.ascontiguousarray(u)); g = be.global_check(ug)
    r = dict(violations=int(g["violations"]), dwc_ok=bool(g.get("dwc_ok")), nan_skipped=int(g.get("nan_skipped", 0)), total_blocks=int(g.get("total_blocks", -1)))
    del ug
    if cpu_check:
        c = be.cpu_check(u); r["cpu_indep_violations_by_family"] = {k: int(v) for k, v in c.items()}; r["cpu_indep_violations"] = int(sum(c.values()))
        r["gpu_cpu_agree"] = bool(r["violations"] == r["cpu_indep_violations"])
    return r


def run_subject(be, A, idx, sid, path, svol, cpu_fp, sm):
    t_all = time.perf_counter()
    if svol is not None: vol = svol
    else:
        import nibabel as nib
        vol = be.preprocess(nib.load(path).get_fdata(dtype=np.float32))
    R = dict(subject=sid, index=idx, input_sha256=C.sha_f(path) if path else None, vol_sha256=C.arr_sha(vol), vol_shape=list(vol.shape), backend=be.name, cuda_executed=be.cuda)
    # ---- GPU side (production K=1 path, then the same production kernels with explicit tie policy) ----
    lo, hi, l = be.prep(vol); lo_h, hi_h = be.ispan_host(lo, hi)
    g_prod = be.fp_k1(lo, hi, l)
    g_low, m_low = be.fp_tie(lo, hi, l, "lower"); g_high, m_high = be.fp_tie(lo, hi, l, "higher")
    del lo, hi; be.free()
    R["gpu_meta"] = {"lower": C.meta_light(m_low), "higher": C.meta_light(m_high)}
    R["hashes"] = {"gpu_k1_prod_padded": C.arr_sha(g_prod), "gpu_lower_padded": C.arr_sha(g_low), "gpu_higher_padded": C.arr_sha(g_high), "ispan_gpu_lo": C.arr_sha(lo_h), "ispan_gpu_hi": C.arr_sha(hi_h)}
    cmp = {"prod_k1_vs_direct_lower": short(C.cmp_arrays(g_prod, g_low))}
    # ---- CPU side ----
    lo_c, hi_c, l_c = cpu_fp.build_ispan_cpu(vol)
    cmp["ispan_lo_gpu_vs_cpu"], cmp["ispan_hi_gpu_vs_cpu"] = short(C.cmp_arrays(lo_h, lo_c)), short(C.cmp_arrays(hi_h, hi_c)); del lo_h, hi_h
    gpu_out, gpu_meta, cpu_out, cpu_t = {"lower": g_low, "higher": g_high}, {"lower": m_low, "higher": m_high}, {}, {}
    saved = []
    for pol in A.policies:
        st = {}; t0 = time.perf_counter(); c = sm.fp_cpu_tie(lo_c, hi_c, l_c, pol, st); cpu_t[pol] = round(time.perf_counter() - t0, 2)
        k = C.cmp_arrays(gpu_out[pol], c); r = short(k)
        r["tie_events_gpu"], r["tie_events_cpu"] = int(gpu_meta[pol].get("equidistant_next_level_tie_events", -1)), int(st["tie_events"])
        r["transitions_equal"] = bool(gpu_meta[pol].get("level_transitions") == st["level_transitions"])
        r["backward_gpu"], r["backward_cpu"] = int(gpu_meta[pol].get("backward_scheduler_transition_count", -1)), int(st["backward_transitions"])
        r["parity_ok"] = bool(k["equal"] and r["tie_events_gpu"] == r["tie_events_cpu"] and r["transitions_equal"])
        cmp[f"parity_{pol}"] = r; cpu_out[pol] = c; R["hashes"][f"cpu_{pol}_padded"] = C.arr_sha(c)
        if not r["parity_ok"] and A.save_arrays != "never": saved += [(f"{sid}_gpu_{pol}", gpu_out[pol]), (f"{sid}_cpu_{pol}", c)]
    t0 = time.perf_counter(); seq = cpu_fp.fp_cpu(lo_c, hi_c, l_c); R["t_seq_s_info"] = round(time.perf_counter() - t0, 2)   # info only; timing is E6's job
    R["hashes"]["seq_padded"] = C.arr_sha(seq)
    sem = {}
    for nm, arr in (("gpu_lower", g_low), ("gpu_higher", g_high), *[(f"cpu_{p}", cpu_out[p]) for p in A.policies]):
        sem[f"seq_vs_{nm}"] = short(C.cmp_arrays(seq, arr))
    sem["gpu_lower_vs_gpu_higher"] = short(C.cmp_arrays(g_low, g_high))
    if idx < A.clone_check_n:        # the R7 clone in insertion mode must reproduce the frozen Sequential FP on real data too
        ci = sm.fp_cpu_tie(lo_c, hi_c, l_c, "insertion", {}); sem["clone_insertion_vs_seq"] = short(C.cmp_arrays(seq, ci)); del ci
    cmp["semantics"] = sem
    if A.save_arrays == "always" or (A.save_arrays == "mismatch" and any(v.get("n_diff_bits", 0) for k, v in sem.items() if k.startswith("seq_vs_"))):
        saved += [(f"{sid}_seq", seq)]
    inner = lambda a: a[1:-1, 1:-1, 1:-1]
    R["hashes"].update(gpu_k1_out=C.arr_sha(inner(g_prod)), gpu_k1_sample=C.sample_hash(inner(g_prod)), seq_out=C.arr_sha(inner(seq)), seq_sample=C.sample_hash(inner(seq)),
                       gpu_higher_out=C.arr_sha(inner(g_high)))
    # ---- DWC (independent of parity and of equivalence) ----
    chk = idx < A.cpu_check_n
    R["dwc"] = {"gpu_k1": dwc(be, inner(g_prod), chk), "gpu_higher": dwc(be, inner(g_high), False), "seq": dwc(be, inner(seq), False)}
    # ---- DD-FP, recorded separately ----
    if A.with_ddfp:
        be.free(); r = be.ibi(vol, 16, 1, A.max_rounds); u = r["u_dd"]
        R["ddfp"] = dict(R_star=r["R_star"], boundary_z_orig=r["boundary_z_orig"], seam_violations=be.seam(u, r["boundary_z_orig"]), out_sha256=C.arr_sha(u), out_sample=C.sample_hash(u),
                         vs_gpu_k1=short(C.cmp_arrays(u, inner(g_prod))), vs_seq=short(C.cmp_arrays(u, inner(seq))), dwc=dwc(be, u, False))
        if A.save_arrays == "always": saved.append((f"{sid}_ddfp", u))
    # ---- save (capped) ----
    R["saved_arrays"] = []
    if saved and A.save_arrays != "never":
        d = A.out / "mismatch_arrays"; d.mkdir(parents=True, exist_ok=True); room = A.max_saved_gib * 2**30 - sum(p.stat().st_size for p in d.glob("*.npy"))
        for nm, a in saved:
            if a.nbytes <= room: np.save(d / f"{nm}.npy", a); room -= a.nbytes; R["saved_arrays"].append(nm)
            else: R["saved_arrays"].append(nm + " (NOT SAVED: size cap)")
    R["parity_ok"] = bool(all(cmp[f"parity_{p}"]["parity_ok"] for p in A.policies) and cmp["prod_k1_vs_direct_lower"]["equal"])
    R["dwc_ok"] = bool(all(v["violations"] == 0 and v.get("gpu_cpu_agree", True) for v in R["dwc"].values()) and (not A.with_ddfp or R["ddfp"]["dwc"]["violations"] == 0))
    R["comparisons"], R["cpu_seconds_info"] = cmp, cpu_t
    R["foreign_gpu_procs_after"] = C.foreign_procs(A.gpu_index) if be.cuda else []
    R["memory"] = be.memstats(); R["host_rss_mib"] = C.rss_mib(); R["seconds_total_info"] = round(time.perf_counter() - t_all, 1)
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--lib", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--backend", choices=["cupy", "stub"], default="cupy"); ap.add_argument("--brats-root", type=Path); ap.add_argument("--metadata", type=Path)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synthetic", type=int, default=0, help="stub plumbing test: this many random volumes"); ap.add_argument("--synthetic-size", type=int, default=6)
    ap.add_argument("--gpu-index", type=int, default=None); ap.add_argument("--min-free-gib", type=float, default=8); ap.add_argument("--allow-shared", action="store_true")
    ap.add_argument("--cpu-check-n", type=int, default=3); ap.add_argument("--clone-check-n", type=int, default=1)
    ap.add_argument("--policies", default="lower,higher"); ap.add_argument("--no-ddfp", dest="with_ddfp", action="store_false"); ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--save-arrays", choices=["mismatch", "always", "never"], default="mismatch"); ap.add_argument("--max-saved-gib", type=float, default=10)
    ap.add_argument("--stub-fault", choices=["parity", "dwc"], default=None)
    A = ap.parse_args(); A.policies = [p for p in A.policies.split(",") if p]; A.out.mkdir(parents=True, exist_ok=True); (A.out / "subjects").mkdir(exist_ok=True)
    if A.backend == "cupy" and A.stub_fault: raise C.Die(2, "--stub-fault is only valid with --backend stub")
    cpu_fp, sm = cpu_modules(A.lib)
    be = C.make_backend(A.backend, A.repo, A.lib, A.stub_fault)
    guard = C.gpu_guard(A.backend, A.gpu_index, A.min_free_gib, A.allow_shared)
    if A.synthetic: subs = [(s, None, v) for s, _, v in C.synthetic_cohort(A.n, A.synthetic_size)]; meta = None
    else:
        if not A.brats_root: raise C.Die(2, "--brats-root required")
        cl, meta = C.cohort(A.brats_root, A.metadata, A.n, A.seed); subs = [(s, p, None) for s, p in cl]
    head = dict(script="r7_e1.py", r7_version=C.R7_VERSION, script_sha256=C.sha_f(__file__), cm_sha256=C.sha_f(Path(__file__).with_name("r7_cpu_sm.py")), common_sha256=C.sha_f(C.__file__),
                started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), device=be.device_info(), guard=guard, host=C.host_facts(), args={k: str(v) for k, v in vars(A).items()},
                inputs_verified_against_metadata=bool(meta), production_hashes_verified=(C.EXPECTED if A.backend == "cupy" else "stub: not applicable"), python=platform.python_version())
    C.jdump(A.out / "run_meta.json", head)
    wv = np.random.default_rng(1).integers(0, 256, (9, 9, 12), dtype=np.uint8)     # untimed warm-up: compile every kernel path used below
    lo, hi, l = be.prep(wv); be.fp_k1(lo, hi, l); [be.fp_tie(lo, hi, l, t) for t in ("lower", "higher")]; be.global_check(be.h2d(np.zeros((9, 9, 9), np.float32))); be.free()
    done = 0
    for i, (sid, path, sv) in enumerate(subs):
        f = A.out / "subjects" / f"{sid}.json"
        if f.exists(): print(f"[skip] {sid} (done)", flush=True); done += 1; continue
        try: R = run_subject(be, A, i, sid, path, sv, cpu_fp, sm)
        except C.Die: raise
        except BaseException as e:
            C.jdump(A.out / "subjects" / f"{sid}.error.json", dict(subject=sid, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-3000:], backend=be.name))
            print(f"INTERNAL ERROR on {sid}: {type(e).__name__}: {e}", file=sys.stderr); return 3
        C.jdump(f, R); done += 1
        print(f"[{i+1}/{len(subs)}] {sid} parity_ok={R['parity_ok']} dwc_ok={R['dwc_ok']} tie_events(lower/higher)={R['gpu_meta']['lower']['equidistant_next_level_tie_events']}/{R['gpu_meta']['higher']['equidistant_next_level_tie_events']} "
              f"seq==gpu_lower:{R['comparisons']['semantics']['seq_vs_gpu_lower']['equal']} seq==gpu_higher:{R['comparisons']['semantics']['seq_vs_gpu_higher']['equal']} ({R['seconds_total_info']} s)", flush=True)
        if R["foreign_gpu_procs_after"] and not A.allow_shared:
            print("foreign GPU process appeared; stopping (own job only):", R["foreign_gpu_procs_after"], file=sys.stderr); return 4
    C.jdump(A.out / "summary.json", dict(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), subjects_done=done, subjects_expected=len(subs), backend=be.name, cuda_executed=be.cuda))
    return 0


if __name__ == "__main__":
    sys.exit(main())
