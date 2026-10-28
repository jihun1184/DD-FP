#!/usr/bin/env python3
"""R3-B (revision R3.1: + --gpu-index, own-pid filtering of nvidia-smi compute apps; nothing else changed)
Global face+cube DWC check and cost split for the existing N=20 BraTS cohort (seed 42).  Verification script only.

For each subject and each method {gpu_single_pass_k1, ddfp_k16_d1} it records, with CUDA synchronisation around every timer:
  t_gen        output generation, exactly the benchmark's `run_method` pieces:  K1 = build_ispan+FP+D2H ; DD-FP = run_ibi_v10
  t_seam       DD-FP only: production `count_boundary_violations` (CPU numpy, xz/yz faces at the cut planes)  <- INSIDE the 14.85 s of the
               recorded benchmark (runner `benchmark_fixed_local.py` calls it inside run_method), reproduced here separately
  t_h2d,t_gdwc production `verify_dwc_gpu` (global face+cube, all orientations): host->device copy and kernel, NOT in the recorded 14.85 s
  t_cpu        optional independent audit `r1_check53.check` (numpy, written from the DWC definition); not a deployable cost
  hashes       input file, preprocessed volume, FULL output (sha256) and the benchmark's 1/4096-strided sample hash, compared with the
               recorded CSV (`--ref-csv`) if given
Full block-wise certification (all tile-interior + cross-owner blocks) is NOT implemented in production => reported as NOT_MEASURED.
Safety: refuses to start if another process uses the GPU (exit 4) unless --allow-shared; never writes outside --out; no sudo.
Exit: 0 ran, all global checks agree and 0 violations | 1 a violation or GPU/CPU disagreement | 2 env/integrity | 3 internal | 4 GPU busy
Pilot:  python -I r3_global_check_timing.py --repo R --lib L --brats-root B --metadata benchmark_metadata.json --n 1 --out out_pilot
Full :  ... --n 20 --ref-csv benchmark_rows.csv --out out_n20              (--synthetic K --backend stub = CPU plumbing test only)
"""
import argparse, glob, hashlib, json, os, platform, subprocess, statistics as st, sys, time, traceback
from pathlib import Path
import numpy as np

EXPECTED = {"scripts/ddfp/experiment_DDFP_all.py": "b6a9b109f4105e7b77ebed880f669ba670bbd87123deca9509ba5b00993e561e",
            "src/ddfp/cpu_fp.py": "8eae554f7d3d688dee2285f41d20bfbf6e34af98016a3f1ad07f1ff5f8f36a88",
            "src/ddfp/gpu_immersion.py": "b784a376d3466a1d4ded5089a2eee9f27df35006b4926301dd06106b4ed262a4",
            "src/ddfp/gpu_native_worklist.py": "853bf33a6bbd9c91f5304a018f0a51ff53e8418360fe1477208cf81df17163ac",
            "src/ddfp/gpu_seed_aware_worklist.py": "c80f56b9ba7afd63196332d51894168d36f6e9f3e2a51b6b966786311e4eecad"}
METHODS = ("gpu_single_pass_k1", "ddfp_k16_d1")


def sha_b(b): return hashlib.sha256(b).hexdigest()
def sha_f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
def sample_hash(a): return sha_b(np.ascontiguousarray(a.ravel()[::4096]).view(np.uint8))      # identical to benchmark_fixed_local.sample_hash
def die(code, msg): print("ERROR:", msg, file=sys.stderr); sys.exit(code)


def host_facts():
    f = dict(platform=platform.platform(), python=platform.python_version(), numpy=np.__version__)
    try:
        f["cpu_model"] = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
        f["mem_total_gib"] = round(int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))) / 2**20, 1)
        f["loadavg"] = os.getloadavg()
    except Exception as e: f["host_probe_error"] = str(e)
    return f


GPU_INDEX = None          # R3.1: physical nvidia-smi index of the GPU this run uses (None = all GPUs, R3 behaviour)


def nvsmi(q, fmt="csv,noheader"):
    cmd = ["nvidia-smi"] + (["-i", str(GPU_INDEX)] if GPU_INDEX is not None else []) + [f"--query-{q}", f"--format={fmt}"]
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e: return f"UNAVAILABLE({type(e).__name__})"


def other_procs(csv_text):
    """R3.1: compute-app lines (pid, name, mem) whose pid is not this process."""
    return [l for l in csv_text.splitlines() if l.strip() and not l.startswith("UNAVAILABLE") and l.split(",")[0].strip() != str(os.getpid())]


def select_cohort(root, n, seed, meta):
    paths = sorted(glob.glob(str(Path(root) / "*" / "*_flair.nii*")))
    if len(paths) != 1251: die(2, f"expected 1251 FLAIR files under {root}, found {len(paths)}")
    idx = sorted(np.random.default_rng(seed).choice(len(paths), size=20, replace=False).tolist())      # benchmark used n=20, seed=42
    sel = [Path(paths[i]) for i in idx][:n]
    if meta:                                                                                           # input integrity vs recorded cohort
        rec = {c["subject_id"]: c for c in meta["cohort"]}
        if n >= 20 and sorted(rec) != sorted(p.parent.name for p in sel): die(2, "selected cohort differs from recorded cohort")
        for p in sel:
            c = rec.get(p.parent.name)
            if c is None: die(2, f"{p.parent.name} not in recorded cohort")
            if p.name != c["file_name"] or p.stat().st_size != c["size_bytes"] or sha_f(p) != c["sha256"]: die(2, f"input mismatch {p.name}")
    return sel


def check_cpu(a, r1):                                   # independent checker, per block family
    return r1.check(np.ascontiguousarray(a, np.float32), slab=16)[0]


def main():
    global GPU_INDEX
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True); ap.add_argument("--lib", type=Path, required=True)
    ap.add_argument("--brats-root", type=Path); ap.add_argument("--metadata", type=Path); ap.add_argument("--ref-csv", type=Path)
    ap.add_argument("--n", type=int, default=1); ap.add_argument("--reps", type=int, default=1); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rounds", type=int, default=8); ap.add_argument("--seq-subjects", type=int, default=0, help="exact comparison with sequential FP (~110 s each)")
    ap.add_argument("--no-cpu-check", action="store_true"); ap.add_argument("--cpu-check-n", type=int, default=3, help="independent CPU audit on the first N subjects only (it takes minutes per volume)"); ap.add_argument("--allow-shared", action="store_true"); ap.add_argument("--gpu-index", type=int, default=None, help="R3.1: physical GPU index for nvidia-smi queries (multi-GPU servers)"); ap.add_argument("--min-free-gib", type=float, default=8)
    ap.add_argument("--backend", choices=["cupy", "stub"], default="cupy"); ap.add_argument("--synthetic", type=int, default=0, help="stub plumbing test: random volumes of this size")
    ap.add_argument("--out", type=Path, required=True)
    A = ap.parse_args(); A.out.mkdir(parents=True, exist_ok=True); GPU_INDEX = A.gpu_index
    sys.path.insert(0, str(A.lib)); sys.path.insert(0, str(Path(__file__).parent))
    import r1_check53 as r1
    rec = dict(script="r3_global_check_timing.py", script_sha256=sha_f(__file__), started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), host=host_facts(), args={k: str(v) for k, v in vars(A).items()})
    # ---- repo integrity (production files must be the recovered, unmodified ones) --------------------------------------------------------
    if A.backend == "cupy":
        bad = {k: sha_f(A.repo / k) for k, v in EXPECTED.items() if not (A.repo / k).is_file() or sha_f(A.repo / k) != v}
        if bad: die(2, f"production hash mismatch: {sorted(bad)}")
        rec["production_hashes_verified"] = EXPECTED
        sys.path.insert(0, str(A.repo))
        import cupy as cp
        from scripts.ddfp.experiment_DDFP_all import preprocess, run_ibi_v10, count_boundary_violations
        from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu, verify_dwc_gpu
        from src.ddfp.cpu_fp import build_ispan_cpu, fp_cpu
        import nibabel as nib
        sync = cp.cuda.Stream.null.synchronize
        apps = nvsmi("compute-apps=pid,process_name,used_memory")
        rec["gpu"] = dict(name=nvsmi("gpu=name,driver_version,memory.total,memory.used,power.limit,utilization.gpu"), compute_apps_before=apps, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(), cupy=cp.__version__)
        free = cp.cuda.runtime.memGetInfo()[0] / 2**30; rec["gpu"]["free_gib_before"] = round(free, 2)
        if apps.startswith("UNAVAILABLE") and not A.allow_shared: die(2, "nvidia-smi unavailable: cannot verify that no other job uses the GPU (use --allow-shared only if you know it is free)")
        if apps and not apps.startswith("UNAVAILABLE") and not A.allow_shared: die(4, f"GPU in use by other process(es): {apps!r} -- refusing to start (use --allow-shared only with owner consent)")
        if free < A.min_free_gib: die(4, f"only {free:.1f} GiB GPU memory free (< {A.min_free_gib})")
    else:                                                  # CPU stand-ins from the differential script: plumbing test only
        import r3_diff_ibi as D
        cpu_fp, seedref, _ = D.load_lib(A.lib); prod = D.load_production(A.repo, "shim", cpu_fp, seedref, "lower")
        import cupy as cp; cp.asarray = np.asarray; gi = sys.modules["src.ddfp.gpu_immersion"]
        preprocess, run_ibi_v10, count_boundary_violations = prod.preprocess, prod.run_ibi_v10, prod.count_boundary_violations
        build_ispan_gpu, front_propagation_gpu, build_ispan_cpu, fp_cpu, sync = gi.build_ispan_gpu, gi.front_propagation_gpu, cpu_fp.build_ispan_cpu, cpu_fp.fp_cpu, (lambda: None)
        def verify_dwc_gpu(u, verbose=False):
            c = check_cpu(np.asarray(u), r1); v = sum(c.values()); return dict(dwc_ok=v == 0, violations=v, nan_skipped=0, total_blocks=-1, violation_rate=0.0)
        rec["gpu"] = "STUB (CPU stand-ins; timings meaningless)"
    # ---- inputs ---------------------------------------------------------------------------------------------------------------------------
    meta = json.loads(A.metadata.read_text()) if A.metadata else None
    if A.synthetic:
        rng = np.random.default_rng(7); vols = [(f"synthetic{i}", None, rng.integers(0, 256, (A.synthetic,) * 3, dtype=np.uint8)) for i in range(A.n)]
    else:
        if not A.brats_root: die(2, "--brats-root required")
        vols = [(p.parent.name, p, None) for p in select_cohort(A.brats_root, A.n, A.seed, meta)]
        rec["inputs_verified_against_metadata"] = bool(meta)
    ref = {}
    if A.ref_csv:
        import csv
        for r in csv.DictReader(open(A.ref_csv)): ref.setdefault((r["subject_id"], r["method"]), set()).add(r["output_sample_sha256"])

    KDD = 16 if A.backend == "cupy" else 3                  # production K=16 (benchmark); stub plumbing test uses K=3

    def gen(method, vol):
        """returns (u_host, extras). Mirrors benchmark_fixed_local.run_method but splits the DD-FP timers."""
        if method == "gpu_single_pass_k1":
            lo, hi, l = build_ispan_gpu(vol); g = front_propagation_gpu(lo, hi, l, verbose=False)[1:-1, 1:-1, 1:-1]; return cp.asnumpy(g), {}
        r = run_ibi_v10(vol, K=KDD, delta=1, max_rounds=A.max_rounds)
        return r["u_dd"], dict(R_star=int(r["R_star"]), boundary_z_orig=r["boundary_z_orig"])

    # ---- warm-ups (untimed): kernel compilation for every code path ---------------------------------------------------------------------
    wv = np.random.default_rng(1).integers(0, 256, (9, 9, 48 if A.backend == "cupy" else 12), dtype=np.uint8)
    for m in METHODS: gen(m, wv)
    verify_dwc_gpu(cp.asarray(np.zeros((9, 9, 9), np.float32)), verbose=False); sync()

    rows, out_arrays = [], {}
    for subj_i, (sid, path, svol) in enumerate(vols):
        vol = svol if svol is not None else preprocess(nib.load(path).get_fdata(dtype=np.float32))
        for rep in range(A.reps):
            for m in METHODS:
                sync(); t0 = time.perf_counter(); u, ex = gen(m, vol); sync(); t_gen = time.perf_counter() - t0
                row = dict(subject=sid, method=m, rep=rep, input_sha256=sha_f(path) if path else None, vol_sha256=sha_b(vol.tobytes()), shape=list(vol.shape), out_shape=list(u.shape),
                           t_gen_s=round(t_gen, 6), out_sha256=sha_b(np.ascontiguousarray(u).tobytes()), out_sample_sha256=sample_hash(u), R_star=ex.get("R_star"),
                           gpu_busy_after=(nvsmi("compute-apps=pid,process_name,used_memory") if A.backend == "cupy" else ""))
                row["gpu_other_procs_after"] = other_procs(row["gpu_busy_after"])          # R3.1: excludes this process
                if ex:                                                       # DD-FP seam scan, as inside the recorded benchmark
                    t0 = time.perf_counter(); row["seam_violations"] = int(count_boundary_violations(u, ex["boundary_z_orig"])); row["t_seam_s"] = round(time.perf_counter() - t0, 6)
                ug = None
                try:
                    sync(); t0 = time.perf_counter(); ug = cp.asarray(u); sync(); row["t_h2d_s"] = round(time.perf_counter() - t0, 6)
                    t0 = time.perf_counter(); g = verify_dwc_gpu(ug, verbose=False); sync(); row["t_global_gpu_s"] = round(time.perf_counter() - t0, 6)
                    row["global_gpu"] = {k: (v if isinstance(v, (int, float, bool)) else str(v)) for k, v in g.items()}
                finally:
                    del ug
                    if A.backend == "cupy": cp.get_default_memory_pool().free_all_blocks()
                if not A.no_cpu_check and subj_i < A.cpu_check_n:
                    t0 = time.perf_counter(); c = check_cpu(u, r1); row["t_cpu_indep_s"] = round(time.perf_counter() - t0, 3); row["cpu_indep_viol"] = c
                    row["gpu_cpu_agree"] = bool(row["global_gpu"]["violations"] == sum(c.values()))
                rf = ref.get((sid, m))
                if rf is not None: row["sample_hash_matches_recorded"] = row["out_sample_sha256"] in rf
                rows.append(row)
                (A.out / "rows.jsonl").open("a").write(json.dumps(row) + "\n"); print(json.dumps({k: row[k] for k in row if k not in ("input_sha256", "vol_sha256")}), flush=True)
                if rep == 0: out_arrays[(sid, m)] = u
        # exact comparisons between methods on this subject (K1 vs DD-FP) and optional sequential FP
        k1, dd = out_arrays.get((sid, "gpu_single_pass_k1")), out_arrays.get((sid, "ddfp_k16_d1"))
        cmpr = dict(subject=sid, k1_vs_ddfp_differing_cells=int((k1 != dd).sum()) if k1 is not None and dd is not None and k1.shape == dd.shape else None)
        if A.seq_subjects > len([r for r in rec.setdefault("pair_compare", [])]):
            t0 = time.perf_counter(); lo, hi, l = build_ispan_cpu(vol); sq = fp_cpu(lo, hi, l)[1:-1, 1:-1, 1:-1]; cmpr["t_seq_s"] = round(time.perf_counter() - t0, 2)
            cmpr["seq_vs_k1_cells"] = int((sq != k1).sum()); cmpr["seq_vs_ddfp_cells"] = int((sq != dd).sum()); cmpr["seq_sha256"] = sha_b(np.ascontiguousarray(sq).tobytes()); del sq
        rec["pair_compare"].append(cmpr); out_arrays.clear()
    # ---- aggregate --------------------------------------------------------------------------------------------------------------------------
    def agg(m, key):
        v = [r[key] for r in rows if r["method"] == m and key in r]; return dict(n=len(v), mean=round(st.mean(v), 6), sd=round(st.stdev(v), 6) if len(v) > 1 else None) if v else None
    rec["aggregate"] = {m: {k: agg(m, k) for k in ("t_gen_s", "t_seam_s", "t_h2d_s", "t_global_gpu_s", "t_cpu_indep_s")} for m in METHODS}
    rec["rows"] = rows; rec["blockwise_certification"] = "NOT_IMPLEMENTED_IN_PRODUCTION / NOT_MEASURED"
    rec["total_violations_global_gpu"] = sum(r["global_gpu"]["violations"] for r in rows)
    rec["gpu_cpu_disagreements"] = sum(1 for r in rows if r.get("gpu_cpu_agree") is False)
    rec["sample_hash_mismatch_vs_recorded"] = sum(1 for r in rows if r.get("sample_hash_matches_recorded") is False)
    (A.out / "summary.json").write_text(json.dumps(rec, indent=1)); print(json.dumps({k: rec[k] for k in ("aggregate", "total_violations_global_gpu", "gpu_cpu_disagreements", "sample_hash_mismatch_vs_recorded")}, indent=1))
    sys.exit(0 if rec["total_violations_global_gpu"] == 0 and rec["gpu_cpu_disagreements"] == 0 else 1)


if __name__ == "__main__":
    try: main()
    except SystemExit: raise
    except Exception: traceback.print_exc(); sys.exit(3)
