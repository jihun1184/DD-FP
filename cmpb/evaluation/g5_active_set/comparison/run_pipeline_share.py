#!/usr/bin/env python3
"""
run_pipeline_share.py -- what share of a DWC-interpolation pipeline is the FP step, and what does AF change end to end?  NOT EXECUTED ON A GPU by the author.

Stages timed per subject (median over --reps, one process, wall clock with device sync; same FP timed scope as run_gpu_gate.py):
  S1 prep        : env.prep(vol)                       (build_ispan; GPU on the server)
  S2 fp_frozen   : frozen K=1 FP        S2' fp_af : AF FP        (output hashes are compared; mismatch => run flagged)
  S3 d2h/post    : np.asarray of the FP output (already included in the FP return of the frozen/AF functions; reported as 0 here)
  S4 downstream  : OPTIONAL CPU stand-ins applied to the FP output (scipy.ndimage): label (6-conn), binary_closing, gaussian_filter.
                   These are NOT the project's downstream modules; they only bound a typical post-processing cost. Replace with the real stages
                   by passing --extra-stage module:function (callable(np.ndarray)->any) one or more times.
Reported per subject and cohort: share_fp = t_fp / (t_prep + t_fp + t_down), and the end-to-end speedup (frozen pipeline / AF pipeline).
Amdahl cross-check: E2E_speedup = 1 / (1 - p + p/s) with p = share_fp(frozen), s = t_fp_frozen / t_fp_af.

No decision rule is attached: this is a descriptive measurement.  Plumbing check:  python3 -I comparison/run_pipeline_share.py --backend shim --synthetic 2 --synthetic-size 8 --out OUT
Server:  python3 -I comparison/run_pipeline_share.py --backend cupy --runner-dir ... --repo ... --lib ... --brats-root ... --metadata ... --n 20 --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
import af_worklist as AF  # noqa: E402
import run_gpu_gate as RG  # noqa: E402


def stand_in_stages():
    from scipy import ndimage as ndi
    return {
        "label6": lambda a: ndi.label(np.isfinite(a)),
        "binary_closing": lambda a: ndi.binary_closing(np.isfinite(a), iterations=1),
        "gaussian": lambda a: ndi.gaussian_filter(np.nan_to_num(a), 1.0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["cupy", "shim"], default="cupy")
    ap.add_argument("--runner-dir", type=Path); ap.add_argument("--repo", type=Path); ap.add_argument("--lib", type=Path)
    ap.add_argument("--brats-root", type=Path); ap.add_argument("--metadata", type=Path)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--cap", type=int, default=AF.DEFAULT_CAP)
    ap.add_argument("--no-standins", action="store_true")
    ap.add_argument("--extra-stage", action="append", default=[], help="module:function, callable(np.ndarray)")
    ap.add_argument("--synthetic", type=int, default=0); ap.add_argument("--synthetic-size", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    A = ap.parse_args()

    env = RG.ShimEnv() if A.backend == "shim" else RG.CupyEnv(A)
    cp = env.cp; F = AF.load_frozen()
    stages = {} if A.no_standins else stand_in_stages()
    for spec in A.extra_stage:
        mod, fn = spec.split(":"); stages[spec] = getattr(importlib.import_module(mod), fn)
    A.out.mkdir(parents=True, exist_ok=True)
    res = dict(backend=env.name, cuda_executed=env.cuda, reps=A.reps, stages=list(stages), subjects={}, flagged=[])
    for sid, _p, vol in env.cohort(A):
        T = {k: [] for k in ("prep", "fp_frozen", "fp_af", "down")}
        per_stage = {k: [] for k in stages}
        ref_hash = None
        for rep in range(A.reps):
            env.free(); env.sync()
            t0 = time.perf_counter(); lo, hi, l = env.prep(vol); env.sync(); T["prep"].append(time.perf_counter() - t0)
            for m in (("frozen", "af") if rep % 2 == 0 else ("af", "frozen")):
                env.free(); env.sync()
                t0 = time.perf_counter()
                out = env.frozen(cp, lo, hi, l) if m == "frozen" else AF.boutry_active_frontier(cp, lo, hi, l, "lower", frozen=F, frontier_cap=A.cap)
                env.sync(); T["fp_" + m].append(time.perf_counter() - t0)
                h = hashlib.sha256(np.ascontiguousarray(out[0]).tobytes()).hexdigest()
                if ref_hash is None:
                    ref_hash = h
                elif h != ref_hash:
                    res["flagged"].append(dict(subject=sid, rep=rep, method=m, note="output hash differs"))
                arr = np.asarray(out[0])
            d = 0.0
            for k, f in stages.items():
                t0 = time.perf_counter(); f(arr); dt = time.perf_counter() - t0; per_stage[k].append(dt); d += dt
            T["down"].append(d)
        med = {k: st.median(v) for k, v in T.items()}
        pipe_f = med["prep"] + med["fp_frozen"] + med["down"]; pipe_a = med["prep"] + med["fp_af"] + med["down"]
        p = med["fp_frozen"] / pipe_f; s = med["fp_frozen"] / med["fp_af"]
        res["subjects"][sid] = dict(median_s=med, stage_median_s={k: st.median(v) for k, v in per_stage.items()}, share_fp_frozen=p,
                                    share_fp_af=med["fp_af"] / pipe_a, e2e_speedup=pipe_f / pipe_a, fp_speedup=s,
                                    amdahl_e2e=1 / (1 - p + p / s), pipeline_frozen_s=pipe_f, pipeline_af_s=pipe_a)
        print(f"{sid}: FP share(frozen)={p:.3f} FP x{s:.2f} E2E x{pipe_f/pipe_a:.2f}", flush=True)
    S = list(res["subjects"].values())
    if S:
        res["cohort"] = {k: st.median(x[k] for x in S) for k in ("share_fp_frozen", "share_fp_af", "e2e_speedup", "fp_speedup", "amdahl_e2e")}
    res["note"] = "descriptive only" + ("" if env.cuda else "; SHIM run: timings are CPU-thread plumbing values, NOT evidence")
    (A.out / "pipeline_share.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res.get("cohort", {}), indent=1))
    return 0 if not res["flagged"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
