#!/usr/bin/env python3
"""
run_generic_gpu.py -- G5 candidate: generic pile-split worklist vs AF vs frozen on REAL CUDA (RTX 4090 server).  NOT EXECUTED by the author.

Question (CMPB novelty gate, fixed BEFORE any GPU number exists):
  Does a textbook GPU worklist (comparison/generic_worklists.py, variant pile/nearest-lower: Near-Far / IWPP style pile + scan-compaction split,
  same FP contract) preserve the frozen output AND state, and how fast is it relative to AF?

Per subject: (1) correctness of PILE against the frozen reference (output bytes, aux arrays, all counters, level_transitions,
closure_rounds_by_level); (2) timing of {frozen, af, pile} in a per-(subject,rep) pseudo-random order, one process, one GPU, same timed scope as
run_gpu_gate.py; (3) one traced pass recording pile sizes / split work; (4) optional memory sampler (peak device delta).

PRE-REGISTERED DECISION RULE (n >= 20 subjects, reps >= 3; applied to per-subject median times; r = t_AF / t_PILE):
  C0  PILE correctness must be 100% (output + state + rounds), otherwise the generic design is NOT FP-preserving at scale -> AF-specific design
      is validated as necessary (verdict "AF-NECESSARY").
  R1  median(r) >= 1.25 and min(r) >= 1.0 -> "PILE-DOMINATES-AF": a generic contract-aware worklist is faster; AF's per-visit gather is not the
      FP-specific mechanism worth claiming; recast the contribution as contract-preserving active execution (design-agnostic).
  R2  0.80 <= median(r) < 1.25 -> "COMPARABLE": no speed advantage for AF; AF differs only by memory footprint and its overflow fallback.
  R3  median(r) < 0.80 -> "AF-FASTER": AF has a measurable advantage over the generic design (report the cause from the traced pass).
Extra device memory is reported, not part of the speed rule.

Usage (server):  python3 -I comparison/run_generic_gpu.py --backend cupy --runner-dir ... --repo ... --lib ... --brats-root ... --metadata ... --n 20 --out OUT
Plumbing check (CPU):  python3 -I comparison/run_generic_gpu.py --backend shim --synthetic 3 --synthetic-size 8 --out OUT
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "comparison"))
import af_worklist as AF  # noqa: E402
import generic_worklists as GW  # noqa: E402
import run_gpu_gate as RG  # noqa: E402  (ShimEnv / CupyEnv / compare / sha reused unchanged)

PRE = dict(r1_median=1.25, r1_min=1.0, r2_lo=0.80)


def decide(summary, rows):
    subs = summary["subjects"]
    if len(subs) < 20 or summary["reps"] < 3:
        return dict(verdict="NOT-DECIDABLE (n<20 or reps<3)", rule=PRE)
    if not summary["pile_correct_all"]:
        return dict(verdict="AF-NECESSARY (generic pile design lost output/state equivalence)", rule=PRE)
    r = [s["ratio_af_over_pile"] for s in subs.values()]
    med, mn = st.median(r), min(r)
    if med >= PRE["r1_median"] and mn >= PRE["r1_min"]:
        v = "PILE-DOMINATES-AF"
    elif med >= PRE["r2_lo"]:
        v = "COMPARABLE"
    else:
        v = "AF-FASTER"
    return dict(verdict=v, median_ratio_af_over_pile=med, min_ratio=mn, max_ratio=max(r), rule=PRE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["cupy", "shim"], default="cupy")
    ap.add_argument("--runner-dir", type=Path); ap.add_argument("--repo", type=Path); ap.add_argument("--lib", type=Path)
    ap.add_argument("--brats-root", type=Path); ap.add_argument("--metadata", type=Path)
    ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--order-seed", type=int, default=20261023); ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cap", type=int, default=AF.DEFAULT_CAP)
    ap.add_argument("--no-mem", dest="mem", action="store_false")
    ap.add_argument("--synthetic", type=int, default=0); ap.add_argument("--synthetic-size", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    A = ap.parse_args()

    env = RG.ShimEnv() if A.backend == "shim" else RG.CupyEnv(A)
    cp = env.cp
    F = AF.load_frozen()
    A.out.mkdir(parents=True, exist_ok=True)
    rows_path = A.out / "rows.jsonl"
    if rows_path.exists() and rows_path.stat().st_size > 0:
        raise SystemExit(f"{rows_path} exists and is not empty: use a fresh --out")
    subjects = env.cohort(A)
    code = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in
            ("comparison/generic_worklists.py", "comparison/run_generic_gpu.py", "src/af_worklist.py", "frozen/gpu_native_worklist.py")}
    summary = dict(backend=env.name, cuda_executed=env.cuda, n_subjects=len(subjects), reps=A.reps, cap=A.cap, frozen_sha256=AF.FROZEN_SHA256,
                   code_sha256=code, python=platform.python_version(), platform=platform.platform(), subjects={}, pile_correct_all=True)
    if env.cuda:
        summary["device"] = env.be.device_info()

    def run(m, lo, hi, l):
        if m == "frozen":
            return env.frozen(cp, lo, hi, l)
        if m == "af":
            return AF.boutry_active_frontier(cp, lo, hi, l, "lower", frozen=F, frontier_cap=A.cap)
        return GW.generic_pile_worklist(cp, lo, hi, l, "nearest-lower", 256, list_cap=A.cap, pile_cap=None, frozen=F)

    for si, (sid, path, vol) in enumerate(subjects):
        lo, hi, l = env.prep(vol)
        rec = dict(subject=sid)
        env.free()
        ref = env.frozen(cp, lo, hi, l)
        got = GW.generic_pile_worklist(cp, lo, hi, l, "nearest-lower", 256, list_cap=A.cap, pile_cap=None, frozen=F, trace=True)
        bad = RG.compare(ref, got)
        rec["pile_correct"] = not bad
        rec["pile_mismatch"] = bad
        rec["output_sha256"] = RG.sha(ref[0]); rec["frozen_rounds"] = int(ref[1]["total_activation_rounds"])
        g = got[1]["gw"]; ps = np.array(g["pile_sizes"]); fs = np.array(g["frontier_sizes"])
        rec["pile_trace"] = dict(N=g["N"], visits=g["stats"]["visits"], rounds=g["stats"]["rounds"], max_pile=g["stats"]["max_pile"],
                                 median_pile_at_visit=float(np.median(ps)), p90_pile=float(np.percentile(ps, 90)),
                                 split_elements=g["stats"]["split_elements"], full_grid_elements_equivalent=g["stats"]["visits"] * g["N"],
                                 max_frontier=int(fs.max()))
        if bad:
            summary["pile_correct_all"] = False
        del ref, got
        for rep in range(A.reps):
            order = [("frozen", "af", "pile")[i] for i in np.random.default_rng([A.order_seed, si, rep]).permutation(3)]
            for pos, m in enumerate(order):
                env.free(); env.sync()
                t0 = time.perf_counter()
                out = run(m, lo, hi, l)
                env.sync()
                dt = time.perf_counter() - t0
                row = dict(subject=sid, rep=rep, method=m, order_position=pos, t_total_s=dt, out_sha256=RG.sha(out[0]),
                           rounds=out[1]["total_activation_rounds"])
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                rec.setdefault("times", {}).setdefault(m, []).append(dt)
                if row["out_sha256"] != rec["output_sha256"] or row["rounds"] != rec["frozen_rounds"]:
                    summary["pile_correct_all"] = False
                    rec.setdefault("timed_run_mismatch", []).append(dict(rep=rep, method=m))
                del out
        med = {m: st.median(v) for m, v in rec["times"].items()}
        rec["median_s"] = med
        rec["speedup_af_vs_frozen"] = med["frozen"] / med["af"]
        rec["speedup_pile_vs_frozen"] = med["frozen"] / med["pile"]
        rec["ratio_af_over_pile"] = med["af"] / med["pile"]
        if A.mem and env.cuda:
            mem = {}
            for m in ("frozen", "af", "pile"):
                env.free(); env.sync()
                with env.sampler() as s:
                    out = run(m, lo, hi, l)
                    env.sync()
                mem[m] = s.result(); del out
            rec["memory_peak_vram_delta_mib"] = {m: mem[m]["peak_vram_delta_mib"] for m in mem}
        summary["subjects"][sid] = rec
        (A.out / "summary.partial.json").write_text(json.dumps(summary, indent=1, default=str))
        print(f"[{si+1}/{len(subjects)}] {sid}: pile_correct={rec['pile_correct']} AF/frozen={rec['speedup_af_vs_frozen']:.2f}x "
              f"PILE/frozen={rec['speedup_pile_vs_frozen']:.2f}x t_AF/t_PILE={rec['ratio_af_over_pile']:.2f}", flush=True)
    rows = [json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]
    V = decide(summary, rows)
    if not env.cuda:
        V["verdict"] = "NOT-MEASURED (shim run; plumbing only)"
    summary["decision"] = V
    (A.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in V.items() if k != "rule"}, indent=1))
    return 0 if summary["pile_correct_all"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
