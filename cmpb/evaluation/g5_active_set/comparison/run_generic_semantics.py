#!/usr/bin/env python3
"""
run_generic_semantics.py -- which generic GPU worklist designs preserve the frozen FP contract?

For every synthetic case (tests/harness.py generators) the reference is the UNMODIFIED frozen ``boutry_bucket_gpu`` (SHA 853bf33a...).
Compared variants (comparison/generic_worklists.py):
  AF                      sanity row (already covered by the A1/G3/G4 gates)
  pile/nearest-lower      generic Near-Far/IWPP-style pile+split worklist, FP contract (expected: everything identical)
  pile/nearest-higher     tie policy "higher": compared with frozen "higher" (state) and frozen "lower" (output)
  pile/lowest-first, pile/monotone-wrap, pile/highest-first, level-agnostic-frontier   controls (expected: output differs)
  pile/nearest-lower with tiny list/pile caps    generic code has NO fallback: how often does it fail?
Per variant it reports: output identical, full state identical (all aux arrays + all counters + level sequence + rounds), rounds identical.

Backends: --backend shim (CPU threads, evidence tier SHIM) or --backend cupy (run on the GPU server; NOT executed by the author).
Usage: python3 -I comparison/run_generic_semantics.py --cases 140 --out results/generic_semantics_shim.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "comparison"))
import af_worklist as AF  # noqa: E402
import generic_worklists as GW  # noqa: E402
import harness as Hh  # noqa: E402

META_KEYS = Hh.META_KEYS


def state_diff(ref, got):
    """list of differing components (empty == identical in everything the frozen run reports)."""
    ro, rm, ra = ref[:3]
    go, gm, ga = got[:3]
    bad = []
    if not np.array_equal(ro, go, equal_nan=True):
        bad.append("output")
    for k in META_KEYS:
        if rm[k] != gm[k]:
            bad.append("meta:" + k)
    for k in ra:
        if not np.array_equal(ra[k], ga[k]):
            bad.append("aux:" + k)
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["shim", "cupy"], default="shim")
    ap.add_argument("--cases", type=int, default=140)
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.backend == "cupy":
        import cupy as cp
        cp.cuda.Device(0).use()
    else:
        from fakecupy import FakeCupy
        cp = FakeCupy(seed=a.seed, threads=a.threads)
    F = AF.load_frozen()
    rng = random.Random(a.seed); nprng = np.random.default_rng(a.seed)
    VAR = ["AF", "pile/nearest-lower", "pile/nearest-higher@higher-ref", "pile/nearest-higher@lower-ref", "pile/lowest-first",
           "pile/monotone-wrap", "pile/highest-first", "level-agnostic-frontier"]
    R = {v: dict(runs=0, output_identical=0, state_identical=0, rounds_identical=0, errors=0, first_diffs={}) for v in VAR}
    R["pile/nearest-lower@tiny-cap"] = dict(runs=0, capacity_errors=0, identical_when_completed=0, completed=0)
    by_kind = {}
    t0 = time.time()
    tot = dict(cases=0, rounds=0, cells=0, ties=0, backward=0, ties_cases=0)

    def rec(v, ref, got_fn):
        r = R[v]; r["runs"] += 1
        try:
            got = got_fn()
        except Exception as e:  # noqa: BLE001
            r["errors"] += 1; r.setdefault("error_msgs", []).append(str(e)[:120]); return
        d = state_diff(ref, got) if got[2] else (["output"] if not np.array_equal(ref[0], got[0], equal_nan=True) else [])
        out_ok = "output" not in d
        r["output_identical"] += int(out_ok)
        if got[2]:
            r["state_identical"] += int(not d)
            r["rounds_identical"] += int(got[1]["total_activation_rounds"] == ref[1]["total_activation_rounds"])
        else:
            r["rounds_identical"] += int(got[1]["total_activation_rounds"] == ref[1]["total_activation_rounds"])
        for x in d[:2]:
            r["first_diffs"][x] = r["first_diffs"].get(x, 0) + 1

    for i in range(a.cases):
        kind, lo, hi, linf = Hh.gen_case(rng, nprng, i)
        block = rng.choice([8, 16, 32, 64])
        ref = F.boutry_bucket_gpu(cp, lo, hi, linf, "lower", block)
        refh = F.boutry_bucket_gpu(cp, lo, hi, linf, "higher", block)
        rec("AF", ref, lambda: AF.boutry_active_frontier(cp, lo, hi, linf, "lower", block, None, frozen=F))
        rec("pile/nearest-lower", ref, lambda: GW.generic_pile_worklist(cp, lo, hi, linf, "nearest-lower", block, frozen=F))
        rec("pile/nearest-higher@higher-ref", refh, lambda: GW.generic_pile_worklist(cp, lo, hi, linf, "nearest-higher", block, frozen=F))
        rec("pile/nearest-higher@lower-ref", ref, lambda: GW.generic_pile_worklist(cp, lo, hi, linf, "nearest-higher", block, frozen=F))
        for pol in ("lowest-first", "monotone-wrap", "highest-first"):
            rec("pile/" + pol, ref, lambda pol=pol: GW.generic_pile_worklist(cp, lo, hi, linf, pol, block, frozen=F))
        rec("level-agnostic-frontier", ref, lambda: GW.generic_level_agnostic_frontier(cp, lo, hi, linf, block, frozen=F))
        # no-fallback behaviour at small capacities
        for cap in rng.sample([1, 2, 3, 5, 9, 17, 40], 2):
            r = R["pile/nearest-lower@tiny-cap"]; r["runs"] += 1
            try:
                got = GW.generic_pile_worklist(cp, lo, hi, linf, "nearest-lower", block, list_cap=cap, pile_cap=cap, frozen=F)
            except RuntimeError as e:
                if "capacity" in str(e):
                    r["capacity_errors"] += 1; continue
                raise
            r["completed"] += 1; r["identical_when_completed"] += int(not state_diff(ref, got))
        m = ref[1]
        k = by_kind.setdefault(kind, dict(cases=0, tie_cases=0, mean_rounds=0.0))
        k["cases"] += 1; k["tie_cases"] += int(m["equidistant_next_level_tie_events"] > 0); k["mean_rounds"] += m["total_activation_rounds"]
        tot["cases"] += 1; tot["rounds"] += m["total_activation_rounds"]; tot["cells"] += int(lo.size)
        tot["ties"] += m["equidistant_next_level_tie_events"]; tot["backward"] += m["backward_scheduler_transition_count"]
        tot["ties_cases"] += int(m["equidistant_next_level_tie_events"] > 0)
    for k in by_kind.values():
        k["mean_rounds"] = round(k["mean_rounds"] / k["cases"], 1)
    res = dict(backend=a.backend, frozen_sha256=AF.FROZEN_SHA256, seed=a.seed, cases=a.cases, seconds=round(time.time() - t0, 1),
               totals=tot, by_kind=by_kind, variants=R)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "by_kind"}, indent=1))


if __name__ == "__main__":
    main()
