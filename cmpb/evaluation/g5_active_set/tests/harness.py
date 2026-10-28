#!/usr/bin/env python3
"""
harness.py -- A1 correctness gate on the CPU execution shim ("SHIM" evidence tier; no GPU involved).

For every synthetic case it runs
  (1) the UNMODIFIED frozen ``boutry_bucket_gpu`` (v1.1, SHA 853bf33a) on FakeCupy, snapshotting all state arrays
      after every ``expand_frontier`` launch  == frozen round boundary,
  (2) the new active-frontier driver for several list capacities (huge, and tiny ones that force gather
      overflow and mid-visit fallback), snapshotting after every round,
  (3) the frozen CPU state machine ``cpu_shadow_worklist_v1`` (final output + counters),
  (4) a sequential, random-pop-order bucket-FP model (Lemma C check at every level-visit end),
and compares per-round hashes of u / discovered / processed / scheduled_level / pending_count / pending_bits /
schedule_dir_count / frontier, the final output, all counters, the level sequence and closure_rounds_by_level.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import af_worklist as AF  # noqa: E402
from fakecupy import FakeCupy  # noqa: E402

KEYS = ("u", "discovered", "processed", "scheduled_level", "pending_count", "pending_bits", "schedule_dir_count", "frontier")
META_KEYS = ("nan_count", "discovered_count", "processed_count", "pending_total_final", "pending_bitset_nonzero_words_final",
             "schedule_direction_counts", "scheduler_transition_count", "backward_scheduler_transition_count",
             "equidistant_next_level_tie_events", "total_activation_rounds", "closure_rounds_by_level", "level_transitions")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FROZEN = AF.load_frozen()
CPU_FP = _load(ROOT / "frozen" / "cpu_fp.py", "cpu_fp_frozen")


def H(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def snap(arrs) -> dict:
    return {k: H(arrs[k]) for k in KEYS}


# ----------------------------------------------------------------------------------------------------------
def run_frozen(cp, lo, hi, l, block):
    seen, rounds = {}, []

    def hook(name, named, g, b):
        for k, v in named.items():
            if isinstance(v, np.ndarray):
                seen[k] = v
        if name == "expand_frontier":
            rounds.append(dict(level=int(named["current_level"]), h=snap(seen)))

    cp.hooks.append(hook)
    try:
        out, meta, aux = FROZEN.boutry_bucket_gpu(cp, lo, hi, l, "lower", block)
    finally:
        cp.hooks.remove(hook)
    return out, meta, aux, rounds


def run_af(cp, lo, hi, l, block, cap, guard=8, new_code=None, hooks=None):
    rounds = []

    def rh(info):
        rounds.append(dict(level=int(info["level"]), kind=info["kind"], h=snap(info["arrays"]),
                           pend0=int(info["arrays"]["pending_count"][info["level"]])))

    for h in hooks or []:
        cp.hooks.append(h)
    try:
        out, meta, aux = AF.boutry_active_frontier(cp, lo, hi, l, "lower", block, cap, frozen=FROZEN, new_code=new_code,
                                                   round_hook=rh, guard=guard, trace=True)
    finally:
        for h in hooks or []:
            cp.hooks.remove(h)
    return out, meta, aux, rounds


def compare_runs(ref, got):
    """ref/got = (out, meta, aux, rounds).  Returns list of human-readable discrepancies (empty == identical)."""
    bad = []
    ro, rm, ra, rr = ref
    go, gm, ga, gr = got
    if not np.array_equal(ro, go, equal_nan=True) or H(ro) != H(go):
        bad.append("final output differs")
    for k in META_KEYS:
        if rm[k] != gm[k]:
            bad.append(f"meta[{k}] differs")
    for k in ra:
        if not np.array_equal(ra[k], ga[k]):
            bad.append(f"aux[{k}] differs")
    if len(rr) != len(gr):
        bad.append(f"round count {len(rr)} vs {len(gr)}")
    for i, (a, b) in enumerate(zip(rr, gr)):
        if a["level"] != b["level"]:
            bad.append(f"round {i}: level {a['level']} vs {b['level']}")
            break
        d = [k for k in KEYS if a["h"][k] != b["h"][k]]
        if d:
            bad.append(f"round {i} (level {a['level']}): state differs in {d}")
            break
    return bad


# ----------------------------------------------------------------------------------------------------------
def seq_visit_model(lo, hi, l0, rng):
    """Sequential bucket FP, lower tie, RANDOM pop order inside a level.  Returns the state hashes at every visit end."""
    Wp, Hp, Dp = lo.shape
    N = lo.size
    lof, hif = lo.ravel(), hi.ravel()
    u = np.full(N, np.nan, np.float32)
    disc = np.zeros(N, np.int32); proc = np.zeros(N, np.int32); sched = np.full(N, -1, np.int32)
    buckets = [[] for _ in range(256)]
    dirs = np.zeros(3, np.int32)

    def nbrs(idx):
        x, rem = divmod(idx, Hp * Dp); y, z = divmod(rem, Dp)
        for dx, dy, dz in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)):
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < Wp and 0 <= ny < Hp and 0 <= nz < Dp:
                yield (nx * Hp + ny) * Dp + nz

    disc[0] = 1; sched[0] = l0; buckets[l0].append(0)
    cur, visits = l0, []
    while True:
        b = buckets[cur]
        while b:
            idx = b.pop(rng.randrange(len(b)))
            u[idx] = np.float32(cur); proc[idx] = 1
            for q in nbrs(idx):
                if disc[q]:
                    continue
                disc[q] = 1
                lv = float(cur)
                s = lof[q] if lv < lof[q] else (hif[q] if lv > hif[q] else lv)
                t = int(round(float(s)))
                sched[q] = t; buckets[t].append(q)
                dirs[0 if t < cur else (1 if t == cur else 2)] += 1
        cnt = np.array([len(x) for x in buckets], np.int32)
        bits = np.zeros(8, np.uint32)
        for lv in range(256):
            if cnt[lv] > 0:
                bits[lv >> 5] |= np.uint32(1 << (lv & 31))
        visits.append(dict(level=cur, h=dict(u=H(u), discovered=H(disc), processed=H(proc), scheduled_level=H(sched),
                                             pending_count=H(cnt), pending_bits=H(bits), schedule_dir_count=H(dirs))))
        occ = [lv for lv in range(256) if cnt[lv] > 0]
        if not occ:
            break
        d = min(abs(lv - cur) for lv in occ)
        cur = min(lv for lv in occ if abs(lv - cur) == d)  # lower tie
    return u, visits


def visit_ends(rounds):
    return [dict(level=r["level"], h=r["h"]) for r in rounds if r["pend0"] == 0]


def lemma_c_check(rounds, visits):
    ends = visit_ends(rounds)
    if len(ends) != len(visits):
        return [f"visit count {len(ends)} vs sequential {len(visits)}"]
    for i, (a, b) in enumerate(zip(ends, visits)):
        if a["level"] != b["level"]:
            return [f"visit {i}: level {a['level']} vs {b['level']}"]
        d = [k for k in b["h"] if a["h"][k] != b["h"][k]]
        if d:
            return [f"visit {i} (level {a['level']}): state differs in {d}"]
    return []


# ----------------------------------------------------------------------------------------------------------
def gen_case(rng: random.Random, nprng: np.random.Generator, i: int):
    kinds = ["vol_alpha", "vol_uniform", "interval", "interval_wide", "smooth", "tube", "plateau"]
    kind = kinds[i % len(kinds)]
    if kind == "interval" or kind == "interval_wide":
        shp = (rng.randint(2, 7), rng.randint(2, 7), rng.randint(2, 8))
        step = rng.choice([1, 5, 10, 20, 50])
        levels = rng.choice([2, 4, 8, 12])
        wmax = 2 if kind == "interval" else 6
        l = np.minimum(nprng.integers(0, levels, shp) * step, 255)
        w = nprng.integers(0, wmax, shp) * step
        lo = l.astype(np.float32); hi = np.minimum(255, l + w).astype(np.float32)
        lo[0, 0, 0] = 0; hi[0, 0, 0] = 0
        return kind, lo, hi, 0.0
    if kind == "vol_alpha":
        shp = (rng.randint(2, 6), rng.randint(2, 6), rng.randint(2, 6))
        alph = rng.choice([[0, 255], [0, 100, 255], [0, 60, 120, 255], [10, 200], [0, 1, 2, 3]])
        vol = nprng.choice(alph, shp).astype(np.uint8)
    elif kind == "vol_uniform":
        shp = (rng.randint(2, 5), rng.randint(2, 5), rng.randint(2, 6))
        vol = nprng.integers(0, 256, shp).astype(np.uint8)
    elif kind == "smooth":
        from scipy.ndimage import gaussian_filter
        shp = (rng.randint(5, 9), rng.randint(5, 9), rng.randint(5, 9))
        f = gaussian_filter(nprng.standard_normal(shp), rng.choice([0.8, 1.2, 2.0]))
        f = (f - f.min()) / (np.ptp(f) + 1e-9)
        k = rng.choice([3, 8, 40, 256])
        vol = np.floor(f * (k - 1e-6)).astype(np.int64) * (255 // max(1, k - 1))
        vol = np.clip(vol, 0, 255).astype(np.uint8)
    elif kind == "tube":
        shp = (rng.randint(5, 9), rng.randint(5, 9), rng.randint(5, 9))
        vol = np.full(shp, rng.choice([255, 200, 90]), np.uint8)
        p = [rng.randrange(s) for s in shp]
        val = rng.choice([0, 30, 255])
        for _ in range(rng.randint(10, 60)):
            vol[tuple(p)] = val
            ax = rng.randrange(3); p[ax] = min(shp[ax] - 1, max(0, p[ax] + rng.choice([-1, 1])))
    else:  # plateau: one level almost everywhere (deep same-level closure) with a few other cells
        shp = (rng.randint(4, 9), rng.randint(4, 9), rng.randint(4, 9))
        vol = np.full(shp, rng.choice([128, 77]), np.uint8)
        for _ in range(rng.randint(1, 6)):
            vol[tuple(rng.randrange(s) for s in shp)] = rng.choice([0, 255, 10, 200])
    lo, hi, linf = CPU_FP.build_ispan_cpu(vol)
    return kind, lo, hi, linf


def run_cases(args):
    cp = FakeCupy(seed=args.seed, threads=args.threads, jitter=args.jitter)
    rng = random.Random(args.seed)
    nprng = np.random.default_rng(args.seed)
    res = dict(cases=0, failures=[], by_kind={}, overflow=dict(gather_overflow_visits=0, mid_fallback_visits=0, af_runs=0,
                                                              af_runs_with_fallback=0, rounds_list=0, rounds_frozen=0),
               lemma_c=dict(cases=0, visits_compared=0, failures=0),
               vs_shadow=dict(cases=0, failures=0), total_rounds=0, total_cells=0, tie_events=0, backward=0, runs_af=0)
    t0 = time.time()
    for i in range(args.cases):
        kind, lo, hi, linf = gen_case(rng, nprng, i)
        block = rng.choice([8, 16, 32, 64])
        ref = run_frozen(cp, lo, hi, linf, block)
        rm = ref[1]
        # frozen CPU state machine (independent oracle of the final output and counters)
        sh_out, sh_meta = FROZEN.cpu_shadow_worklist_v1(lo, hi, linf, "lower")
        bad_sh = []
        if not np.array_equal(sh_out, ref[0], equal_nan=True):
            bad_sh.append("cpu_shadow output != frozen-on-shim output")
        for k in ("total_activation_rounds", "level_transitions", "schedule_direction_counts",
                  "equidistant_next_level_tie_events", "backward_scheduler_transition_count"):
            if sh_meta[k] != rm[k]:
                bad_sh.append(f"cpu_shadow {k} differs")
        res["vs_shadow"]["cases"] += 1
        if bad_sh:
            res["vs_shadow"]["failures"] += 1
            res["failures"].append(dict(case=i, kind=kind, where="shadow", detail=bad_sh))
        # sequential random-order model: Lemma C at visit ends
        seq_u, visits = seq_visit_model(lo, hi, int(linf), rng)
        # AF runs
        caps = [None] + rng.sample([1, 2, 3, 5, 9, 17, 40], args.small_caps)
        for cap in caps:
            got = run_af(cp, lo, hi, linf, block, cap)
            bad = compare_runs(ref, got)
            st = got[1]["af"]["stats"]
            res["runs_af"] += 1
            res["overflow"]["af_runs"] += 1
            res["overflow"]["gather_overflow_visits"] += st["visits_gather_overflow"]
            res["overflow"]["mid_fallback_visits"] += st["visits_mid_fallback"]
            res["overflow"]["rounds_list"] += st["rounds_list"]
            res["overflow"]["rounds_frozen"] += st["rounds_frozen"]
            res["overflow"]["af_runs_with_fallback"] += int(st["visits_gather_overflow"] + st["visits_mid_fallback"] > 0)
            if bad:
                res["failures"].append(dict(case=i, kind=kind, cap=cap, block=block, where="af_vs_frozen", detail=bad[:3]))
            if cap is None or cap == caps[1]:
                lc = lemma_c_check(got[3], visits)
                res["lemma_c"]["cases"] += 1
                res["lemma_c"]["visits_compared"] += len(visits)
                if not np.array_equal(seq_u.reshape(got[0].shape), got[0], equal_nan=True):
                    lc.append("final output differs from sequential random-order model")
                if lc:
                    res["lemma_c"]["failures"] += 1
                    res["failures"].append(dict(case=i, kind=kind, cap=cap, where="lemma_c", detail=lc[:3]))
        res["cases"] += 1
        bk = res["by_kind"].setdefault(kind, 0); res["by_kind"][kind] = bk + 1
        res["total_rounds"] += rm["total_activation_rounds"]
        res["total_cells"] += int(lo.size)
        res["tie_events"] += rm["equidistant_next_level_tie_events"]
        res["backward"] += rm["backward_scheduler_transition_count"]
        if args.progress and (i + 1) % args.progress == 0:
            print(f"  {i+1}/{args.cases} cases, failures={len(res['failures'])}, {time.time()-t0:.0f}s", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    res["config"] = dict(seed=args.seed, cases=args.cases, threads=args.threads, jitter=args.jitter, small_caps=args.small_caps,
                         frozen_sha256=AF.FROZEN_SHA256)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--small-caps", type=int, default=2)
    ap.add_argument("--progress", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    res = run_cases(args)
    if args.out:
        args.out.write_text(json.dumps(res, indent=1))
    brief = {k: res[k] for k in ("cases", "runs_af", "seconds", "by_kind", "overflow", "lemma_c", "vs_shadow", "total_rounds", "total_cells",
                                  "tie_events", "backward")}
    brief["n_failures"] = len(res["failures"])
    print(json.dumps(brief, indent=1))
    for f in res["failures"][:8]:
        print("FAIL", f)
    return 1 if res["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
