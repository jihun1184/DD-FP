#!/usr/bin/env python3
"""
run_gpu_gate.py -- A1 correctness (BraTS) and performance gate for the active-frontier prototype.  RUN THIS ON THE RTX 4090 SERVER.

STATUS OF THIS FILE (v2, post-G3).  v1 (SHA ae2d38dd..., kept byte-identical in tests/legacy/run_gpu_gate_G3_as_executed.py) produced the
G3 evidence on the RTX 4090.  v2 changes ONLY (a) the verdict logic -> tests/gate_verdict.py (fail-closed: no GO without a memory measurement,
no performance verdict when correctness is not PASS, no cohort GO for n<20), (b) timed runs are re-checked against the reference hash/round count
and rows.jsonl must be fresh, (c) code/environment hashes are written into summary.json, (d) default --small-cap 20000 (200000 left the
fallback unexercised on 14 of the 20 G3 subjects whose largest frontier is below 200000).  The measured path (kernels, timing scope, memory
sampler) is unchanged.  v2 has been exercised only with ``--backend shim --synthetic`` (CPU threads); NOT yet run on a GPU.

What it does, per subject (same cohort / preprocessing / ispan builder as the R7 runner):
  1. CORRECTNESS  frozen production K=1 path (seed_aware_boutry_bucket_gpu(boundary_seeds=None) -> frozen v1.1)
                  vs ``boutry_active_frontier``: padded output bytes, discovered/processed/scheduled_level/pending arrays,
                  every counter, level_transitions, closure_rounds_by_level.   Plus a forced-small-cap pass (fallback on real data).
  2. TIMING       reps x {frozen, af} in a pseudo-random, per-(subject,rep) order, in ONE process on ONE GPU; each run starts
                  with pool release + sync and ends with sync before the clock stops.  Timed scope of both = the full function call
                  (interval validation, kernels, final D2H).  Hashes are computed after the clock stops.
  3. BREAKDOWN    separate, intrusive pass (not used for the headline numbers): gather / fused-round / frozen activate+expand /
                  select kernel GPU time (events), host sync wait, scalar D2H, final-array D2H, launch time, round & visit counts,
                  per-round frontier sizes and per-visit closure depth.
  4. MEMORY       separate pass with the polling sampler of r7_common (peak device delta, RSS) and CuPy pool bytes.
Gate (fixed in advance, MECH_GATE 5.3 / review): correctness must be 100%;  performance GO if median cohort speedup >= 2.0 and
every subject >= 1.3;  NO-GO if median < 1.3;  otherwise HOLD.  Extra device memory must be measured for every subject and <= 100 MiB for GO.
The exact (fail-closed) rule and the gather-share aggregation rule are in tests/gate_verdict.py.
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))
import af_worklist as AF  # noqa: E402
import gate_verdict as GV  # noqa: E402

META_KEYS = ("nan_count", "discovered_count", "processed_count", "pending_total_final", "pending_bitset_nonzero_words_final",
             "schedule_direction_counts", "scheduler_transition_count", "backward_scheduler_transition_count",
             "equidistant_next_level_tie_events", "total_activation_rounds", "closure_rounds_by_level", "level_transitions")


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------------------
class ProfiledCp:
    """Transparent proxy around cupy that times kernel launches (events), stream syncs and asnumpy copies.  Intrusive."""

    def __init__(self, cp):
        self._cp = cp
        self.host = {"sync_s": 0.0, "d2h_scalar_s": 0.0, "d2h_big_s": 0.0, "launch_s": 0.0}
        self.n = {"sync": 0, "d2h_scalar": 0, "d2h_big": 0}
        self.ev = []
        self.gpu_ms = {}
        proxy = self

        class _Null:
            @staticmethod
            def synchronize():
                t = time.perf_counter(); cp.cuda.Stream.null.synchronize()
                proxy.host["sync_s"] += time.perf_counter() - t; proxy.n["sync"] += 1

        class _Stream:
            null = _Null

        class _Cuda:
            Stream = _Stream

            def __getattr__(self, k):
                return getattr(cp.cuda, k)

        self.cuda = _Cuda()

    def __getattr__(self, k):
        return getattr(self._cp, k)

    def asnumpy(self, a):
        t = time.perf_counter(); r = self._cp.asnumpy(a); dt = time.perf_counter() - t
        key = "d2h_big" if getattr(a, "nbytes", 0) > (1 << 20) else "d2h_scalar"
        self.host[key + "_s"] += dt; self.n[key] += 1
        return r

    def RawModule(self, code, options=()):
        m = self._cp.RawModule(code=code, options=options)
        proxy = self

        class _M:
            def get_function(self, name):
                k = m.get_function(name)

                def call(grid, block, args):
                    e0 = proxy._cp.cuda.Event(); e1 = proxy._cp.cuda.Event()
                    t = time.perf_counter(); e0.record(); k(grid, block, args); e1.record()
                    proxy.host["launch_s"] += time.perf_counter() - t
                    proxy.ev.append((name, e0, e1))
                return call
        return _M()

    def resolve(self):
        for name, e0, e1 in self.ev:
            e1.synchronize()
            self.gpu_ms[name] = self.gpu_ms.get(name, 0.0) + float(self._cp.cuda.get_elapsed_time(e0, e1))
        self.ev.clear()


# ---------------------------------------------------------------------------------------------------------------
class ShimEnv:
    """CPU plumbing test: synthetic volumes, kernels on CPU threads."""
    name, cuda = "shim", False

    def __init__(self):
        sys.path.insert(0, str(ROOT / "tests"))
        from fakecupy import FakeCupy
        import harness as Hh
        self.cp = FakeCupy(seed=3, threads=4)
        self.Hh, self.F = Hh, Hh.FROZEN

    def cohort(self, A):
        rng = np.random.default_rng(7)
        return [(f"synthetic{i}", None, rng.integers(0, 256, (A.synthetic_size,) * 3, dtype=np.uint8)) for i in range(A.synthetic)]

    def prep(self, vol):
        return self.Hh.CPU_FP.build_ispan_cpu(vol)

    def frozen(self, cp, lo, hi, l):
        return self.F.boutry_bucket_gpu(cp, lo, hi, l, "lower")

    def free(self): pass
    def sync(self): pass
    def memstats(self): return {}
    def sampler(self): return None


class CupyEnv:
    name, cuda = "cupy", True

    def __init__(self, A):
        sys.path.insert(0, str(A.runner_dir))
        import r7_common as C
        self.C = C
        self.be = C.make_backend("cupy", A.repo, A.lib)
        self.cp = self.be.cp
        self.A = A

    def cohort(self, A):
        import nibabel as nib
        coh, _ = self.C.cohort(A.brats_root, A.metadata, A.n, A.seed)
        out = []
        for sid, path in coh:
            vol = self.be.preprocess(nib.load(path).get_fdata(dtype=np.float32))
            out.append((sid, path, vol))
        return out

    def prep(self, vol):
        return self.be.prep(vol)

    def frozen(self, cp, lo, hi, l):  # exactly what front_propagation_gpu calls for boundary_seeds=None
        return self.be._tie(cp, lo, hi, l, None, "lower")

    def free(self): self.be.free()
    def sync(self): self.be.sync()
    def memstats(self): return self.be.memstats()
    def sampler(self): return self.C.Sampler(self.be)


# ---------------------------------------------------------------------------------------------------------------
def compare(ref, got):
    ro, rm, ra = ref[:3]
    go, gm, ga = got[:3]
    bad = []
    if sha(ro) != sha(go):
        bad.append("output bytes differ")
    for k in META_KEYS:
        if rm[k] != gm[k]:
            bad.append(f"meta[{k}] differs")
    for k in ra:
        if not np.array_equal(ra[k], ga[k]):
            bad.append(f"aux[{k}] differs")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["cupy", "shim"], default="cupy")
    ap.add_argument("--runner-dir", type=Path, help="DDFP_R7_GPU_SERVER_RUNNER_R0/r7 (contains r7_common.py)")
    ap.add_argument("--repo", type=Path, help="sealed production checkout (DD-FP-R7)")
    ap.add_argument("--lib", type=Path, help="DDFP_R7_GPU_SERVER_RUNNER_R0/lib")
    ap.add_argument("--brats-root", type=Path); ap.add_argument("--metadata", type=Path)
    ap.add_argument("--n", type=int, default=1, help="1 first, then 20 only after the 1-subject gate passed")
    ap.add_argument("--seed", type=int, default=42, help="cohort seed (as E1/E6)")
    ap.add_argument("--order-seed", type=int, default=20261022)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cap", type=int, default=AF.DEFAULT_CAP)
    ap.add_argument("--small-cap", type=int, default=20_000, help="forced-fallback correctness pass (G3 used 200000: no fallback on 14/20 subjects)")
    ap.add_argument("--no-profile", dest="profile", action="store_false")
    ap.add_argument("--no-mem", dest="mem", action="store_false")
    ap.add_argument("--synthetic", type=int, default=0); ap.add_argument("--synthetic-size", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    A = ap.parse_args()

    env = ShimEnv() if A.backend == "shim" else CupyEnv(A)
    cp = env.cp
    F = AF.load_frozen()
    A.out.mkdir(parents=True, exist_ok=True)
    rows_path = A.out / "rows.jsonl"
    if rows_path.exists() and rows_path.stat().st_size > 0:
        raise SystemExit(f"{rows_path} already exists and is not empty: use a fresh --out directory (the verdict requires exactly reps*2 rows per subject)")
    subjects = env.cohort(A)
    code = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in
            ("tests/run_gpu_gate.py", "tests/gate_verdict.py", "src/af_worklist.py", "frozen/gpu_native_worklist.py")}
    summary = dict(backend=env.name, cuda_executed=env.cuda, n_subjects=len(subjects), reps=A.reps, cap=A.cap, small_cap=A.small_cap,
                   frozen_sha256=AF.FROZEN_SHA256, code_sha256=code, verdict_code=GV.VERDICT_CODE_VERSION,
                   python=platform.python_version(), platform=platform.platform(), subjects={}, correctness_ok=True)
    if env.cuda:
        summary["device"] = env.be.device_info()
    ratios_by_subject = {}
    for si, (sid, path, vol) in enumerate(subjects):
        lo, hi, l = env.prep(vol)
        rec = dict(subject=sid)
        # ---- 1. correctness ------------------------------------------------------------------------------------
        env.free()
        ref = env.frozen(cp, lo, hi, l)
        got = AF.boutry_active_frontier(cp, lo, hi, l, "lower", frozen=F, frontier_cap=A.cap, trace=True)
        bad = compare(ref, got)
        rec["correct_default_cap"] = not bad
        rec["mismatch_default_cap"] = bad
        small = AF.boutry_active_frontier(cp, lo, hi, l, "lower", frozen=F, frontier_cap=A.small_cap)
        bad2 = compare(ref, small)
        rec["correct_small_cap"] = not bad2
        rec["mismatch_small_cap"] = bad2
        rec["small_cap_stats"] = small[1]["af"]["stats"]
        rec["af_stats"] = got[1]["af"]["stats"]
        fs = got[1]["af"]["frontier_sizes"]
        vd = got[1]["af"]["visit_depths"]
        rec["frontier_size_hist"] = dict(n=len(fs), min=int(min(fs)), median=float(st.median(fs)), p90=float(np.percentile(fs, 90)),
                                          p99=float(np.percentile(fs, 99)), max=int(max(fs)), sum=int(sum(fs)))
        rec["visit_depth_hist"] = dict(n=len(vd), median=float(st.median(vd)), p90=float(np.percentile(vd, 90)), max=int(max(vd)))
        rec["frozen_rounds"] = int(ref[1]["total_activation_rounds"])
        rec["frozen_counters"] = {k: ref[1][k] for k in META_KEYS if k not in ("closure_rounds_by_level", "level_transitions")}
        rec["output_sha256"] = sha(ref[0])
        if bad or bad2:
            summary["correctness_ok"] = False
        del ref, got, small
        # ---- 2. timing (cross-order) ----------------------------------------------------------------------------
        for rep in range(A.reps):
            order = [("frozen", "af")[i] for i in np.random.default_rng([A.order_seed, si, rep]).permutation(2)]
            for pos, m in enumerate(order):
                env.free(); env.sync()
                t0 = time.perf_counter()
                out = env.frozen(cp, lo, hi, l) if m == "frozen" else AF.boutry_active_frontier(cp, lo, hi, l, "lower", frozen=F, frontier_cap=A.cap)
                env.sync()
                dt = time.perf_counter() - t0
                row = dict(subject=sid, rep=rep, method=m, order_position=pos, t_total_s=dt, out_sha256=sha(out[0]),
                           rounds=out[1]["total_activation_rounds"], mem=env.memstats())
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                rec.setdefault("times", {}).setdefault(m, []).append(dt)
                if row["out_sha256"] != rec["output_sha256"] or row["rounds"] != rec["frozen_rounds"]:
                    summary["correctness_ok"] = False
                    rec.setdefault("timed_run_mismatch", []).append(dict(rep=rep, method=m))
                del out
        tf, ta = st.median(rec["times"]["frozen"]), st.median(rec["times"]["af"])
        rec["median_s"] = dict(frozen=tf, af=ta)
        rec["speedup"] = tf / ta
        ratios_by_subject[sid] = rec["speedup"]
        # ---- 3. breakdown (intrusive) ---------------------------------------------------------------------------
        if A.profile:
            env.free(); env.sync()
            pc = ProfiledCp(cp) if env.cuda else cp
            t0 = time.perf_counter()
            (env.frozen(pc, lo, hi, l))
            env.sync()
            t_f = time.perf_counter() - t0
            bf = {}
            if env.cuda:
                pc.resolve(); bf = dict(total_s=t_f, host=pc.host, counts=pc.n, gpu_ms=pc.gpu_ms)
            env.free(); env.sync()
            t0 = time.perf_counter()
            o = AF.boutry_active_frontier(cp, lo, hi, l, "lower", frozen=F, frontier_cap=A.cap, profile=True, trace=True)
            env.sync()
            t_a = time.perf_counter() - t0
            rec["breakdown"] = dict(frozen=bf, af=dict(total_s=t_a, **{k: v for k, v in o[1]["af"].items() if k in ("prof", "stats", "t_core_s")}),
                                    note="intrusive pass; event-based GPU times + extra syncs; not the headline numbers")
            del o
        # ---- 4. memory -------------------------------------------------------------------------------------------
        if A.mem and env.cuda:
            mem = {}
            for m in ("frozen", "af"):
                env.free(); env.sync()
                with env.sampler() as s:
                    out = env.frozen(cp, lo, hi, l) if m == "frozen" else AF.boutry_active_frontier(cp, lo, hi, l, "lower", frozen=F, frontier_cap=A.cap)
                    env.sync()
                mem[m] = dict(sampler=s.result(), pool=env.memstats())
                del out
            mem["extra_device_mib_af_minus_frozen"] = mem["af"]["sampler"]["peak_vram_delta_mib"] - mem["frozen"]["sampler"]["peak_vram_delta_mib"]
            rec["memory"] = mem
        summary["subjects"][sid] = rec
        (A.out / "summary.partial.json").write_text(json.dumps(summary, indent=1, default=str))
        print(f"[{si+1}/{len(subjects)}] {sid}: correct={rec['correct_default_cap'] and rec['correct_small_cap']} "
              f"speedup={rec['speedup']:.2f} (frozen {tf:.2f}s, af {ta:.2f}s)", flush=True)

    rows = [json.loads(x) for x in rows_path.read_text().splitlines() if x.strip()]
    V = GV.decide(summary, rows)
    if not env.cuda:
        V["performance"] = "NOT-MEASURED (shim run)" if V["correctness"] == "PASS" else V["performance"]
    summary["verdict"] = V
    summary["gather_share"] = GV.gather_share_stats(summary)
    (A.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in V.items() if k != "rule"}, indent=1))
    print("gather share:", json.dumps({k: v for k, v in summary["gather_share"].items() if k != "per_subject"}))
    return 0 if V["correctness"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
