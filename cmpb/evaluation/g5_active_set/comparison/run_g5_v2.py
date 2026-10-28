#!/usr/bin/env python3
"""
run_g5_v2.py -- G5 (corrected): frozen K=1 vs AF vs FP-aware generic pile worklist on REAL CUDA.  NOT EXECUTED ON A GPU by the author.

Replaces comparison/run_generic_gpu.py (v1, kept untouched).  Decision rules live in comparison/g5_rules.py and were fixed before any GPU number existed.

Design (answers to the G5 correction list)
 1. Statuses.  Every run gets exactly one status: OK | CORRECTNESS_FAIL | RESOURCE_FAIL | IMPL_ERROR | CUDA_FAULT, per method.  There is no "AF-NECESSARY".
 2. Warm-up.  Context init + NVRTC compile + module load + first launch are paid in a separate warm-up on a 24^3 crop (cold vs warm call time are recorded per
    method as `init_cost_s`) and never enter a timed row.  Every full-size subject additionally starts with an untimed check run of each method.
 3. Timing scope.  All three methods are called through the same kind of entry point with the same inputs (lo, hi, l_inf, "lower", block 256) and return
    host arrays (out + 5 aux) -> identical H2D/D2H and identical input validation (_validate_host_intervals is called inside each of them).  The timer
    runs from just before the call to just after env.sync(); validation against the frozen reference happens AFTER the timer.
 4. Memory.  Per method: peak VRAM delta (own cudaMemGetInfo poller, plus the project sampler when present) and CuPy pool stats; for the generic pile also
    the analytic allocation (bytes per array) and the real occupancy (max pile, max frontier, max next-list) relative to the caps.
 5. Order/reps.  reps is a multiple of 3 (default 6).  Per subject the execution order is counterbalanced: 3 reps = the 3 rotations of a random permutation,
    6 reps = all 6 permutations.  Every (subject, rep, method) must have exactly one timed row; every executed run is compared with the frozen reference
    (output bytes, all aux arrays, all counters, level_transitions, closure_rounds_by_level).
 6. Statistics.  Per-subject median ratio r = t_AF / t_PILE, practical-equivalence margin 1.10, 95 % bootstrap CI of the median over subjects, separate
    MIXED-SUBJECT-DEPENDENT verdict, noise and order-effect reports.
 7. Decomposition pass (one intrusive pass per method and subject): GPU time per kernel name (events), launches, launched threads, host syncs, D2H copies.

Stages: --stage 1 = first G3 subject only; writes STAGE1_PASS.json only if correctness, resources and memory probes all pass.
        --stage 2 = the 20 G3 subjects; refuses to start without a matching STAGE1_PASS.json (same code hashes, same GPU).
Usage (server; run tests/collect_env.py into the run directory first):
  python3 -I comparison/run_g5_v2.py --backend cupy --stage 1 --runner-dir ... --repo ... --lib ... --brats-root ... --metadata ... \
      --g3-summary gate_4090_n20_*/summary.json --out OUT_G5_STAGE1
  python3 -I comparison/run_g5_v2.py --backend cupy --stage 2 ... --stage1-dir OUT_G5_STAGE1 --out OUT_G5_STAGE2
Plumbing check (CPU, no evidence):  python3 -I comparison/run_g5_v2.py --backend shim --stage 1 --synthetic 2 --synthetic-size 8 --reps 3 --out OUT
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics as st
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests")); sys.path.insert(0, str(ROOT / "comparison"))
import af_worklist as AF  # noqa: E402
import generic_worklists_v2 as GW  # noqa: E402
import g5_rules as R  # noqa: E402
import run_gpu_gate as RG  # noqa: E402  (ShimEnv / CupyEnv / META_KEYS reused unchanged)

BLOCK = 256
CODE_FILES = ("comparison/run_g5_v2.py", "comparison/g5_rules.py", "comparison/generic_worklists_v2.py", "src/af_worklist.py",
              "frozen/gpu_native_worklist.py", "tests/run_gpu_gate.py")


def sha_bytes(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def compare(ref, got) -> list[str]:
    """component-level differences between two (out, meta, aux) results; empty list == identical in everything the frozen run reports."""
    ro, rm, ra = ref[:3]
    go, gm, ga = got[:3]
    bad = []
    if ro.shape != go.shape or sha_bytes(ro) != sha_bytes(go):
        bad.append("output bytes differ")
    for k in RG.META_KEYS:
        if k not in gm or rm[k] != gm[k]:
            bad.append(f"meta[{k}] differs")
    if set(ra) != set(ga):
        bad.append(f"aux key sets differ: {sorted(set(ra) ^ set(ga))}")
    for k in ra:
        if k in ga and not np.array_equal(ra[k], ga[k]):
            bad.append(f"aux[{k}] differs")
    return bad


# ----------------------------------------------------------------------------------------------------------------------------------------------
class ProfCp:
    """Intrusive proxy: per kernel name GPU time (events), launches, launched threads; host syncs and D2H copies.  Only used in the decomposition pass."""

    def __init__(self, cp):
        self._cp = cp
        self.host = {"sync_s": 0.0, "d2h_scalar_s": 0.0, "d2h_big_s": 0.0}
        self.n = {"sync": 0, "d2h_scalar": 0, "d2h_big": 0}
        self.ev = []
        self.gpu_ms: dict[str, float] = {}
        self.launches: dict[str, int] = {}
        self.threads: dict[str, int] = {}
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
                    e0.record(); k(grid, block, args); e1.record()
                    proxy.ev.append((name, e0, e1))
                    proxy.launches[name] = proxy.launches.get(name, 0) + 1
                    g = int(np.prod(grid)); b = int(np.prod(block))
                    proxy.threads[name] = proxy.threads.get(name, 0) + g * b
                return call
        return _M()

    def resolve(self):
        for name, e0, e1 in self.ev:
            e1.synchronize()
            self.gpu_ms[name] = self.gpu_ms.get(name, 0.0) + float(self._cp.cuda.get_elapsed_time(e0, e1))
        self.ev.clear()

    def report(self):
        self.resolve()
        return dict(gpu_ms=self.gpu_ms, gpu_ms_total=sum(self.gpu_ms.values()), launches=self.launches, launches_total=sum(self.launches.values()),
                    threads=self.threads, threads_total=sum(self.threads.values()), host=self.host, counts=self.n)


class VramPoll:
    """Peak device-memory use of the whole device during a block (cudaMemGetInfo polling).  mem_info() -> (free_bytes, total_bytes)."""

    def __init__(self, mem_info, interval=0.002):
        self.mem_info, self.interval = mem_info, interval
        self.min_free = None; self.free0 = None; self.n = 0; self._stop = threading.Event(); self._t = None

    def _sample(self):
        f = int(self.mem_info()[0]); self.n += 1
        self.min_free = f if self.min_free is None else min(self.min_free, f)

    def _loop(self):
        while not self._stop.is_set():
            self._sample(); time.sleep(self.interval)

    def __enter__(self):
        self.free0 = int(self.mem_info()[0]); self.min_free = self.free0
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join(); self._sample()

    def result(self):
        return dict(peak_vram_delta_mib=(self.free0 - self.min_free) / 2**20, baseline_free_mib=self.free0 / 2**20, samples=self.n, interval_s=self.interval)


# ----------------------------------------------------------------------------------------------------------------------------------------------
def order_for(si: int, rep: int, seed: int) -> list[str]:
    """counterbalanced execution order: rotations of a random permutation; odd blocks use the reversed permutation (6 reps = all 6 permutations)."""
    base = [R.METHODS[i] for i in np.random.default_rng([seed, si]).permutation(3)]
    block, k = divmod(rep, 3)
    seq = base if block % 2 == 0 else base[::-1]
    return seq[k:] + seq[:k]


def code_hashes():
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CODE_FILES}


def crop(vol, side=24):
    sl = tuple(slice(max(0, int(s) // 2 - side // 2), max(0, int(s) // 2 - side // 2) + side) for s in vol.shape)
    return vol[sl]


class Runner:
    def __init__(self, env, A, F):
        self.env, self.A, self.F = env, A, F
        self.inject = [tuple(x.split(":")) for x in (A.inject or [])]
        self.cp = env.cp
        self.af_native_profile = False

    def call(self, m, cp, lo, hi, l, trace=False):
        if m == "frozen":
            return self.F.boutry_bucket_gpu(cp, lo, hi, l, "lower", BLOCK)
        if m == "af":
            return AF.boutry_active_frontier(cp, lo, hi, l, "lower", BLOCK, self.A.cap, frozen=self.F, trace=trace, profile=self.af_native_profile)
        return GW.generic_pile_worklist(cp, lo, hi, l, "nearest-lower", BLOCK, list_cap=self.A.cap, pile_cap=None, frozen=self.F, trace=trace)

    def _inject(self, m, phase, rep, stage_when):
        """test-only fault injection (shim)."""
        for spec in self.inject:
            im, kind = spec[0], spec[1]
            irep = int(spec[2]) if len(spec) > 2 else None
            if im == m and (irep is None or (phase == "timed" and irep == rep)) and stage_when == ("post" if kind == "corrupt" else "pre"):
                yield kind

    def run(self, m, phase, subj, rep, pos, lo, hi, l, ref=None, cp=None, trace=False, timed=True, extra=None):
        """one run -> (row, result|None).  Timer: perf_counter just before the call .. just after env.sync(); validation after the timer."""
        env = self.env
        cpx = cp if cp is not None else self.cp
        env.free(); env.sync()
        row = dict(subject=subj, rep=rep, method=m, order_position=pos, phase=phase, status=R.OK, detail="", t_total_s=None)
        res = None
        t0 = time.perf_counter()
        try:
            for kind in self._inject(m, phase, rep, "pre"):
                if kind == "oom":
                    raise R.GenericResourceError("injected capacity failure")
                if kind == "impl":
                    raise RuntimeError("injected implementation error")
                if kind == "cudafault":
                    raise type("CUDARuntimeError", (Exception,), {})("injected illegal memory access")
            res = self.call(m, cpx, lo, hi, l, trace=trace)
            env.sync()
            row["t_total_s"] = time.perf_counter() - t0
        except BaseException as e:  # noqa: BLE001  (every failure is classified, never swallowed)
            row["status"] = R.classify_exception(e)
            row["detail"] = f"{type(e).__name__}: {str(e)[:300]}"
            row["t_total_s"] = None
            try:
                env.sync()
            except BaseException:  # noqa: BLE001
                pass
            return row, None
        for kind in self._inject(m, phase, rep, "post"):
            if kind == "corrupt":
                res[0].flat[0] = np.float32(-12345.0)
        row["out_sha256"] = sha_bytes(res[0]); row["rounds"] = int(res[1]["total_activation_rounds"])
        if ref is not None:
            bad = compare(ref, res)
            if bad:
                row["status"] = R.CORRECTNESS_FAIL; row["detail"] = "; ".join(bad[:8])
        meta = res[1]
        if m == "pile" and "gw" in meta:
            g = meta["gw"]
            row["info"] = dict(stats=g["stats"], alloc_mib=g["alloc"]["total_mib"], occupancy=g["occupancy"])
        elif m == "af" and "af" in meta:
            row["info"] = dict(stats=meta["af"].get("stats"))
        if extra:
            row.update(extra)
        return row, res


class Sink:
    def __init__(self, path):
        self.path = path; self.rows = []

    def add(self, row):
        self.rows.append(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["cupy", "shim"], default="cupy")
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True)
    ap.add_argument("--stage1-dir", type=Path)
    ap.add_argument("--runner-dir", type=Path); ap.add_argument("--repo", type=Path); ap.add_argument("--lib", type=Path)
    ap.add_argument("--brats-root", type=Path); ap.add_argument("--metadata", type=Path)
    ap.add_argument("--g3-summary", type=Path, help="G3 summary.json: defines the 20 subjects, their order and the reference speed-ups")
    ap.add_argument("--seed", type=int, default=42, help="cohort seed (as G3/E1/E6)")
    ap.add_argument("--order-seed", type=int, default=20261023)
    ap.add_argument("--reps", type=int, default=6, help="multiple of 3 (>=3); 6 = all execution orders")
    ap.add_argument("--cap", type=int, default=AF.DEFAULT_CAP, help="list capacity of AF and of the generic pile (same value)")
    ap.add_argument("--no-memory", dest="memory", action="store_false"); ap.add_argument("--no-profile", dest="profile", action="store_false")
    ap.add_argument("--synthetic", type=int, default=0); ap.add_argument("--synthetic-size", type=int, default=6)
    ap.add_argument("--inject", action="append", help="TEST ONLY (shim): METHOD:KIND[:REP], KIND in corrupt|oom|impl|cudafault")
    ap.add_argument("--skip-stage1-check", action="store_true", help="TEST ONLY (shim)")
    ap.add_argument("--out", type=Path, required=True)
    A = ap.parse_args()
    if A.reps < R.MIN_REPS or A.reps % 3:
        raise SystemExit("--reps must be a multiple of 3 and >= 3")
    if A.backend == "cupy" and (A.inject or A.skip_stage1_check):
        raise SystemExit("--inject/--skip-stage1-check are test-only and refused on the cupy backend")
    if A.backend == "cupy" and not A.g3_summary:
        raise SystemExit("--g3-summary is required on the cupy backend (fixes the cohort)")

    env = RG.ShimEnv() if A.backend == "shim" else RG.CupyEnv(A)
    cp = env.cp
    F = AF.load_frozen()
    code = code_hashes()
    A.out.mkdir(parents=True, exist_ok=True)
    rows_path = A.out / "rows.jsonl"
    if rows_path.exists() and rows_path.stat().st_size > 0:
        raise SystemExit(f"{rows_path} exists and is not empty: use a fresh --out")
    sink = Sink(rows_path)

    # ---- cohort -----------------------------------------------------------------------------------------------------------------------------
    g3 = json.loads(A.g3_summary.read_text()) if A.g3_summary else None
    g3_ids = list(g3["subjects"]) if g3 else None
    if A.backend == "shim":
        pool = env.cohort(A)                                     # [(sid, None, vol)]
        loaders = {sid: (lambda v=vol: v) for sid, _p, vol in pool}
        ids = [sid for sid, _p, _v in pool]
        expected = g3_ids
    else:
        import nibabel as nib
        coh, _ = env.C.cohort(A.brats_root, A.metadata, 20, A.seed)
        paths = {sid: path for sid, path in coh}
        if [s for s in g3_ids if s not in paths]:
            raise SystemExit("cohort drift: G3 subjects missing from the cohort resolved with this seed/metadata -> stop")
        loaders = {sid: (lambda p=paths[sid]: env.be.preprocess(nib.load(p).get_fdata(dtype=np.float32))) for sid in g3_ids}
        ids = g3_ids
        expected = g3_ids
    run_ids = ids[:1] if A.stage == 1 else ids
    if A.stage == 2 and not A.skip_stage1_check:
        gate_file = (A.stage1_dir or Path("/nonexistent")) / "STAGE1_PASS.json"
        if not gate_file.is_file():
            raise SystemExit("stage 2 requires --stage1-dir containing STAGE1_PASS.json (stage 1 must pass first)")
        gp = json.loads(gate_file.read_text())
        dev_now = env.be.device_info() if env.cuda else {}
        if gp.get("code_sha256") != code or gp.get("device", {}).get("gpu_name") != dev_now.get("gpu_name"):
            raise SystemExit("STAGE1_PASS.json was produced with different code or a different GPU: rerun stage 1")
    summary = dict(rules_version=R.RULES_VERSION, stage=A.stage, backend=env.name, cuda_executed=env.cuda, reps=A.reps, cap=A.cap, block=BLOCK,
                   order_seed=A.order_seed, frozen_sha256=AF.FROZEN_SHA256, code_sha256=code, rules=dict(margin=R.MARGIN, boot_b=R.BOOT_B, boot_seed=R.BOOT_SEED,
                   noise_range=R.NOISE_RANGE, noise_cell_fraction=R.NOISE_CELL_FRACTION), python=platform.python_version(), platform=platform.platform(),
                   argv=sys.argv, subjects={}, memory={}, profile={}, aborted=None)
    summary["device"] = env.be.device_info() if env.cuda else {}
    runner = Runner(env, A, F)

    def finish(abort=None):
        summary["aborted"] = abort
        meta = dict(stage=A.stage, cuda_executed=env.cuda, reps=A.reps, subjects=run_ids, expected_subjects=expected)
        V = R.decide(sink.rows, meta)
        summary["decision"] = V
        if A.stage == 1:
            gate = R.stage1_gate(sink.rows, meta, summary)
            summary["stage1_gate"] = gate
            if gate["passed"] and not abort:
                (A.out / "STAGE1_PASS.json").write_text(json.dumps(dict(code_sha256=code, device=summary["device"], subject=run_ids[0], rules_version=R.RULES_VERSION), indent=1))
        if g3 and summary["subjects"]:
            sp = [rec["speedup_af_vs_frozen"] for rec in summary["subjects"].values() if "speedup_af_vs_frozen" in rec]
            g3sp = [g3["subjects"][s]["speedup"] for s in summary["subjects"] if s in g3["subjects"] and "speedup" in g3["subjects"][s]]
            if sp and g3sp:
                a, b = float(np.median(sp)), float(np.median(g3sp))
                summary["g3_consistency"] = dict(median_speedup_af_vs_frozen_now=a, median_speedup_g3=b, relative_difference=a / b - 1,
                                                 warning=bool(abs(a / b - 1) > R.G3_SPEEDUP_DRIFT))
        (A.out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
        write_table(A.out / "subject_table.csv", summary)
        print(json.dumps({k: v for k, v in V.items() if k not in ("speed", "status_counts", "rules_version")}, indent=1, default=str))
        if A.stage == 1:
            print("stage1_gate:", json.dumps(summary["stage1_gate"]))
        bad = abort or V["verdict"].startswith(("INVALID", "ABORTED", "PILE-")) or (A.stage == 1 and not summary["stage1_gate"]["passed"]) or not env.cuda
        return 0 if (env.cuda and not bad) else (4 if abort else 3)

    def fatal(row, why):
        sink.add(row)
        return finish(abort=why)

    # ---- warm-up (never timed rows) ---------------------------------------------------------------------------------------------------------
    first_vol = loaders[run_ids[0]]()
    tiny = crop(first_vol)
    warm = dict(ok=True, per_method={})
    if env.cuda:
        (cp.zeros(1) + 1).sum(); env.sync()                      # context + cuBLAS-free trivial kernel before any cold timing
    lo_t, hi_t, l_t = env.prep(tiny)
    ref_t = None
    for m in R.METHODS:
        cold_row, res_c = runner.run(m, "warmup", "warmup", 0, 0, lo_t, hi_t, l_t, ref=ref_t)
        warm_row, res_w = runner.run(m, "warmup", "warmup", 1, 0, lo_t, hi_t, l_t, ref=ref_t)
        for r_ in (cold_row, warm_row):
            sink.add(r_)
        if m == "frozen" and res_w is not None:
            ref_t = res_w
        ok = cold_row["status"] == R.OK and warm_row["status"] == R.OK
        warm["per_method"][m] = dict(cold_s=cold_row["t_total_s"], warm_s=warm_row["t_total_s"], init_cost_s=(cold_row["t_total_s"] - warm_row["t_total_s"]) if ok else None,
                                     status=[cold_row["status"], warm_row["status"]])
        warm["ok"] = warm["ok"] and ok
        if "CUDA_FAULT" in (cold_row["status"], warm_row["status"]) or (m == "frozen" and not ok):
            summary["warmup"] = warm
            return finish(abort=f"warm-up failure: {m}")
    # warm-up outputs are compared against the frozen result on the same tiny input (done inside run via ref) for af/pile
    summary["warmup"] = warm
    if not warm["ok"]:
        return finish(abort="warm-up failure (see rows)")
    del lo_t, hi_t

    # ---- subjects ---------------------------------------------------------------------------------------------------------------------------
    for si, sid in enumerate(run_ids):
        vol = first_vol if si == 0 else loaders[sid]()
        lo, hi, l = env.prep(vol)
        rec = dict(subject=sid)
        # (a) untimed check: frozen reference, then AF and PILE vs reference
        row, ref = runner.run("frozen", "check", sid, -1, 0, lo, hi, l)
        sink.add(row)
        if ref is None:
            return finish(abort=f"frozen reference failed on {sid}: {row['status']} {row['detail']}")
        rec["output_sha256"] = sha_bytes(ref[0]); rec["frozen_rounds"] = int(ref[1]["total_activation_rounds"])
        for m in ("af", "pile"):
            row, got = runner.run(m, "check", sid, -1, 0, lo, hi, l, ref=ref, trace=True)
            sink.add(row)
            rec[f"check_{m}"] = dict(status=row["status"], detail=row["detail"])
            if row["status"] == R.CUDA_FAULT:
                return finish(abort=f"CUDA fault in {m} on {sid}")
            if m == "af" and row["status"] != R.OK:
                return finish(abort=f"AF disagrees with frozen on {sid}: {row['detail']}")
            if m == "pile" and got is not None:
                ps = np.array(got[1]["gw"]["pile_sizes"]); fs = np.array(got[1]["gw"]["frontier_sizes"])
                rec["pile_trace"] = dict(N=got[1]["gw"]["N"], visits=got[1]["gw"]["stats"]["visits"], rounds=got[1]["gw"]["stats"]["rounds"],
                                         max_pile=int(ps.max()), median_pile_at_visit=float(np.median(ps)), p90_pile=float(np.percentile(ps, 90)),
                                         split_elements=got[1]["gw"]["stats"]["split_elements"], max_frontier=int(fs.max()),
                                         alloc=got[1]["gw"]["alloc"], occupancy=got[1]["gw"]["occupancy"])
            if m == "af" and got is not None and "af" in got[1]:
                rec["af_stats"] = got[1]["af"].get("stats")
            del got
        # (b) timed runs
        for rep in range(A.reps):
            for pos, m in enumerate(order_for(si, rep, A.order_seed)):
                row, res = runner.run(m, "timed", sid, rep, pos, lo, hi, l, ref=ref)
                sink.add(row)
                if row["status"] == R.CUDA_FAULT:
                    return finish(abort=f"CUDA fault in {m} on {sid} rep {rep}")
                if m in ("frozen", "af") and row["status"] != R.OK:
                    return finish(abort=f"{m} failed on {sid} rep {rep}: {row['status']} {row['detail']}")
                del res
        med = R.per_subject_medians(sink.rows, [sid])[sid]
        rec["median_s"] = med
        if "frozen" in med and "af" in med:
            rec["speedup_af_vs_frozen"] = med["frozen"] / med["af"]
        if "frozen" in med and "pile" in med:
            rec["speedup_pile_vs_frozen"] = med["frozen"] / med["pile"]
        if "af" in med and "pile" in med:
            rec["ratio_af_over_pile"] = med["af"] / med["pile"]
        rec["rep_times_s"] = {m: [r_["t_total_s"] for r_ in sink.rows if r_["subject"] == sid and r_["method"] == m and r_["phase"] == "timed" and r_["status"] == R.OK]
                              for m in R.METHODS}
        # (c) memory
        if A.memory:
            mem = {}
            for m in R.METHODS:
                entry = {}
                poll = None
                try:
                    mi = cp.cuda.runtime.memGetInfo if env.cuda else None
                    poll = VramPoll(mi) if mi else None
                except Exception as e:  # noqa: BLE001
                    entry["vram_poll_error"] = f"{type(e).__name__}: {e}"
                proj = env.sampler() if env.cuda else None
                env.free(); env.sync()
                ctx = [x for x in (poll, proj) if x is not None]
                for c_ in ctx:
                    c_.__enter__()
                try:
                    row, res = runner.run(m, "memory", sid, -1, 0, lo, hi, l, ref=ref)
                    env.sync()
                finally:
                    for c_ in reversed(ctx):
                        c_.__exit__(None, None, None)
                sink.add(row)
                if poll is not None:
                    entry["vram_poll"] = poll.result()
                if proj is not None:
                    try:
                        entry["project_sampler"] = proj.result()
                    except Exception as e:  # noqa: BLE001
                        entry["project_sampler_error"] = f"{type(e).__name__}: {e}"
                entry["pool"] = env.memstats(); entry["status"] = row["status"]
                if row["status"] == R.CUDA_FAULT:
                    return finish(abort=f"CUDA fault in memory pass ({m}) on {sid}")
                mem[m] = entry
                del res
            summary["memory"] = mem if A.stage == 1 else summary.get("memory", {})
            rec["memory"] = mem
        # (d) decomposition (intrusive; not headline numbers)
        if A.profile:
            prof = {}
            for m in R.METHODS:
                pc = ProfCp(cp)
                try:
                    row, res = runner.run(m, "profile", sid, -1, 0, lo, hi, l, ref=ref, cp=pc)
                    env.sync()
                    sink.add(row)
                    prof[m] = dict(status=row["status"], total_s=row["t_total_s"], **(pc.report() if row["status"] == R.OK else {}))
                except Exception as e:  # noqa: BLE001  (instrumentation failure never invalidates the comparison)
                    prof[m] = dict(status="PROFILE_UNAVAILABLE", detail=f"{type(e).__name__}: {e}")
            # AF's own event profile (as in G3: gather / round / finalize GPU ms); instrumentation only, never part of the verdict
            try:
                runner.af_native_profile = True
                row, res = runner.run("af", "profile", sid, -1, 0, lo, hi, l, ref=ref, trace=True)
                env.sync()
                sink.add(row)
                prof["af_native"] = dict(status=row["status"], total_s=row["t_total_s"],
                                         **({k: v for k, v in res[1]["af"].items() if k in ("prof", "stats", "t_core_s")} if res is not None else {}))
            except Exception as e:  # noqa: BLE001
                prof["af_native"] = dict(status="PROFILE_UNAVAILABLE", detail=f"{type(e).__name__}: {e}")
            finally:
                runner.af_native_profile = False
            rec["profile"] = prof
        del ref
        summary["subjects"][sid] = rec
        (A.out / "summary.partial.json").write_text(json.dumps(summary, indent=1, default=str))
        sp = rec.get("speedup_af_vs_frozen"); sq = rec.get("speedup_pile_vs_frozen"); rr = rec.get("ratio_af_over_pile")
        print(f"[{si+1}/{len(run_ids)}] {sid}: AF/frozen={sp and round(sp,2)} PILE/frozen={sq and round(sq,2)} t_AF/t_PILE={rr and round(rr,3)} "
              f"check af={rec['check_af']['status']} pile={rec['check_pile']['status']}", flush=True)
    return finish()


def write_table(path, summary):
    cols = ["subject", "t_frozen_s", "t_af_s", "t_pile_s", "speedup_af", "speedup_pile", "r_af_over_pile", "pile_check", "pile_peak_vram_mib", "af_peak_vram_mib",
            "frozen_peak_vram_mib", "pile_alloc_mib", "pile_max_pile", "pile_max_frontier", "pile_visits"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for sid, r in summary["subjects"].items():
            med = r.get("median_s", {}); mem = r.get("memory", {}); pt = r.get("pile_trace", {})

            def pv(m):
                e = mem.get(m, {})
                return e.get("vram_poll", {}).get("peak_vram_delta_mib")
            w.writerow([sid, med.get("frozen"), med.get("af"), med.get("pile"), r.get("speedup_af_vs_frozen"), r.get("speedup_pile_vs_frozen"),
                        r.get("ratio_af_over_pile"), r.get("check_pile", {}).get("status"), pv("pile"), pv("af"), pv("frozen"),
                        pt.get("alloc", {}).get("total_mib"), pt.get("max_pile"), pt.get("max_frontier"), pt.get("visits")])


if __name__ == "__main__":
    raise SystemExit(main())
