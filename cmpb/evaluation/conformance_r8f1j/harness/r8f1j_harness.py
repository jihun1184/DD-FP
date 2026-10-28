#!/usr/bin/env python3
"""R8-F1J Stage 2-pre harness: Round-0 GPU semantic conformance (Q1-Q10, protocol R8-F1I section 3).
Read-only w.r.t. production code (SHA-gated import). Backends:
  gpu          production CuPy modules from --repo (needs a CUDA GPU)
  cpu-selftest CPU stand-ins with optional fault injection: validates the *harness*, is NOT GPU evidence.
Tiers: selftest | T0 | T1 | T2.  Exit code 0=PASS tier, 2=counterexample/mismatch, 3=HOLD (error/tool).
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, platform, resource, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(PKG / "lib"))
import r8f1j_oracle as O                                   # noqa: E402
from cpu_fp import build_ispan_cpu                         # noqa: E402
from seedref import seed_aware_cpu_reference               # noqa: E402
from ibi_cpu import ibi as ibi_cpu                         # noqa: E402

SEED = 20271701
FROZEN = json.load(open(HERE / "FROZEN_SOURCES.json"))
Q3_FIELDS = ["level_transitions", "schedule_direction_counts", "equidistant_next_level_tie_events",
             "total_activation_rounds", "processed_count", "discovered_count", "nan_count",
             "backward_scheduler_transition_count"]
HARD = ["q1_ispan", "q2_tile", "q4_sched", "q5_B", "det"]                  # stop tier + COUNTEREXAMPLE
SOFT_VIOL = ["q6_seam", "q7_Bdwc", "q8_tiledwc", "q9_preserve", "q7_kernel_vs_oracle"]  # stop tier (theory/implementation)
INFO = ["q3_meta"]                                                         # record only -> CONDITIONAL


def sha_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def sha_arr(a) -> str: return sha_bytes(np.ascontiguousarray(a).tobytes())


# ------------------------------------------------------------------ backends
class CpuSelf:
    """CPU stand-ins with the GPU backend's interface. `fault` injects a defect to test comparator power."""
    name = "cpu-selftest"
    def __init__(self, fault: str | None = None): self.fault, self.calls = fault, 0
    def ispan(self, vol):
        lo, hi, l = build_ispan_cpu(vol); hi = hi.copy()
        if self.fault == "ispan": hi.flat[int(np.flatnonzero(hi < 255)[-1])] += 1   # valid-but-wrong interval (passes input validation)
        return lo, hi, float(l)
    def tile(self, lo, hi, l):
        self.calls += 1
        out, meta, aux = seed_aware_cpu_reference(lo, hi, l, None, "lower"); out = out.copy()
        meta = dict(meta); aux = {"scheduled_level": aux["scheduled_level"].copy()}
        if self.fault == "tile": out.flat[out.size // 2] += 1
        if self.fault == "raise" and self.calls % 7 == 0: raise RuntimeError("injected CUDA-like error")
        if self.fault == "meta": meta["level_transitions"] = meta["level_transitions"][:-1] + [[0, 0]]
        if self.fault == "sched": aux["scheduled_level"].flat[1] += 1
        if self.fault == "nondet" and self.calls % 2 == 0: out.flat[out.size // 2] += 1
        return out, meta, aux
    def B(self, vol, K):
        B = ibi_cpu(vol, K, 1, max_rounds=0)["hist"][0].copy()
        if self.fault == "B": B.flat[B.size // 2] += 1
        if self.fault == "seam" and K > 1: B[:, :, 2 * O.split_domain_1d(vol.shape[2], K)[0][1] - 1] += 5
        return B
    def dwc(self, u):
        r = O.dwc_count_np(u)
        if self.fault == "dwc_kernel": r["violations"] += 1
        return {"violations": r["violations"], "nan_skipped": r["nan_skipped"], "total_blocks": r["total_blocks"]}
    def env(self): return {"backend": self.name, "fault": self.fault}


class Gpu:
    name = "gpu"
    def __init__(self, repo: Path, allow_unfrozen: bool = False):
        import cupy as cp
        self.cp = cp; self.repo = repo
        self.shas = check_sources(repo, allow_unfrozen)
        sys.path.insert(0, str(repo))
        from src.ddfp import gpu_immersion as gi, gpu_seed_aware_worklist as gs
        spec = importlib.util.spec_from_file_location("ddfp_experiment", repo / "scripts/ddfp/experiment_DDFP_all.py")
        ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
        self.gi, self.gs, self.ex = gi, gs, ex
    def ispan(self, vol):
        lo, hi, l = self.gi.build_ispan_gpu(vol)
        return self.cp.asnumpy(lo), self.cp.asnumpy(hi), float(l)
    def tile(self, lo, hi, l):
        # direct call keeps meta/aux (front_propagation_gpu discards them); same path as production unseeded call
        out, meta, aux = self.gs.seed_aware_boutry_bucket_gpu(
            self.cp, self.cp.asarray(lo), self.cp.asarray(hi), l, boundary_seeds=None, tie_policy="lower")
        self.cp.cuda.Stream.null.synchronize()
        return np.asarray(self.cp.asnumpy(out), np.float32), meta, {"scheduled_level": np.asarray(self.cp.asnumpy(aux["scheduled_level"]))}
    def B(self, vol, K):
        r = self.ex.run_ibi_v10(vol, K, 1, max_rounds=0)       # Round 0 only
        self.cp.cuda.Stream.null.synchronize()
        if r["R_star"] != 0: raise RuntimeError(f"run_ibi_v10 ran IBI rounds (R*={r['R_star']})")
        return np.asarray(r["u_dd"], np.float32)
    def dwc(self, u):
        r = self.gi.verify_dwc_gpu(self.cp.asarray(np.asarray(u, np.float32)), verbose=False)
        return {k: int(r[k]) for k in ("violations", "nan_skipped", "total_blocks")}
    def env(self):
        cp = self.cp; d = cp.cuda.Device(); p = cp.cuda.runtime.getDeviceProperties(d.id)
        return {"backend": "gpu", "gpu_name": p["name"].decode(), "compute_capability": d.compute_capability,
                "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(), "driver": cp.cuda.runtime.driverGetVersion(),
                "cupy": cp.__version__, "repo": str(self.repo)}


def check_sources(repo: Path, allow_unfrozen=False) -> dict:
    paths = {"gpu_immersion.py": repo / "src/ddfp/gpu_immersion.py", "gpu_native_worklist.py": repo / "src/ddfp/gpu_native_worklist.py",
             "gpu_seed_aware_worklist.py": repo / "src/ddfp/gpu_seed_aware_worklist.py", "cpu_fp.py": repo / "src/ddfp/cpu_fp.py",
             "experiment_DDFP_all.py": repo / "scripts/ddfp/experiment_DDFP_all.py",
             "seedref.py": PKG / "lib/seedref.py", "ibi_cpu.py": PKG / "lib/ibi_cpu.py"}
    got = {k: (sha_bytes(p.read_bytes()) if p.exists() else "MISSING") for k, p in paths.items()}
    bad = {k: (got[k][:12], FROZEN[k][:12]) for k in got if got[k] != FROZEN[k]}
    if bad and not allow_unfrozen: raise RuntimeError(f"source SHA mismatch (fail-closed): {bad}")
    return {"sha": got, "mismatch": bad}


def nvidia_smi() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.used,memory.total,utilization.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e: return f"nvidia-smi unavailable: {e!r}"


# ------------------------------------------------------------------ one case
def cpu_reference(vol, K):
    """CPU side of one case; pure function of (vol,K) so it can be prefetched in a process pool."""
    D = vol.shape[2]; tiles = []
    for z0, z1, z0e, z1e in O.tile_windows(D, K):
        lo, hi, l = build_ispan_cpu(vol[:, :, z0e:z1e])
        out, meta, aux = seed_aware_cpu_reference(lo, hi, l, None, "lower")
        tiles.append(dict(lo=lo, hi=hi, l=l, out=out, meta=meta, sched=aux["scheduled_level"]))
    return dict(tiles=tiles, B=ibi_cpu(vol, K, 1, max_rounds=0)["hist"][0])


def check_case(be, vol, K, ref=None, tie_info=False, do_det=True):
    """Run Q1-Q11 for one (vol,K). Returns (verdict dict, repro arrays on failure)."""
    D = vol.shape[2]; ref = ref or cpu_reference(vol, K)
    v = {k: True for k in HARD + SOFT_VIOL + INFO}; notes = []; repro = {}
    interiors = []; q3_diff = []
    lo_g_all, hi_g_all = O.ispan_oracle(vol)
    for k, (z0, z1, z0e, z1e) in enumerate(O.tile_windows(D, K)):
        win = vol[:, :, z0e:z1e]; c = ref["tiles"][k]
        lo, hi, l = be.ispan(win); lo_o, hi_o = O.ispan_oracle(win)
        ok = O.bits_equal(lo, c["lo"]) and O.bits_equal(hi, c["hi"]) and O.bits_equal(lo, lo_o) and O.bits_equal(hi, hi_o) and l == c["l"] == 0.0
        v["q1_ispan"] &= ok
        out, meta, aux = be.tile(lo, hi, l)
        if do_det:
            out2, meta2, aux2 = be.tile(lo, hi, l)
            v["det"] &= O.bits_equal(out, out2) and np.array_equal(aux["scheduled_level"], aux2["scheduled_level"]) \
                and all(meta[f] == meta2[f] for f in Q3_FIELDS if f in meta)
        v["q2_tile"] &= O.bits_equal(out, c["out"])
        v["q4_sched"] &= np.array_equal(aux["scheduled_level"], c["sched"])
        for f in Q3_FIELDS:
            if meta.get(f) != c["meta"].get(f): v["q3_meta"] = False; q3_diff.append((k, f))
        # P1 evidence on the GPU output itself: clamp inside interval, output equals scheduled level
        v["q8_tiledwc"] &= bool(((out >= lo) & (out <= hi)).all()) and bool((out == aux["scheduled_level"]).all()) \
            and all(bool((f == 0).all()) for f in (out[0], out[-1], out[:, 0], out[:, -1], out[:, :, 0], out[:, :, -1]))
        for arr, tag in ((out, "padded"), (out[1:-1, 1:-1, 1:-1], "interior")):
            kg, ko = be.dwc(arr), O.dwc_count_np(arr)
            v["q7_kernel_vs_oracle"] &= (kg["violations"], kg["nan_skipped"], kg["total_blocks"]) == (ko["violations"], ko["nan_skipped"], ko["total_blocks"])
            v["q8_tiledwc"] &= kg["violations"] == 0 and kg["nan_skipped"] == 0 and ko["violations"] == 0
        interiors.append(out[1:-1, 1:-1, 1:-1])
        if tie_info and O.bits_equal(c["out"], seed_aware_cpu_reference(c["lo"], c["hi"], c["l"], None, "higher")[0]) is False:
            notes.append("tie_lower_ne_higher")
        if not all(v[q] for q in ("q1_ispan", "q2_tile", "q4_sched", "det")): repro[f"tile{k}_lo"] = lo; repro[f"tile{k}_hi"] = hi; repro[f"tile{k}_gpu"] = out; repro[f"tile{k}_cpu"] = c["out"]
    B = be.B(vol, K)
    if do_det: v["det"] &= O.bits_equal(B, be.B(vol, K))
    v["q5_B"] = O.bits_equal(B, ref["B"]) and O.bits_equal(B, O.assemble_round0(interiors, D, K))
    kg, ko = be.dwc(B), O.dwc_count_np(B)
    v["q7_kernel_vs_oracle"] &= (kg["violations"], kg["nan_skipped"], kg["total_blocks"]) == (ko["violations"], ko["nan_skipped"], ko["total_blocks"])
    v["q7_Bdwc"] = kg["violations"] == 0 and ko["violations"] == 0 and kg["nan_skipped"] == 0 and not np.isnan(B).any()
    v["q6_seam"] = O.seam_predicates(B, None, None, D, K)["seam_mono"] == 0
    inner = (slice(1, -1),) * 3
    v["q9_preserve"] = O.bits_equal(B[::2, ::2, ::2], vol.astype(np.float32)) and bool(((B >= lo_g_all[inner]) & (B <= hi_g_all[inner])).all())
    if not all(v[q] for q in HARD + SOFT_VIOL): repro.update(B_gpu=B, B_cpu=ref["B"], vol=vol)
    return dict(v, q3_diff=q3_diff[:6], notes=notes, dwc_B=kg, B_sha=sha_arr(B)), repro


# ------------------------------------------------------------------ inputs
def inputs_T0():
    for si, sh in enumerate(O.SHAPES):
        for ai, A in enumerate((2, 4, 256)):
            for i in range(20):
                rng = np.random.default_rng([SEED, 0, si, ai, i]); D = sh[2]
                yield f"T0_{sh[0]}x{sh[1]}x{sh[2]}_A{A}_{i:02d}", O.gen_volume("iid", sh, A, rng), 1 + i % min(6, D)


def inputs_T1(sample=None, stride=1):
    cases = [("a", m, K) for m in range(4096) for K in (1, 2, 3)] + [("b", m, K) for m in range(6561) for K in (1, 2)]
    if sample:
        idx = np.random.default_rng([SEED, 1]).permutation(len(cases))[:sample]; cases = [cases[i] for i in sorted(idx)]
    for g, m, K in cases[::stride]:
        if g == "a": vol = np.array([(m >> b) & 1 for b in range(12)], np.uint8).reshape(2, 2, 3)
        else: vol = np.array([(m // 3 ** b) % 3 for b in range(8)], np.uint8).reshape(2, 2, 2)
        yield f"T1{g}_{m:04d}_K{K}", vol, K


def t2_plan():
    combos = [(f, si, A, K) for f in O.FAMILIES for si, sh in enumerate(O.SHAPES) for A in (2, 3, 4, 16, 256) for K in range(1, min(6, sh[2]) + 1)]
    extra = np.random.default_rng([SEED, 2]).choice(len(combos), 1500 - len(combos), replace=False)
    return [(c, 0) for c in combos] + [(combos[i], 1) for i in sorted(extra)]


def inputs_T2():
    for n, ((f, si, A, K), inst) in enumerate(t2_plan()):
        sh = O.SHAPES[si]; rng = np.random.default_rng([SEED, 3, O.FAMILIES.index(f), si, A, K, inst])
        yield f"T2_{n:04d}_{f}_{sh[0]}x{sh[1]}x{sh[2]}_A{A}_K{K}_i{inst}", O.gen_volume(f, sh, A, rng), K


# ------------------------------------------------------------------ comparator / validator self-test
def selftest(be_factory, out: Path):
    """(i) comparator detects each injected fault, (ii) DWC validator detects non-DWC fields incl. NaN, (iii) clean run passes."""
    res = {"fault_detection": {}, "dwc_negative_controls": {}}
    vol = O.gen_volume("iid", (3, 3, 8), 4, np.random.default_rng([SEED, 9]))
    ref = cpu_reference(vol, 3)
    clean, _ = check_case(CpuSelf(), vol, 3, ref); res["clean_pass"] = all(clean[q] for q in HARD + SOFT_VIOL)
    expect = {"ispan": "q1_ispan", "tile": "q2_tile", "meta": "q3_meta", "sched": "q4_sched", "B": "q5_B", "nondet": "det",
              "seam": "q6_seam", "dwc_kernel": "q7_kernel_vs_oracle"}
    for fault, q in expect.items():
        try: r, _ = check_case(CpuSelf(fault), vol, 3, ref)
        except Exception as e: res["fault_detection"][fault] = f"raised {type(e).__name__}"; continue
        res["fault_detection"][fault] = {"expected_flag": q, "flag_value": bool(r[q]), "detected": not r[q]}
    # DWC negative controls on the actual backend kernel
    be = be_factory(); rng = np.random.default_rng([SEED, 10]); neg = {}
    chk = np.zeros((2, 2, 1), np.float32); chk[0, 0, 0] = chk[1, 1, 0] = 1
    cube = np.zeros((2, 2, 2), np.float32); cube[0, 0, 0] = cube[1, 1, 1] = 1
    nanf = rng.integers(0, 3, (4, 4, 4)).astype(np.float32); nanf[1, 2, 1] = np.nan
    rnd = [rng.integers(0, 3, (int(a), int(b), int(c))).astype(np.float32) for a, b, c in rng.integers(2, 7, (60, 3))]
    for name, arr in [("face_checker", chk), ("cube_only", cube), ("nan_field", nanf)] + [(f"rand{i}", r) for i, r in enumerate(rnd)]:
        kg, ko = be.dwc(arr), O.dwc_count_np(arr)
        same = (kg["violations"], kg["nan_skipped"], kg["total_blocks"]) == (ko["violations"], ko["nan_skipped"], ko["total_blocks"])
        neg[name] = dict(kernel=kg, oracle={k: ko[k] for k in ("violations", "nan_skipped", "total_blocks")}, counts_match=same)
    res["dwc_negative_controls"] = neg
    res["dwc_all_match"] = all(x["counts_match"] for x in neg.values())
    res["dwc_detects"] = neg["face_checker"]["kernel"]["violations"] == 1 and neg["cube_only"]["kernel"]["violations"] == 1 and neg["nan_field"]["kernel"]["nan_skipped"] > 0
    res["n_nonDWC_random_fields"] = sum(x["oracle"]["violations"] > 0 for n, x in neg.items() if n.startswith("rand"))
    res["pass"] = bool(res["clean_pass"] and all(isinstance(x, dict) and x["detected"] for x in res["fault_detection"].values())
                       and res["dwc_all_match"] and res["dwc_detects"] and res["n_nonDWC_random_fields"] >= 5)
    (out / "selftest.json").write_text(json.dumps(res, indent=1, default=str)); return res


# ------------------------------------------------------------------ tier runner
def run_tier(tier, be, out: Path, args):
    gen = {"T0": inputs_T0, "T1": lambda: inputs_T1(args.sample, args.stride), "T2": inputs_T2}[tier]()
    cases = list(gen)
    if args.part:
        i, n_ = (int(x) for x in args.part.split("/")); cases = cases[i::n_]     # i-th of n interleaved parts
    if args.max_cases: cases = cases[: args.max_cases]
    log = open(out / f"{tier}_cases.jsonl", "w"); t0 = time.time()
    stats = {q: 0 for q in HARD + SOFT_VIOL + INFO}; n = errors = 0; first_fail = None; stop = None; cex = False
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp   # spawn: never fork a process that already holds a CUDA context
    pool = ProcessPoolExecutor(args.cpu_workers, mp_context=mp.get_context("spawn")) if args.cpu_workers > 1 else None
    futs = [pool.submit(cpu_reference, v, K) for _, v, K in cases] if pool else None
    for i, (name, vol, K) in enumerate(cases):
        if time.time() - t0 > args.time_limit_s: stop = f"time_limit {args.time_limit_s}s after {n} cases"; break
        tc = time.time()
        try:
            r, repro = check_case(be, vol, K, futs[i].result() if futs else None, tie_info=(tier == "T0"), do_det=(tier != "T1"))
        except Exception:
            errors += 1; tb = traceback.format_exc(); first_fail = first_fail or (name, "error")
            (out / "witnesses").mkdir(exist_ok=True); np.save(out / "witnesses" / f"{name}_ERR_input.npy", vol)
            log.write(json.dumps(dict(case=name, K=K, error=tb)) + "\n")
            if errors >= 3: stop = "3 errors in tier (HOLD)"; break
            continue
        n += 1
        for q in stats: stats[q] += int(not r[q])
        r.update(case=name, K=K, shape=list(vol.shape), vol_sha=sha_arr(vol), sec=round(time.time() - tc, 4)); log.write(json.dumps(r, default=str) + "\n")
        bad = [q for q in HARD + SOFT_VIOL if not r[q]]
        if bad:
            wd = out / "witnesses"; wd.mkdir(exist_ok=True); np.savez(wd / f"{name}.npz", K=K, bad=np.array(bad), **repro)
            first_fail = (name, bad); cex = True; stop = f"counterexample at {name}: {bad}"; break
    log.close()
    if pool: pool.shutdown(cancel_futures=True)
    # fail-closed: any case error => HOLD (never PASS); mismatch witness => COUNTEREXAMPLE
    verdict = ("COUNTEREXAMPLE" if cex else "HOLD" if errors else "INCOMPLETE" if stop else "CONDITIONAL" if stats["q3_meta"] else "PASS")
    summ = dict(tier=tier, planned=len(cases), completed=n, errors=errors, fail_counts=stats, first_fail=first_fail, stop=stop,
                verdict=verdict, wall_s=round(time.time() - t0, 2), peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    (out / f"{tier}_summary.json").write_text(json.dumps(summ, indent=1, default=str)); return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, choices=["selftest", "T0", "T1", "T2"])
    ap.add_argument("--backend", default="gpu", choices=["gpu", "cpu-selftest"])
    ap.add_argument("--repo", type=Path, default=Path.home() / "jh/New-DDFP/v1/DD-FP-R7")
    ap.add_argument("--allow-unfrozen", action="store_true", help="skip SHA gate (recorded as such; never for evidence runs)")
    ap.add_argument("--fault", default=None); ap.add_argument("--out", type=Path, default=PKG / "results")
    ap.add_argument("--sample", type=int, default=None); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-cases", type=int, default=None); ap.add_argument("--time-limit-s", type=float, default=3600)
    ap.add_argument("--cpu-workers", type=int, default=1); ap.add_argument("--part", default=None, help="i/n interleaved slice of the tier")
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    env = dict(argv=sys.argv, python=sys.version, numpy=np.__version__, platform=platform.platform(), seed=SEED,
               nvidia_smi_start=nvidia_smi(), t_start=time.strftime("%Y-%m-%dT%H:%M:%S%z"), allow_shared=True)
    try:
        be = Gpu(a.repo, a.allow_unfrozen) if a.backend == "gpu" else CpuSelf(a.fault)
        env.update(be.env()); env["sources"] = check_sources(a.repo, True) if a.backend == "gpu" else "n/a (cpu-selftest)"
        env["lib_sha"] = {k: sha_bytes((PKG / "lib" / k).read_bytes()) == FROZEN[k] for k in ("seedref.py", "cpu_fp.py", "ibi_cpu.py")}
        (a.out / f"env_{a.tier}.json").write_text(json.dumps(env, indent=1, default=str))
        res = selftest(lambda: be, a.out) if a.tier == "selftest" else run_tier(a.tier, be, a.out, a)
        if a.backend == "gpu":  # source integrity after the run
            end = check_sources(a.repo, True); env["sources_unchanged"] = end["sha"] == env["sources"]["sha"]
        if a.backend == "gpu":  # diagnostics only: never let a metric failure discard a finished tier
            try: env["gpu_mempool_total_bytes_end"] = int(be.cp.get_default_memory_pool().total_bytes())
            except Exception as e: env["gpu_mempool_total_bytes_end"] = f"unavailable: {e!r}"
        env["nvidia_smi_end"] = nvidia_smi(); (a.out / f"env_{a.tier}.json").write_text(json.dumps(env, indent=1, default=str))
        print(json.dumps(res if a.tier != "selftest" else {k: res[k] for k in ("pass", "clean_pass", "dwc_all_match", "dwc_detects")}, default=str)[:1500])
        ok = res.get("pass") if a.tier == "selftest" else res["verdict"] in ("PASS", "CONDITIONAL")
        sys.exit(0 if ok else (2 if (a.tier != "selftest" and res["verdict"] == "COUNTEREXAMPLE") else 3))
    except SystemExit: raise
    except Exception:
        (a.out / f"HOLD_{a.tier}.txt").write_text(traceback.format_exc()); print(traceback.format_exc()); sys.exit(3)


if __name__ == "__main__":
    main()
