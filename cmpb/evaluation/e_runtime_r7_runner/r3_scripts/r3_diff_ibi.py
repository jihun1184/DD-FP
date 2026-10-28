#!/usr/bin/env python3
"""R3-A  Differential test:  production `run_ibi_v10`  vs  R1 CPU port `ibi_cpu.ibi`   (bitwise, float32).

Independent verification script.  It IMPORTS the (hash-checked, unmodified) production file and never edits it.

Modes
  shim : production `experiment_DDFP_all.py` is executed unchanged, but `cupy` / `nibabel` / `gpu_immersion` are replaced by
         CPU stand-ins whose FP kernel is the SAME reference (`seedref.seed_aware_cpu_reference`) used by the port.
         => tests the SCHEDULER (pin ranges, assembly order, stop rule) only.  It says nothing about CUDA kernels.
  gpu  : real cupy + real `gpu_immersion` (needs an NVIDIA GPU).  => tests scheduler + kernels end to end.
Exit codes: 0 all cases identical (and, if --mutation-test, both injected faults detected) | 1 mismatch / undetected fault
            2 environment or integrity problem (nothing compared) | 3 internal error
Usage:  python -I r3_diff_ibi.py --repo <patched DD-FP tree> --lib <dir with cpu_fp.py seedref.py ibi_cpu.py> --mode shim|gpu
                                 [--n-random 60] [--seed 20261014] [--tie lower] [--mutation-test] --out result.json
"""
import argparse, hashlib, importlib.util, json, platform, sys, time, traceback, types
from pathlib import Path
import numpy as np

PROD_SHA = "b6a9b109f4105e7b77ebed880f669ba670bbd87123deca9509ba5b00993e561e"          # experiment_DDFP_all.py (production)
LIB_SHA = {"cpu_fp.py": "8eae554f7d3d688dee2285f41d20bfbf6e34af98016a3f1ad07f1ff5f8f36a88",
           "seedref.py": "176e0a59f3330eaec7a32cc073f8269a6b0f53611b64686044aa08da00753296",
           "ibi_cpu.py": "6c361c49267abb468ae54ec95ab5990e7d8f58a1ac8279748991768fe3f351cf"}


def sha(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def fail(code, msg): print("ERROR:", msg, file=sys.stderr); sys.exit(code)


def load_lib(lib: Path):
    """Import the R1 CPU engine after verifying its bytes against the frozen hashes."""
    for name, want in LIB_SHA.items():
        p = lib / name
        if not p.is_file(): fail(2, f"missing {p}")
        if sha(p.read_bytes()) != want: fail(2, f"hash mismatch {name}")
    sys.path.insert(0, str(lib))
    import cpu_fp, seedref, ibi_cpu
    return cpu_fp, seedref, ibi_cpu


def load_production(repo: Path, mode: str, cpu_fp, seedref, tie: str):
    path = repo / "scripts/ddfp/experiment_DDFP_all.py"
    if not path.is_file(): fail(2, f"missing {path}")
    if sha(path.read_bytes()) != PROD_SHA: fail(2, "production experiment_DDFP_all.py hash mismatch (wrong variant?)")
    if mode == "shim":                                    # CPU stand-ins; the production source itself is untouched
        cp = types.ModuleType("cupy")
        cp.zeros = lambda s, dtype=None: np.zeros(s, dtype); cp.full = lambda s, v, dtype=None: np.full(s, v, dtype)
        cp.nan, cp.float32, cp.abs, cp.asnumpy = np.nan, np.float32, np.abs, np.asarray
        cp.cuda = types.SimpleNamespace(Stream=types.SimpleNamespace(null=types.SimpleNamespace(synchronize=lambda: None)))
        gi = types.ModuleType("src.ddfp.gpu_immersion")
        gi.build_ispan_gpu = cpu_fp.build_ispan_cpu
        def fp(lo, hi, l_inf, verbose=False, boundary_seeds=None, **_):
            return seedref.seed_aware_cpu_reference(lo, hi, l_inf, boundary_seeds, tie)[0]
        gi.front_propagation_gpu = fp
        cf = types.ModuleType("src.ddfp.cpu_fp"); cf.build_ispan_cpu, cf.fp_cpu = cpu_fp.build_ispan_cpu, cpu_fp.fp_cpu
        pk = types.ModuleType("src"); pk.__path__ = []; pk2 = types.ModuleType("src.ddfp"); pk2.__path__ = []
        nb = types.ModuleType("nibabel")
        sys.modules.update({"cupy": cp, "nibabel": nb, "src": pk, "src.ddfp": pk2, "src.ddfp.gpu_immersion": gi, "src.ddfp.cpu_fp": cf})
    spec = importlib.util.spec_from_file_location("ddfp_prod", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)      # runs unmodified production source
    return mod


def make_cases(prod, n_random, seed):
    """Deterministic case list: the 3x3x9 census example, the two production synthetics, then random small volumes."""
    rng = np.random.default_rng(seed)
    yield dict(id="census_3x3x9_K3_d1", vol=rng.integers(0, 4, (3, 3, 9), dtype=np.uint8), K=3, delta=1)
    for name, v in prod.make_synth_volumes(10).items():
        for K in (2, 3): yield dict(id=f"{name}_n10_K{K}_d1", vol=v, K=K, delta=1)
    shapes = [(3, 3, 9), (5, 5, 12), (6, 6, 15), (7, 6, 16)]; alph = [2, 4, 16, 256, 256, 256]       # 256-level volumes are the ones that need IBI round 2 (discriminating for pinning)
    for i in range(n_random):
        sh = shapes[rng.integers(len(shapes))]; a = int(alph[rng.integers(len(alph))])
        K = int(rng.integers(2, min(4, sh[2] // 3) + 1)); d = int(rng.integers(0, 3))   # delta=0 is a negative control (no co-containment)
        yield dict(id=f"rand{i:03d}_{sh[0]}x{sh[1]}x{sh[2]}_a{a}_K{K}_d{d}", vol=rng.integers(0, a, sh).astype(np.uint8) * (255 // max(a - 1, 1)), K=K, delta=d)


def compare(case, prod, ibi_cpu, tie):
    v, K, d = np.ascontiguousarray(case["vol"], np.uint8), case["K"], case["delta"]
    t0 = time.perf_counter(); a = prod.run_ibi_v10(v, K=K, delta=d); ta = time.perf_counter() - t0
    t0 = time.perf_counter(); b = ibi_cpu.ibi(v, K, d, tie=tie); tb = time.perf_counter() - t0
    ua, ub = np.asarray(a["u_dd"], np.float32), np.asarray(b["u"], np.float32)
    shape_ok = ua.shape == ub.shape
    diff = (ua.view(np.uint32) != ub.view(np.uint32)) if shape_ok else None          # bitwise (NaN-safe)
    n_diff = int(diff.sum()) if shape_ok else -1
    return dict(id=case["id"], shape=list(v.shape), K=K, delta=d, vol_sha256=sha(v.tobytes()), out_shape_equal=shape_ok,
                R_prod=int(a["R_star"]), R_port=int(b["R"]), bz_equal=list(a["boundary_z_orig"]) == list(b["bz"]),
                n_cells=int(ua.size), n_diff=n_diff, first_diff=(list(map(int, np.argwhere(diff)[0])) if n_diff > 0 else None),
                nonfinite=int((~np.isfinite(ua)).sum() + (~np.isfinite(ub)).sum()),
                seam_prod=int(prod.count_boundary_violations(ua, a["boundary_z_orig"])), seam_port=int(ibi_cpu.seam_viol(ub, b["bz"])),
                sha_prod=sha(ua.tobytes()), sha_port=sha(ub.tobytes()), t_prod_s=round(ta, 4), t_port_s=round(tb, 4),
                identical=bool(shape_ok and n_diff == 0 and a["R_star"] == b["R"] and list(a["boundary_z_orig"]) == list(b["bz"])))


def run_suite(cases, prod, ibi_cpu, tie):
    out = []
    for c in cases:
        try: out.append(compare(c, prod, ibi_cpu, tie))
        except Exception as e: out.append(dict(id=c["id"], identical=False, error=f"{type(e).__name__}: {e}"))
    return out


def mutants(prod):
    """Two scheduler faults injected in memory (production file on disk stays untouched); used only to prove the comparator can fail."""
    orig_asm, orig_pin = prod._assemble_boundary, prod._pin_and_seed_expanded
    def asm_no_left(u_target, u_sub, z_b, z0_ext, D2):                      # drops the (2z_b-1)-cell of the boundary write
        s = u_target[:, :, 2 * z_b - 1].copy() if 2 * z_b - 1 >= 0 else None
        orig_asm(u_target, u_sub, z_b, z0_ext, D2)
        if s is not None: u_target[:, :, 2 * z_b - 1] = s
    def pin_short(U_lo, U_hi, seeds, asm, sub_z_start, g0, g1, W2s, H2s):  # pins one layer too few
        orig_pin(U_lo, U_hi, seeds, asm, sub_z_start, g0, max(g0, g1 - 1), W2s, H2s)
    return {"mut_boundary_write_skipped": ("_assemble_boundary", asm_no_left, orig_asm),
            "mut_pin_range_short": ("_pin_and_seed_expanded", pin_short, orig_pin)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, type=Path); ap.add_argument("--lib", required=True, type=Path)
    ap.add_argument("--mode", choices=["shim", "gpu"], required=True); ap.add_argument("--n-random", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261014); ap.add_argument("--tie", choices=["lower", "higher"], default="lower")
    ap.add_argument("--mutation-test", action="store_true"); ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if a.mutation_test and a.mode != "shim": fail(2, "--mutation-test is only defined for --mode shim")
    cpu_fp, seedref, ibi_cpu = load_lib(a.lib)
    try: prod = load_production(a.repo, a.mode, cpu_fp, seedref, a.tie)
    except SystemExit: raise
    except Exception as e: fail(2, f"cannot import production module in mode={a.mode}: {type(e).__name__}: {e}")
    cases = list(make_cases(prod, a.n_random, a.seed))
    rec = dict(script="r3_diff_ibi.py", mode=a.mode, tie=a.tie, seed=a.seed, n_cases=len(cases), prod_sha256=PROD_SHA, lib_sha256=LIB_SHA,
               python=platform.python_version(), numpy=np.__version__, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if a.mode == "gpu":
        import cupy as cp; rec["gpu"] = cp.cuda.runtime.getDeviceProperties(0)["name"].decode(); rec["cupy"] = cp.__version__
    res = run_suite(cases, prod, ibi_cpu, a.tie); rec["cases"] = res
    rec["n_identical"] = sum(r.get("identical", False) for r in res); rec["all_identical"] = rec["n_identical"] == len(res)
    rec["n_cases_R_ge_2"] = sum(r.get("R_prod", 0) >= 2 for r in res)       # cases whose pinned rounds actually change the output
    rec["rounds_hist"] = {str(k): sum(r.get("R_prod") == k for r in res) for k in sorted({r.get("R_prod") for r in res if "R_prod" in r})}
    ok = rec["all_identical"]
    if a.mutation_test:                                                    # the comparator must FAIL on injected faults
        det = {}
        for name, (attr, fn, orig) in mutants(prod).items():
            setattr(prod, attr, fn)
            try: mres = run_suite(cases, prod, ibi_cpu, a.tie)
            finally: setattr(prod, attr, orig)
            det[name] = dict(n_cases_flagged=sum(not r.get("identical", False) for r in mres), n_cases=len(mres))
        rec["mutation_test"] = det; ok = ok and all(v["n_cases_flagged"] > 0 for v in det.values())
    rec["verdict"] = "PASS" if ok else "FAIL"; a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1)); print(json.dumps({k: v for k, v in rec.items() if k != "cases"}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try: main()
    except SystemExit: raise
    except Exception: traceback.print_exc(); sys.exit(3)
