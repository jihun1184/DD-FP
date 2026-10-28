#!/usr/bin/env python3
"""R8-F1L Phase 0 items 1-6 on the UNMODIFIED production source under CPU stand-ins (shim mode). NOT GPU evidence.
Verifies for B = run_ibi_v10(max_rounds=0):  Round 0 runs exactly once; no IBI/seed round; owner assembly equals the
independent oracle; B is the Round-0 state of C; A/B/C input windows are identical. Counters wrap the two GPU entry points
that run_ibi_v10 resolves as module globals (production source untouched)."""
import json, sys, hashlib
from pathlib import Path
import numpy as np

V1 = Path(sys.argv[1]); OUT = Path(sys.argv[2]); NCASE = int(sys.argv[3]) if len(sys.argv) > 3 else 120
J = V1 / "R8F1J_ROUND0_GPU_CONFORMANCE_V1"
sys.path[:0] = [str(J / "lib"), str(J / "harness"), str(V1 / "DDFP_R7_GPU_SERVER_RUNNER_R0/r3_scripts")]
import cpu_fp, seedref, ibi_cpu, r8f1j_oracle as O
import r3_diff_ibi as D
prod = D.load_production(V1 / "DD-FP-R7", "shim", cpu_fp, seedref, "lower")      # hash-gated: raises if SHA != b6a9b109...
import numpy as np

calls = {"build": [], "fp": [], "pin": 0}
_b, _f, _p = prod.build_ispan_gpu, prod.front_propagation_gpu, prod._pin_and_seed_expanded
def build(vol): calls["build"].append(tuple(vol.shape)); return _b(vol)
def fp(lo, hi, l_inf, verbose=False, boundary_seeds=None, **k):
    calls["fp"].append(dict(shape=tuple(lo.shape), seeded=boundary_seeds is not None, n_seed=(0 if boundary_seeds is None else int((~np.isnan(boundary_seeds)).sum()))))
    return _f(lo, hi, l_inf, verbose=verbose, boundary_seeds=boundary_seeds, **k)
def pin(*a, **k): calls["pin"] += 1; return _p(*a, **k)
prod.build_ispan_gpu, prod.front_propagation_gpu, prod._pin_and_seed_expanded = build, fp, pin
def run(vol, K, rounds):
    calls["build"].clear(); calls["fp"].clear(); calls["pin"] = 0
    r = prod.run_ibi_v10(vol, K=K, delta=1, max_rounds=rounds)
    return r, dict(build=list(calls["build"]), fp=list(calls["fp"]), pin=calls["pin"])

def oracle_B(vol, K):                    # independent Round-0: per-window FP via CPU stand-ins + oracle owner assembly
    Wd = vol.shape[2]; tiles = []
    for (_z0, _z1, a, b) in O.tile_windows(Wd, K, 1):
        lo, hi, l = cpu_fp.build_ispan_cpu(vol[:, :, a:b])
        tiles.append(np.ascontiguousarray(seedref.seed_aware_cpu_reference(lo, hi, l, None, "lower")[0][1:-1, 1:-1, 1:-1], np.float32))
    return O.assemble_round0(tiles, Wd, K, 1)

rng = np.random.default_rng(20261018); res = []; fails = []
shapes = [(3, 3, 9), (4, 4, 12), (5, 5, 12), (6, 6, 15), (4, 5, 16)]; alph = [2, 4, 16, 256, 256, 256]
for i in range(NCASE):
    sh = shapes[rng.integers(len(shapes))]; A = int(alph[rng.integers(len(alph))]); K = int(rng.integers(2, min(4, sh[2] // 3) + 1))
    vol = (rng.integers(0, A, sh) * (255 // max(A - 1, 1))).astype(np.uint8)
    B, cB = run(vol, K, 0); C, cC = run(vol, K, 8); C1, cC1 = run(vol, K, 1)
    uB, uC = np.asarray(B["u_dd"], np.float32), np.asarray(C["u_dd"], np.float32)
    cpu0 = ibi_cpu.ibi(vol, K, 1, 8)["hist"][0].astype(np.float32); orc = oracle_B(vol, K)
    chk = dict(
        B_R_star_zero=B["R_star"] == 0,
        B_fp_calls_eq_K=len(cB["fp"]) == K and len(cB["build"]) == K,
        B_no_seeded_fp=not any(c["seeded"] for c in cB["fp"]),
        B_no_pin=cB["pin"] == 0,
        B_windows_eq_C_round0=cB["build"] == cC["build"][:K] and [c["shape"] for c in cB["fp"]] == [c["shape"] for c in cC["fp"][:K]],
        C_round0_unseeded=not any(c["seeded"] for c in cC["fp"][:K]),
        C_extra_rounds_seeded=all(c["seeded"] for c in cC["fp"][K:]) and len(cC["fp"]) == K * (1 + C["R_star"]) and C["R_star"] >= 1 and cC["pin"] > 0,
        counter_discriminates_max_rounds_1=len(cC1["fp"]) == 2 * K and cC1["pin"] > 0,      # negative control: counters DO see a second round
        B_eq_oracle_owner_assembly=bool(np.array_equal(uB.view(np.uint32), orc.view(np.uint32))),
        B_eq_cpu_round0_hist0=bool(np.array_equal(uB.view(np.uint32), cpu0.view(np.uint32))),
        B_finite=bool(np.isfinite(uB).all()), B_original_voxels=bool(np.array_equal(uB[::2, ::2, ::2], vol.astype(np.float32))),
        same_shape_dtype_ABC=uB.shape == uC.shape and uB.dtype == uC.dtype == np.float32)
    res.append(dict(i=i, shape=sh, K=K, A=A, B_eq_C=bool(np.array_equal(uB.view(np.uint32), uC.view(np.uint32))), R_C=int(C["R_star"]),
                    n_diff_B_C=int((uB.view(np.uint32) != uC.view(np.uint32)).sum()), **chk))
    if not all(v for k, v in chk.items()): fails.append((i, [k for k, v in chk.items() if not v]))
keys = [k for k in res[0] if k.startswith(("B_", "C_", "same", "counter"))]
summary = dict(n_cases=len(res), n_fail_cases=len(fails), fails=fails[:10], per_check_pass={k: sum(bool(r[k]) for r in res) for k in keys},
               n_B_ne_C=sum(not r["B_eq_C"] for r in res), mean_R_C=float(np.mean([r["R_C"] for r in res])),
               production_sha256=hashlib.sha256(Path(prod.__file__).read_bytes()).hexdigest(),
               scope="CPU shim stand-ins for build_ispan_gpu/front_propagation_gpu; verifies run_ibi_v10 control flow + owner assembly only; not GPU evidence")
OUT.write_text(json.dumps(dict(summary=summary, cases=res), indent=1, default=str) + "\n"); print(json.dumps(summary, indent=1, default=str))
sys.exit(1 if fails else 0)
