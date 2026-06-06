"""
scripts/part_a/exp_a1_correctness.py
Exp A1: DWC Correctness Verification
"""
from __future__ import annotations
import csv, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.benchmark_utils import generate_synthetic_volume, verify_dwc
from src.preprocessing.preprocessor import NoInterpPreprocessor, NaiveInterpPreprocessor

try:
    import cupy as cp
    from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu
    _CUPY_OK = True
except Exception as e:
    _CUPY_OK = False
    print(f"[WARN] CuPy/GPU unavailable ({e}). DD-FP results will be skipped.")


def _make_cfg(prep_type):
    return SimpleNamespace(preprocessing=SimpleNamespace(
        type=prep_type, naive_mode="bilinear",
        ddfp_overlap=1, ddfp_cache_dir=None,
    ))


def ddfp_interp_2d(vol_f32):
    """DD-FP on 2D float32 [0,1]. Returns float32 expanded grid."""
    vol_u8 = (vol_f32 * 255).clip(0, 255).astype(np.uint8)
    vol3d  = vol_u8[:, :, np.newaxis]
    U_lo, U_hi, l_inf = build_ispan_gpu(vol3d)
    u_pad = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)
    return cp.asnumpy(u_pad[1:-1, 1:-1, 1:-1])[:, :, 0].astype(np.float32)


def ddfp_interp_3d(vol_u8):
    """DD-FP on 3D uint8. Returns float32 expanded grid."""
    U_lo, U_hi, l_inf = build_ispan_gpu(vol_u8)
    u_pad = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)
    return cp.asnumpy(u_pad[1:-1, 1:-1, 1:-1]).astype(np.float32)


def no_interp_3d(vol_u8):
    """3D no_interp: max-of-neighbours at every non-0-cell position."""
    s = vol_u8.astype(np.float32)
    W, H, D = s.shape
    u = np.zeros((2*W-1, 2*H-1, 2*D-1), dtype=np.float32)
    u[::2, ::2, ::2] = s
    u[1::2, ::2,  ::2]  = np.maximum(s[:-1,:,:], s[1:,:,:])
    u[::2,  1::2, ::2]  = np.maximum(s[:,:-1,:], s[:,1:,:])
    u[::2,  ::2,  1::2] = np.maximum(s[:,:,:-1], s[:,:,1:])
    mm4 = lambda a,b,c,d: np.maximum(np.maximum(a,b), np.maximum(c,d))
    u[1::2,1::2,::2]  = mm4(s[:-1,:-1,:],s[1:,:-1,:],s[:-1,1:,:],s[1:,1:,:])
    u[1::2,::2,1::2]  = mm4(s[:-1,:,:-1],s[1:,:,:-1],s[:-1,:,1:],s[1:,:,1:])
    u[::2,1::2,1::2]  = mm4(s[:,:-1,:-1],s[:,1:,:-1],s[:,:-1,1:],s[:,1:,1:])
    c8 = np.stack([s[:-1,:-1,:-1],s[1:,:-1,:-1],s[:-1,1:,:-1],s[1:,1:,:-1],
                   s[:-1,:-1,1:], s[1:,:-1,1:], s[:-1,1:,1:], s[1:,1:,1:]])
    u[1::2,1::2,1::2] = c8.max(0)
    return u


def naive_interp_3d(vol_u8):
    """3D naive: trilinear zoom."""
    from scipy.ndimage import zoom as ndimage_zoom
    factors = tuple((2*s-1)/s for s in vol_u8.shape)
    return np.clip(ndimage_zoom(vol_u8.astype(np.float32), factors, order=1),
                   0, 255).astype(np.float32)


def run_a1(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    prep_no    = NoInterpPreprocessor(_make_cfg("no_interp"))
    prep_naive = NaiveInterpPreprocessor(_make_cfg("naive_interp"))

    csv_path   = out_dir / "a1_correctness.csv"
    fieldnames = ["image_id","ndim","orig_shape","method",
                  "n_violations","max_abs_error","violation_rate","time_s","dwc_ok"]
    rows        = []
    ddfp_all_ok = True

    print(f"\n{'='*70}")
    print("  Exp A1: DWC Correctness")
    print(f"{'='*70}")
    print(f"  {'image_id':<28} {'method':<20} {'viols':>8} {'rate':>10}  {'ok':>4}")
    print(f"  {'-'*68}")

    def _record(iid, ndim, shape_str, method_name, interp_fn, vol_arg, vol_u8_arg):
        nonlocal ddfp_all_ok
        t0 = time.perf_counter()
        try:
            interp  = interp_fn(vol_arg)
            elapsed = time.perf_counter() - t0
            result  = verify_dwc(vol_u8_arg, interp)
        except Exception as ex:
            print(f"  [{iid}] {method_name}: ERROR {ex}")
            return
        ok = result["n_violations"] == 0
        if method_name == "ddfp_delta1" and not ok:
            ddfp_all_ok = False
        print(f"  {iid:<28} {method_name:<20} "
              f"{result['n_violations']:>8,}  "
              f"{result['violation_rate']:>9.4f}  "
              f"{'OK' if ok else 'FAIL':>4}")
        rows.append({
            "image_id": iid, "ndim": ndim, "orig_shape": shape_str,
            "method": method_name,
            "n_violations": result["n_violations"],
            "max_abs_error": round(result["max_abs_error"], 4),
            "violation_rate": round(result["violation_rate"], 8),
            "time_s": round(elapsed, 4), "dwc_ok": ok,
        })

    # 2D
    for seed in range(3):
        vol_u8  = generate_synthetic_volume((256, 256), seed=seed)
        vol_f32 = vol_u8.astype(np.float32) / 255.0
        iid, shape = f"synth_2d_s{seed}", "256x256"
        if _CUPY_OK:
            _record(iid, 2, shape, "ddfp_delta1", ddfp_interp_2d, vol_f32, vol_u8)
        _record(iid, 2, shape, "no_interp",    lambda v: prep_no(v)[0],    vol_f32, vol_u8)
        _record(iid, 2, shape, "naive_interp", lambda v: prep_naive(v)[0], vol_f32, vol_u8)

    # 3D
    for seed in range(2):
        vol_u8 = generate_synthetic_volume((64, 64, 64), seed=10+seed)
        iid, shape = f"synth_3d_64_s{seed}", "64x64x64"
        if _CUPY_OK:
            _record(iid, 3, shape, "ddfp_delta1", ddfp_interp_3d, vol_u8, vol_u8)
        _record(iid, 3, shape, "no_interp",    no_interp_3d,    vol_u8, vol_u8)
        _record(iid, 3, shape, "naive_interp", naive_interp_3d, vol_u8, vol_u8)

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)

    ddfp_rows  = [r for r in rows if r["method"] == "ddfp_delta1"]
    no_rows    = [r for r in rows if r["method"] == "no_interp"]
    naive_rows = [r for r in rows if r["method"] == "naive_interp"]

    print(f"\n  {'--- Summary ---':^66}")
    if ddfp_rows:
        all_zero = all(r["n_violations"] == 0 for r in ddfp_rows)
        print(f"  DD-FP   violations=0 for all images: {'PASS' if all_zero else 'FAIL'}")
    if no_rows:
        print(f"  no_interp has violations:    "
              f"{'YES (expected)' if any(r['n_violations']>0 for r in no_rows) else 'NO (unexpected)'}")
    if naive_rows:
        print(f"  naive_interp has violations: "
              f"{'YES (expected)' if any(r['n_violations']>0 for r in naive_rows) else 'NO (unexpected)'}")

    print(f"\n  CSV -> {csv_path}")
    print(f"{'='*70}\n")

    if ddfp_rows and not ddfp_all_ok:
        raise AssertionError("A1 FAILED: DD-FP produced DWC violations!")


if __name__ == "__main__":
    run_a1(ROOT / "results" / "part_a")
