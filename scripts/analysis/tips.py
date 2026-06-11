"""
scripts/analysis/tips.py
=======================
Topological Integrity per Second (TIPS) measurement.
Implements paper eq:tips and reproduces tab:speedup.

    TIPS = N_safe / t_wall

where N_safe = |Ω̂| (expanded-lattice cell count) when the method provides
a formal DWC guarantee (Theorem 2.11), and N_safe = 0 otherwise.

Methods measured
----------------
  seq_cpu  : CPU Algorithm 1 sequential FP          — DWC guaranteed ✓
  gpu_k1   : GPU single-pass K=1 (no IBI)           — TIPS = 0 (no guarantee)
  ddfp_k16 : GPU DD-FP K=16, δ=1, IBI               — DWC guaranteed ✓

Scalability
-----------
  N=20   (paper tab:speedup): includes seq_cpu (~188s/subject → ~60 min total)
  N=1251 (full dataset)     : use --skip-cpu to measure only GPU methods
                              (~4.2s/subject × 1251 ≈ 87 min for DD-FP)

Usage
-----
  # Reproduce tab:speedup (N=20, with CPU)
  python scripts/analysis/tips.py \\
      --brats data/BraTS2021 --n 20 --seed 42

  # Full dataset TIPS, GPU only (~87 min)
  python scripts/analysis/tips.py \\
      --brats data/BraTS2021 --n 1251 --seed 42 --skip-cpu

  # Quick smoke-test (5 subjects, GPU only)
  python scripts/analysis/tips.py \\
      --brats data/BraTS2021 --n 5 --skip-cpu --K 16

Notes
-----
  Paper tab:speedup uses N=20 because CPU timing (~188s/subject) makes
  N=1251 impractical for the CPU row. Full-dataset TIPS (GPU only) is
  provided for supplementary reproducibility; it does not replace the
  paper's N=20 table values.
"""

from __future__ import annotations
import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path
from statistics import median, stdev, mean

import numpy as np
import nibabel as nib

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "ddfp"))

try:
    from experiment_DDFP_all import (
        build_ispan_cpu,
        fp_cpu,
        run_ibi_v10,
        count_boundary_violations,
    )
    from src.ddfp.gpu_immersion import (
        build_ispan_gpu,
        front_propagation_gpu,
    )
    import cupy as cp
    _GPU_OK = True
except (ImportError, ModuleNotFoundError):
    _GPU_OK = False


# ── Core formula (eq:tips) ────────────────────────────────────────────────────

def compute_tips(vol_shape: tuple[int, int, int],
                 elapsed_s: float,
                 dwc_guaranteed: bool) -> float:
    """
    TIPS = N_safe / t_wall  (paper eq:tips).

    N_safe: expanded-lattice cell count when formal DWC guarantee holds,
            0 otherwise.

    Counting rule:
      - seq_cpu and ddfp_k16 (δ≥1 + IBI): N_safe = |Ω̂| = (2W-1)(2H-1)(2D-1)
      - gpu_k1 (K=1, no IBI):             N_safe = 0
        A single undetected violation invalidates all downstream topological
        inference, so non-guaranteed throughput does not count as TIPS.

    Args:
        vol_shape      : (W, H, D) of the original uint8 volume
        elapsed_s      : wall-clock time in seconds
        dwc_guaranteed : True if Theorem 2.11 guarantee applies

    Returns:
        TIPS in cells/second, or 0.0 if not DWC-guaranteed
    """
    if not dwc_guaranteed or elapsed_s <= 0:
        return 0.0
    W, H, D = vol_shape
    n_safe = (2 * W - 1) * (2 * H - 1) * (2 * D - 1)
    return float(n_safe) / elapsed_s


def tips_summary_str(tips: float) -> str:
    """Format TIPS as scientific notation string matching paper style."""
    if tips == 0.0:
        return "0"
    import math
    exp    = int(math.floor(math.log10(tips)))
    coeff  = tips / 10 ** exp
    return f"{coeff:.1f}×10^{exp}"


# ── Timing functions ──────────────────────────────────────────────────────────

def time_seq_cpu(vol_u8: np.ndarray) -> dict:
    """CPU Algorithm 1 sequential FP (Boutry 2015)."""
    t0 = time.perf_counter()
    U_lo, U_hi, l_inf = build_ispan_cpu(vol_u8)
    fp_cpu(U_lo, U_hi, l_inf)
    elapsed = time.perf_counter() - t0
    return {"elapsed_s": elapsed, "dwc_guaranteed": True, "r_star": None}


def time_gpu_k1(vol_u8: np.ndarray) -> dict:
    """GPU single-pass K=1 (no IBI — no formal DWC guarantee)."""
    t0 = time.perf_counter()
    U_lo, U_hi, l_inf = build_ispan_gpu(vol_u8)
    u_pad = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)
    cp.cuda.Stream.null.synchronize()
    cp.asnumpy(u_pad)                   # include device→host transfer
    elapsed = time.perf_counter() - t0
    return {"elapsed_s": elapsed, "dwc_guaranteed": False, "r_star": None}


def time_ddfp(vol_u8: np.ndarray, K: int, delta: int,
              max_rounds: int = 8) -> dict:
    """GPU DD-FP IBI — DWC guaranteed by Theorem 2.11 when δ≥1."""
    t0 = time.perf_counter()
    res = run_ibi_v10(vol_u8, K=K, delta=delta,
                      max_rounds=max_rounds, verbose=False)
    elapsed = time.perf_counter() - t0
    slices_z  = [res["boundary_z_orig"]] if isinstance(
        res["boundary_z_orig"], int) else res["boundary_z_orig"]
    viols     = count_boundary_violations(res["u_dd"], slices_z)
    guaranteed = (viols == 0) and (delta >= 1)
    return {
        "elapsed_s":       elapsed,
        "dwc_guaranteed":  guaranteed,
        "r_star":          res["R_star"],
        "boundary_viols":  viols,
    }


# ── Subject discovery ─────────────────────────────────────────────────────────

def find_subjects(brats_dir: str, n: int, seed: int) -> list[tuple[str, str]]:
    """Locate FLAIR paths; sample n subjects with given seed."""
    paths = sorted(glob.glob(os.path.join(brats_dir, "*", "*_flair.nii*")))
    if not paths:
        sys.exit(f"[ERROR] No FLAIR files found in {brats_dir}")
    total = len(paths)
    if n <= 0 or n >= total:
        chosen = paths
    else:
        rng     = np.random.default_rng(seed)
        indices = sorted(rng.choice(total, size=n, replace=False))
        chosen  = [paths[i] for i in indices]
    return [(Path(p).parent.name, p) for p in chosen]


# ── GPU warmup ────────────────────────────────────────────────────────────────

def gpu_warmup(n: int = 2) -> None:
    print(f"GPU warmup ({n} rounds)...", flush=True)
    rng     = np.random.default_rng(0)
    dummy   = rng.integers(1, 256, (240, 240, 155), dtype=np.uint8)
    for _ in range(n):
        time_gpu_k1(dummy)
    print("  warmup done\n")


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_tips(brats_dir: str, n: int, seed: int, K: int, delta: int,
             max_rounds: int, skip_cpu: bool,
             output_csv: str, n_warmup: int) -> list[dict]:
    """Measure TIPS for all three methods across n subjects."""

    subjects = find_subjects(brats_dir, n, seed)
    actual_n = len(subjects)
    print(f"Subjects: {actual_n}  (requested={n}, seed={seed})")
    print(f"Config:   K={K}, δ={delta}, max_rounds={max_rounds}")
    if skip_cpu:
        print("Note: --skip-cpu set — seq_cpu rows omitted")
    print()

    gpu_warmup(n_warmup)

    records: list[dict] = []

    for i, (sid, path) in enumerate(subjects):
        print(f"[{i+1:4d}/{actual_n}] {sid}", flush=True)
        try:
            data = nib.load(path).get_fdata(dtype=np.float32)
            # Reorder to (D,H,W) if needed, then to (W,H,D) for immersion_pipeline
            if data.shape[-1] < data.shape[0]:
                data = np.transpose(data, (2, 0, 1))  # (H,W,D) → (D,H,W)
            # (D,H,W) → (W,H,D) with .copy() for C-contiguous GPU transfer
            vol = data.transpose(2, 1, 0).copy()
            # Normalise [0,255] uint8 via percentile clip (same as preprocess())
            nz = vol[vol > 0]
            if len(nz) == 0:
                print(f"  [SKIP] {sid}: empty volume"); continue
            p1, p99 = np.percentile(nz, [1, 99])
            out = np.zeros(vol.shape, np.float32)
            m = vol > 0
            out[m] = np.clip((vol[m] - p1) / (p99 - p1 + 1e-9), 0, 1)
            vol = (out * 255).astype(np.uint8)
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        shape = vol.shape

        # seq_cpu
        if not skip_cpu:
            r = time_seq_cpu(vol)
            tips = compute_tips(shape, r["elapsed_s"], r["dwc_guaranteed"])
            records.append({
                "subject": sid, "method": "seq_cpu",
                "elapsed_s": round(r["elapsed_s"], 2),
                "dwc_guaranteed": r["dwc_guaranteed"],
                "r_star": "", "boundary_viols": 0,
                "n_safe": (2*shape[0]-1)*(2*shape[1]-1)*(2*shape[2]-1),
                "tips": round(tips, 0),
            })
            print(f"  seq_cpu  {r['elapsed_s']:.1f}s  "
                  f"TIPS={tips_summary_str(tips)}", flush=True)

        # gpu_k1
        r = time_gpu_k1(vol)
        records.append({
            "subject": sid, "method": "gpu_k1",
            "elapsed_s": round(r["elapsed_s"], 3),
            "dwc_guaranteed": False,
            "r_star": "", "boundary_viols": 0, "n_safe": 0, "tips": 0,
        })
        print(f"  gpu_k1   {r['elapsed_s']:.3f}s  TIPS=0 (no guarantee)",
              flush=True)

        # ddfp
        r = time_ddfp(vol, K=K, delta=delta, max_rounds=max_rounds)
        tips = compute_tips(shape, r["elapsed_s"], r["dwc_guaranteed"])
        n_safe = (2*shape[0]-1)*(2*shape[1]-1)*(2*shape[2]-1) \
                 if r["dwc_guaranteed"] else 0
        records.append({
            "subject": sid, "method": f"ddfp_K{K}",
            "elapsed_s": round(r["elapsed_s"], 2),
            "dwc_guaranteed": r["dwc_guaranteed"],
            "r_star": r["r_star"], "boundary_viols": r["boundary_viols"],
            "n_safe": n_safe, "tips": round(tips, 0),
        })
        print(f"  ddfp_K{K} {r['elapsed_s']:.2f}s  "
              f"R*={r['r_star']}  viols={r['boundary_viols']}  "
              f"TIPS={tips_summary_str(tips)}", flush=True)

    _save_csv(records, output_csv)
    _print_summary(records, K, skip_cpu)
    return records


# ── Output helpers ────────────────────────────────────────────────────────────

def _save_csv(records: list[dict], path: str) -> None:
    if not records:
        return
    fieldnames = ["subject", "method", "elapsed_s", "dwc_guaranteed",
                  "r_star", "boundary_viols", "n_safe", "tips"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"\nSaved → {path}  ({len(records)} rows)")


def _stats(vals: list[float]) -> str:
    if not vals:
        return "—"
    m = mean(vals)
    s = stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.2f} ± {s:.2f}  [{min(vals):.2f}, {max(vals):.2f}]"


def _print_summary(records: list[dict], K: int, skip_cpu: bool) -> None:
    from collections import defaultdict
    by_method: dict[str, list] = defaultdict(list)
    for r in records:
        by_method[r["method"]].append(r)

    print(f"\n{'='*70}")
    print(f"  TIPS Summary  (N={len(set(r['subject'] for r in records))})")
    print(f"{'='*70}")
    print(f"  {'Method':<14}  {'Time (s)':^28}  {'TIPS (cells/s)':^20}  DWC")
    print(f"  {'─'*14}  {'─'*28}  {'─'*20}  {'─'*3}")

    order = (["seq_cpu"] if not skip_cpu else []) + ["gpu_k1", f"ddfp_K{K}"]
    for method in order:
        rows = by_method.get(method, [])
        if not rows:
            continue
        times = [r["elapsed_s"] for r in rows]
        tips  = [float(r["tips"]) for r in rows]
        dwc   = "✓" if rows[0]["dwc_guaranteed"] else "✗"
        tips_str = tips_summary_str(mean(tips)) if any(t > 0 for t in tips) else "0"
        print(f"  {method:<14}  {_stats(times):<28}  {tips_str:<20}  {dwc}")

    # DWC pass rate for ddfp
    ddfp_rows = by_method.get(f"ddfp_K{K}", [])
    if ddfp_rows:
        n_pass = sum(1 for r in ddfp_rows if r["dwc_guaranteed"])
        print(f"\n  DD-FP DWC pass: {n_pass}/{len(ddfp_rows)}")
    print(f"{'='*70}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Measure TIPS (Topological Integrity per Second) on BraTS 2021."
    )
    ap.add_argument("--brats",       required=True,
                    help="BraTS 2021 Training root directory")
    ap.add_argument("--n",           type=int, default=20,
                    help="number of subjects (default=20; -1 = full dataset)")
    ap.add_argument("--seed",        type=int, default=42,
                    help="random seed for subject sampling (default=42)")
    ap.add_argument("--K",           type=int, default=16,
                    help="DD-FP subdomain count (default=16, paper K=16)")
    ap.add_argument("--delta",       type=int, default=1,
                    help="DD-FP overlap (default=1)")
    ap.add_argument("--max-rounds",  type=int, default=8,
                    dest="max_rounds")
    ap.add_argument("--skip-cpu",    action="store_true",
                    dest="skip_cpu",
                    help="skip seq_cpu timing (use for N=1251 full dataset)")
    ap.add_argument("--output",      default="tips_results.csv",
                    help="output CSV path (default: tips_results.csv)")
    ap.add_argument("--warmup",      type=int, default=2,
                    dest="n_warmup",
                    help="GPU warmup rounds (default=2)")
    args = ap.parse_args()

    run_tips(
        brats_dir  = args.brats,
        n          = args.n if args.n > 0 else 10**9,
        seed       = args.seed,
        K          = args.K,
        delta      = args.delta,
        max_rounds = args.max_rounds,
        skip_cpu   = args.skip_cpu,
        output_csv = args.output,
        n_warmup   = args.n_warmup,
    )


if __name__ == "__main__":
    main()