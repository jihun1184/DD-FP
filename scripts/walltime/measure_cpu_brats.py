"""
scripts/walltime/measure_cpu_brats.py
Direct CPU sequential FP timing on BraTS volumes.

Imports build_ispan_cpu, fp_cpu, preprocess, and run_ibi_v10 directly
from experiment_DDFP_all.py (sibling file in scripts/ddfp/).
"""
import sys, time, glob, os
import numpy as np
import nibabel as nib
from pathlib import Path
from collections import defaultdict

# Resolve project root so that scripts/ddfp/ is importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "ddfp"))

from experiment_DDFP_all import build_ispan_cpu, fp_cpu, preprocess, run_ibi_v10


def measure_one(vol_path: str, K: int = 4, delta: int = 1):
    name = Path(vol_path).parent.name
    print(f"\n[{name}]  loading...")

    vol_f  = nib.load(vol_path).get_fdata(dtype=np.float32)
    vol_u8 = preprocess(vol_f)
    W, H, D = vol_u8.shape
    print(f"  volume: {W}×{H}×{D}, expanded: {2*W-1}×{2*H-1}×{2*D-1}")
    print(f"  expanded total cells: {(2*W-1)*(2*H-1)*(2*D-1):,}")

    # CPU sequential FP (Algorithm 1 baseline)
    print("  CPU FP running...", end="", flush=True)
    t0 = time.time()
    U_lo, U_hi, li = build_ispan_cpu(vol_u8)
    u_cpu = fp_cpu(U_lo, U_hi, li)[1:-1, 1:-1, 1:-1]
    t_cpu = time.time() - t0
    print(f" {t_cpu:.1f}s")

    # GPU DD-FP K=4, δ=1
    print("  GPU DD-FP K=4 δ=1 running...", end="", flush=True)
    t0  = time.time()
    res = run_ibi_v10(vol_u8, K=K, delta=delta, max_rounds=8)
    t_dd = time.time() - t0
    print(f" {t_dd:.1f}s  (R*={res['R_star']})")

    speedup = t_cpu / max(t_dd, 1e-9)
    print(f"  Speedup (CPU→DD-FP K={K}): {speedup:.1f}×")
    return {"name": name, "t_cpu": t_cpu, "t_dd": t_dd,
            "speedup": speedup, "R_star": res["R_star"]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Direct CPU vs DD-FP wall-clock timing on BraTS volumes."
    )
    ap.add_argument("--brats", default="../data",
                    help="BraTS 2021 Training root directory")
    ap.add_argument("--n",     type=int, default=1,
                    help="number of subjects to time")
    ap.add_argument("--K",     type=int, default=4,
                    help="DD-FP subdomain count")
    ap.add_argument("--delta", type=int, default=1)
    args = ap.parse_args()

    paths = sorted(glob.glob(
        os.path.join(args.brats, "*", "*_flair.nii*")))[:args.n]
    if not paths:
        print(f"No BraTS files found in {args.brats}")
        sys.exit(1)

    print(f"BraTS {len(paths)} subjects CPU+GPU timing")
    print("=" * 50)
    results = []
    for p in paths:
        r = measure_one(p, K=args.K, delta=args.delta)
        results.append(r)

    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    t_cpus = [r["t_cpu"] for r in results]
    t_dds  = [r["t_dd"]  for r in results]
    spds   = [r["speedup"] for r in results]
    print(f"CPU FP:             {min(t_cpus):.1f} -- {max(t_cpus):.1f}s"
          f"  (mean {sum(t_cpus)/len(t_cpus):.1f}s)")
    print(f"DD-FP K={args.K} δ={args.delta}: {min(t_dds):.1f} -- {max(t_dds):.1f}s"
          f"  (mean {sum(t_dds)/len(t_dds):.1f}s)")
    print(f"Speedup:            {min(spds):.1f} -- {max(spds):.1f}×"
          f"  (mean {sum(spds)/len(spds):.1f}×)")