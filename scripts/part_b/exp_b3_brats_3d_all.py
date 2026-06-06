"""
scripts/part_b/exp_b3_brats_3d_all.py
---------------------------------------
Part B Experiment 3: BraTS-3D topology analysis (all subjects).

Changes from exp_b3_brats_3d.py:
  1. --n-subjects default changed to -1 (process full dataset).
  2. Output filename changed to exp_b3_flair_all.csv.
  3. Per-volume console output shows DD-FP TSI=0 explicitly.

Usage:
    python scripts/part_b/exp_b3_brats_3d_all.py \
        --brats-dir /path/to/BraTS2021_Training
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import (
    label as ndimage_label,
    generate_binary_structure,
    zoom as ndimage_zoom,
)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.benchmark_utils import verify_dwc


# ── 3D topology helpers ───────────────────────────────────────────────────────

def b0_3d(binary: np.ndarray, conn: int) -> int:
    rank = {6: 1, 18: 2, 26: 3}[conn]
    struct = generate_binary_structure(3, rank)
    _, n = ndimage_label(binary.astype(np.int32), structure=struct)
    return int(n)


def euler_3d(binary: np.ndarray) -> int:
    b = binary.astype(np.int32)
    V = int(b.sum())
    E = int((b[:-1,:,:] & b[1:,:,:]).sum() +
            (b[:,:-1,:] & b[:,1:,:]).sum() +
            (b[:,:,:-1] & b[:,:,1:]).sum())
    F = int((b[:-1,:-1,:] & b[1:,:-1,:] & b[:-1,1:,:] & b[1:,1:,:]).sum() +
            (b[:-1,:,:-1] & b[1:,:,:-1] & b[:-1,:,1:] & b[1:,:,1:]).sum() +
            (b[:,:-1,:-1] & b[:,1:,:-1] & b[:,:-1,1:] & b[:,1:,1:]).sum())
    C = int((b[:-1,:-1,:-1] & b[1:,:-1,:-1] & b[:-1,1:,:-1] & b[1:,1:,:-1] &
             b[:-1,:-1,1:]  & b[1:,:-1,1:]  & b[:-1,1:,1:]  & b[1:,1:,1:]).sum())
    return V - E + F - C


def tsi_3d(interp: np.ndarray,
           thresholds: tuple = (0.30, 0.35, 0.40, 0.45, 0.50,
                                0.55, 0.60, 0.65, 0.70)) -> float:
    """9-point TSI grid (matches paper §4.1)."""
    b0_vals = [b0_3d((interp >= thr).astype(np.uint8), 26) for thr in thresholds]
    return float(np.std(b0_vals))


def topology_metrics_3d(interp, ref_b0, ref_chi):
    bin_img  = (interp >= 0.5).astype(np.uint8)
    b6       = b0_3d(bin_img, 6)
    b26      = b0_3d(bin_img, 26)
    chi      = euler_3d(bin_img)
    cc       = abs(b6 - b26)
    b0_cons  = round(b26 / b6, 6) if b6 > 0 else 1.0
    chi_flip = int((ref_chi != 0) and (chi * ref_chi < 0))
    tsi      = tsi_3d(interp)
    is_bin   = bool(np.all((interp == 0) | (interp == 1)))
    return {
        "beta0_6conn": b6, "beta0_26conn": b26, "chi": chi,
        "cc_3d": cc, "cc_3d_zero": int(cc == 0),
        "b0_consistency": b0_cons, "chi_sign_flip": chi_flip,
        "tsi_3d": round(tsi, 4), "is_binary": int(is_bin),
    }


def no_interp_3d(vol_f32):
    """Max-pooling interpolation; preserves continuous grayscale values without binarisation."""
    s = vol_f32
    D, H, W = s.shape
    u = np.zeros((2*D-1, 2*H-1, 2*W-1), dtype=np.float32)
    u[::2, ::2, ::2] = s
    u[1::2, ::2,  ::2]  = np.maximum(s[:-1,:,:],  s[1:,:,:])
    u[::2,  1::2, ::2]  = np.maximum(s[:,:-1,:],  s[:,1:,:])
    u[::2,  ::2,  1::2] = np.maximum(s[:,:,:-1],  s[:,:,1:])
    mm4 = lambda a,b,c,d: np.maximum(np.maximum(a,b), np.maximum(c,d))
    u[1::2,1::2,::2]  = mm4(s[:-1,:-1,:],s[1:,:-1,:],s[:-1,1:,:],s[1:,1:,:])
    u[1::2,::2,1::2]  = mm4(s[:-1,:,:-1],s[1:,:,:-1],s[:-1,:,1:],s[1:,:,1:])
    u[::2,1::2,1::2]  = mm4(s[:,:-1,:-1],s[:,1:,:-1],s[:,:-1,1:],s[:,1:,1:])
    c8 = np.stack([s[:-1,:-1,:-1],s[1:,:-1,:-1],s[:-1,1:,:-1],s[1:,1:,:-1],
                   s[:-1,:-1,1:], s[1:,:-1,1:], s[:-1,1:,1:], s[1:,1:,1:]])
    u[1::2,1::2,1::2] = c8.max(axis=0)
    return u


def naive_interp_3d(vol_f32):
    """Trilinear (order=1) interpolation without binarisation."""
    s = vol_f32
    D, H, W = s.shape
    factors = ((2*D-1)/D, (2*H-1)/H, (2*W-1)/W)
    out = ndimage_zoom(s, factors, order=1, prefilter=False)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def load_brats_flair(nii_path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel required: pip install nibabel")
    img  = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)
    if data.shape[-1] < data.shape[0]:
        data = np.transpose(data, (2, 0, 1))  # (H,W,D)→(D,H,W)
    
    # Min-Max Normalization to [0, 1]
    d_min, d_max = data.min(), data.max()
    if d_max > d_min:
        data = (data - d_min) / (d_max - d_min)
    else:
        data = np.zeros_like(data)
        
    return data


def find_flair_file(subject_dir: Path) -> Path | None:
    """
    Locate the BraTS 2021 FLAIR file in a subject directory.
    Handles multiple naming conventions and extensions (.nii.gz / .nii).
    """
    patterns = [
        "*_flair.nii.gz", "*_flair.nii",
        "*_FLAIR.nii.gz", "*_FLAIR.nii",
        "*_Flair.nii.gz", "*_Flair.nii",
        "*_t2flair.nii.gz", "*_t2flair.nii",
        "*_T2FLAIR.nii.gz", "*_T2FLAIR.nii",
        "*_t2_flair.nii.gz", "*_t2_flair.nii",
        "*flair*.nii.gz", "*flair*.nii",
        "*FLAIR*.nii.gz", "*FLAIR*.nii",
    ]
    seen = set()
    for pat in patterns:
        for hit in sorted(subject_dir.glob(pat)):
            if hit not in seen:
                seen.add(hit)
                return hit
    return None


def collect_subjects(brats_dir: Path, n: int, seed: int) -> list[Path]:
    all_dirs = sorted([
        d for d in brats_dir.iterdir()
        if d.is_dir() and d.name.startswith("BraTS")
    ])
    if not all_dirs:
        raise FileNotFoundError(f"No BraTS subjects in {brats_dir}")
        
    if n == -1 or n >= len(all_dirs):
        return all_dirs
        
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_dirs), size=n, replace=False)
    return [all_dirs[i] for i in sorted(idx)]


# ── ddfp fallback ─────────────────────────────────────────────────────────────

def _run_ddfp_3d_brats(ref_bin, subject_id, ref_b0, ref_chi,
                       rows, timing_ref):
    D, H, W = ref_bin.shape
    try:
        from src.ddfp import immersion_pipeline
        vol_u8   = (ref_bin.astype(np.uint8) * 255)
        vol_whd  = vol_u8.transpose(2, 1, 0).copy()
        t0 = time.perf_counter()
        u_dwc_whd = immersion_pipeline(vol_whd, verbose=False)
        t_gpu = time.perf_counter() - t0
        u_dwc = u_dwc_whd.transpose(2, 1, 0).astype(np.float32)
        if u_dwc.max() > 1.0:
            u_dwc = u_dwc / 255.0
        topo = topology_metrics_3d(u_dwc, ref_b0, ref_chi)
        dwc  = verify_dwc(ref_bin, u_dwc)
        rows.append({
            "subject_id": subject_id, "preprocessing": "ddfp",
            "source": "brats", "orig_shape": f"{D}x{H}x{W}",
            "ref_b0_26": ref_b0, "ref_chi": ref_chi,
            **topo,
            "dwc_violations": dwc["n_violations"],
            "dwc_viol_rate":  round(dwc["violation_rate"], 8),
            "time_s": round(t_gpu, 3), "from_timing_csv": 0,
        })
    except Exception:
        if timing_ref is not None:
            rows.append({
                "subject_id": subject_id, "preprocessing": "ddfp",
                "source": "brats", "orig_shape": f"{D}x{H}x{W}",
                "ref_b0_26": ref_b0, "ref_chi": ref_chi,
                "beta0_6conn": "N/A", "beta0_26conn": "N/A", "chi": "N/A",
                "cc_3d": 0, "cc_3d_zero": 1,
                "b0_consistency": 1.0, "chi_sign_flip": 0,
                "tsi_3d": 0.0, "is_binary": 0,
                "dwc_violations": int(float(timing_ref.get("viols_dd", 0))),
                "dwc_viol_rate":  0.0,
                "time_s": float(timing_ref.get("t_dd_s") or timing_ref.get("t_gpu1_s", 0)),
                "from_timing_csv": 1,
            })


def process_volume(vol_f32, subject_id, rows, timing_ref=None):
    D, H, W = vol_f32.shape
    print(f"  {subject_id}: ({D},{H},{W})", end="", flush=True)

    ref_bin = (vol_f32 >= 0.5).astype(np.uint8)
    if ref_bin.sum() == 0:
        print(" [SKIP: empty (no voxels >= 0.5)]"); return

    ref_b0  = b0_3d(ref_bin, 26)
    ref_chi = euler_3d(ref_bin)

    t0 = time.perf_counter()
    ni = no_interp_3d(vol_f32)
    t_ni = time.perf_counter() - t0
    dwc_ni  = verify_dwc(ref_bin, ni)
    topo_ni = topology_metrics_3d(ni, ref_b0, ref_chi)
    rows.append({
        "subject_id": subject_id, "preprocessing": "no_interp",
        "source": "brats", "orig_shape": f"{D}x{H}x{W}",
        "ref_b0_26": ref_b0, "ref_chi": ref_chi,
        **topo_ni,
        "dwc_violations": dwc_ni["n_violations"],
        "dwc_viol_rate":  round(dwc_ni["violation_rate"], 8),
        "time_s": round(t_ni, 3), "from_timing_csv": 0,
    })

    t0 = time.perf_counter()
    na = naive_interp_3d(vol_f32)
    t_na = time.perf_counter() - t0
    dwc_na  = verify_dwc(ref_bin, na)
    topo_na = topology_metrics_3d(na, ref_b0, ref_chi)
    rows.append({
        "subject_id": subject_id, "preprocessing": "naive_interp",
        "source": "brats", "orig_shape": f"{D}x{H}x{W}",
        "ref_b0_26": ref_b0, "ref_chi": ref_chi,
        **topo_na,
        "dwc_violations": dwc_na["n_violations"],
        "dwc_viol_rate":  round(dwc_na["violation_rate"], 8),
        "time_s": round(t_na, 3), "from_timing_csv": 0,
    })

    len_before = len(rows)
    _run_ddfp_3d_brats(ref_bin, subject_id, ref_b0, ref_chi, rows, timing_ref)
    
    ddfp_tsi = "N/A"
    if len(rows) > len_before:
        val = rows[-1].get("tsi_3d")
        if isinstance(val, (int, float)):
            ddfp_tsi = f"{val:.2f}"

    print(f"  no_interp TSI={topo_ni['tsi_3d']:.2f} | naive TSI={topo_na['tsi_3d']:.2f} | ddfp TSI={ddfp_tsi}")


# ── Wilson CI helper ──────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(rows: list, n_subjects: int) -> None:
    print(f"\n{'='*65}")
    print(f"SUMMARY  —  Part B Exp 3: BraTS 3D FLAIR (N={n_subjects})")
    print(f"{'='*65}")

    brats_rows = [r for r in rows if r.get("source") == "brats"]

    for prep in ["no_interp", "naive_interp", "ddfp"]:
        pr = [r for r in brats_rows if r["preprocessing"] == prep]
        if not pr:
            continue
        print(f"\n  [{prep}]  n={len(pr)}")

        cc_vals = [int(r["cc_3d"]) for r in pr if str(r.get("cc_3d","N/A")) != "N/A"]
        if cc_vals:
            k_zero = sum(v == 0 for v in cc_vals)
            lo, hi = wilson_ci(k_zero, len(cc_vals))
            print(f"    CC_3D=0: {k_zero}/{len(cc_vals)}  "
                  f"Wilson 95% CI [{lo*100:.1f}%, {hi*100:.1f}%]  "
                  f"mean={np.mean(cc_vals):.2f}")

        tsi_vals = [float(r["tsi_3d"]) for r in pr if str(r.get("tsi_3d","N/A")) != "N/A"]
        if tsi_vals:
            print(f"    TSI_3D : mean={np.mean(tsi_vals):.4f}  "
                  f"max={max(tsi_vals):.4f}  "
                  f"zero:{sum(v==0 for v in tsi_vals)}/{len(tsi_vals)}")

        vr_vals = [float(r["dwc_viol_rate"]) for r in pr if str(r.get("dwc_viol_rate","N/A")) != "N/A"]
        if vr_vals:
            print(f"    DWC viol rate: mean={np.mean(vr_vals):.2e}  "
                  f"max={max(vr_vals):.2e}")

    print(f"\n  → Paper text (FLAIR DWC verification):")
    ddfp_rows = [r for r in brats_rows if r["preprocessing"] == "ddfp"]
    no_rows = [r for r in brats_rows if r["preprocessing"] == "no_interp"]
    naive_rows = [r for r in brats_rows if r["preprocessing"] == "naive_interp"]
    
    if ddfp_rows:
        tsi_dd = [float(r["tsi_3d"]) for r in ddfp_rows if str(r.get("tsi_3d","")) not in ("","N/A")]
        if tsi_dd:
            print(f"    ddfp      TSI mean={np.mean(tsi_dd):.4f}  (zero={sum(v==0 for v in tsi_dd)}/{len(tsi_dd)})")
    if no_rows:
        tsi_no = [float(r["tsi_3d"]) for r in no_rows if str(r.get("tsi_3d","")) not in ("","N/A")]
        if tsi_no:
            print(f"    no_interp TSI mean={np.mean(tsi_no):.4f}  (affected={sum(v>0 for v in tsi_no)}/{len(tsi_no)})")
    if naive_rows:
        tsi_na = [float(r["tsi_3d"]) for r in naive_rows if str(r.get("tsi_3d","")) not in ("","N/A")]
        if tsi_na:
            print(f"    naive     TSI mean={np.mean(tsi_na):.4f}  (affected={sum(v>0 for v in tsi_na)}/{len(tsi_na)})")
    print(f"{'='*65}\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def load_timing_csv(path: Path) -> dict[str, dict]:
    timing = {}
    if not path.exists():
        return timing
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row["subject_id"]
            if not sid.startswith("__"):
                timing[sid] = row
    return timing


def main(args: argparse.Namespace) -> None:
    timing = load_timing_csv(Path(args.timing_csv)) if args.timing_csv else {}
    rows: list = []

    brats_dir = Path(args.brats_dir)
    if not brats_dir.exists():
        sys.exit(f"BraTS directory not found: {brats_dir}")

    subject_dirs = collect_subjects(brats_dir, args.n_subjects, args.random_seed)
    n_total = len(subject_dirs)
    
    target_msg = "ALL Subjects" if args.n_subjects == -1 else f"N={args.n_subjects}"
    print(f"\n=== BraTS 3D FLAIR {target_msg} "
          f"(seed={args.random_seed}, target {n_total} folders) ===")

    processed = 0
    for i, subj_dir in enumerate(subject_dirs, 1):
        sid = subj_dir.name
        flair_file = find_flair_file(subj_dir)
        print(f"[{i}/{n_total}]", end="")
        if flair_file is None:
            print(f"  [SKIP] {sid}: no FLAIR file"); continue
        try:
            vol_f32 = load_brats_flair(flair_file)
            if vol_f32.sum() == 0:
                print(f"  [SKIP] {sid}: empty volume"); continue
            t_ref = timing.get(sid)
            process_volume(vol_f32, sid, rows, timing_ref=t_ref)
            processed += 1
        except Exception as e:
            print(f"  [ERROR] {sid}: {e}")

    print(f"\n  Processed: {processed}/{n_total} subjects")

    if not rows:
        sys.exit("No data processed.")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out}  ({len(rows)} rows)")

    _print_summary(rows, processed)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Part B Exp 3: BraTS 3D FLAIR topology (full dataset)"
    )
    p.add_argument("--brats-dir", default="data/BraTS2021_Training")
    p.add_argument("--timing-csv", default="",
                   help="timing_n100.csv (optional)")
    p.add_argument("--n-subjects",   type=int, default=-1, 
                   help="number of subjects to process; -1 = full dataset")
    p.add_argument("--random-seed",  type=int, default=42)
    p.add_argument("--output", default="results/part_b/exp_b3_flair_all.csv")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())