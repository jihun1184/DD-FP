"""
scripts/walltime/gen_timing_n100.py  (K=16 variant)
------------------------------------------------------
Generate timing_k16.csv: BraTS2021 N=100 subjects wall-clock timing.

Changes from brats_n100_sampling.py:
  v4: seq_cpu uses build_ispan_cpu() + fp_cpu() (pure CPU sequential FP),
      consistent with measure_cpu_brats.py.
  v3: added --K argument (default=16) for K=16 timing.
  v2: GPU timing uses perf_counter (avoids CuPy Event API compatibility issues).

Usage:
    # K=16, timing_k16.csv
    python scripts/walltime/gen_timing_n100.py \
        --brats-dir /path/to/BraTS2021_Training \
        --output timing_k16.csv \
        --n-subjects 100 --random-seed 42 --skip-cpu --K 16

    # K=16 with seq_cpu
    python scripts/walltime/gen_timing_n100.py \
        --brats-dir /path/to/BraTS2021_Training \
        --output timing_k16_withcpu.csv \
        --n-subjects 20 --random-seed 42 --K 16
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _candidate in [_HERE, _HERE.parent, _HERE.parent.parent]:
    if (_candidate / "src" / "ddfp").exists():
        sys.path.insert(0, str(_candidate))
        break

# ── GPU imports ───────────────────────────────────────────────────────────────
try:
    import cupy as cp
    from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu
    _CUPY_OK = True
except Exception as e:
    _CUPY_OK = False
    print(f"[WARN] CuPy/GPU unavailable: {e}")

try:
    import nibabel as nib
    _NIB_OK = True
except ImportError:
    _NIB_OK = False
    print("[WARN] nibabel not found — pip install nibabel")

try:
    from src.utils.benchmark_utils import verify_dwc
    _BENCH_OK = True
except Exception:
    _BENCH_OK = False

try:
    _DDFP_DIR = str(Path(__file__).resolve().parent.parent / "ddfp")
    if _DDFP_DIR not in sys.path:
        sys.path.insert(0, _DDFP_DIR)
    from experiment_DDFP_all import (
        split_domain_1d       as _split_domain_1d,
        _pin_and_seed_expanded,
        _assemble_boundary,
        run_ibi_v10, count_boundary_violations
    )
    _DDFP_OK = True
except (ImportError, ModuleNotFoundError) as _e:
    _DDFP_OK = False
    print(f"[WARN] experiment_DDFP_all not found: {_e}")


# ── Constants ─────────────────────────────────────────────────────────────────
# N_safe computed per-volume: (2W-1)(2H-1)(2D-1) — no hardcoded constant

FIELDNAMES = [
    "subject_id", "t_cpu_s", "t_gpu1_s", "t_dd_s",
    "speedup_gpu1", "speedup_dd", "r_star", "viols_dd",
    "dwc_guaranteed_dd", "tips_cpu", "tips_gpu1", "tips_dd",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_brats_flair(nii_path: Path) -> np.ndarray:
    """Load BraTS FLAIR NIfTI and normalise to uint8 [0,255].

    Uses percentile clip [p1,p99] on non-zero voxels — matches preprocess()
    in experiment_DDFP_all.py so timing is consistent with exp_a2/tips.py.
    Volume is returned in (W, H, D) order (C-contiguous) for immersion_pipeline.
    """
    img  = nib.load(str(nii_path))
    data = img.get_fdata(dtype=np.float32)
    if data.shape[-1] < data.shape[0]:
        data = np.transpose(data, (2, 0, 1))  # (H,W,D) → (D,H,W)
    # (D,H,W) → (W,H,D) C-contiguous for GPU transfer
    data = data.transpose(2, 1, 0).copy()
    nz = data[data > 0]
    if len(nz) == 0:
        return np.zeros(data.shape, dtype=np.uint8)
    p1, p99 = float(np.percentile(nz, 1)), float(np.percentile(nz, 99))
    out = np.zeros(data.shape, np.float32)
    m = data > 0
    out[m] = np.clip((data[m] - p1) / (p99 - p1 + 1e-9), 0, 1)
    return (out * 255).astype(np.uint8)


def find_flair_file(subject_dir: Path) -> Path | None:
    for pat in ["*_flair.nii.gz", "*_flair.nii",
                "*_t2flair.nii.gz", "*flair*.nii.gz"]:
        hits = sorted(subject_dir.glob(pat))
        if hits:
            return hits[0]
    return None


def collect_subjects(brats_dir: Path, n: int, seed: int) -> list[Path]:
    all_dirs = sorted([
        d for d in brats_dir.iterdir()
        if d.is_dir() and d.name.startswith("BraTS")
    ])
    if not all_dirs:
        raise FileNotFoundError(f"No BraTS subjects in {brats_dir}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(all_dirs), size=min(n, len(all_dirs)), replace=False)
    return [all_dirs[i] for i in sorted(idx)]


# ── Per-subject timing ────────────────────────────────────────────────────────

def _time_gpu1(vol_whd: np.ndarray, n_repeats: int) -> float:
    """GPU single-pass (K=1). Uses perf_counter to avoid CuPy Event API issues."""
    times = []
    for i in range(n_repeats + 1):
        t0 = time.perf_counter()
        U_lo, U_hi, l_inf = build_ispan_gpu(vol_whd)
        _ = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - t0
        if i > 0:
            times.append(elapsed)
    return float(np.median(times))


def _time_dd(vol_whd: np.ndarray, n_repeats: int,
             K: int = 16, delta: int = 1) -> tuple[float, int, int]:
    """DD-FP IBI (K=args.K, delta=1) using run_ibi_v10."""
    times, r_stars, viols = [], [], []
    for i in range(n_repeats + 1):
        result = run_ibi_v10(vol_whd, K=K, delta=delta)
        if i > 0:
            times.append(result["t_total_s"])
            r_stars.append(result["R_star"])
            u_dd = result["u_dd"]
            n_viols = count_boundary_violations(
                u_dd, result["boundary_z_orig"]
            )
            viols.append(n_viols)
    return (float(np.median(times)),
            int(np.median(r_stars)),
            int(np.median(viols)))


def process_subject(sid: str, flair_path: Path,
                    n_repeats: int, skip_cpu: bool,
                    K: int = 16) -> dict | None:
    try:
        vol_whd = load_brats_flair(flair_path)       # (W,H,D) C-contiguous
    except Exception as e:
        print(f"  [ERROR] {sid}: load — {e}")
        return None

    if vol_whd.sum() == 0:
        print(f"  [SKIP]  {sid}: empty volume")
        return None

    print(f"  {sid}", end="", flush=True)

    t_cpu = None
    if not skip_cpu:
        try:
            from experiment_DDFP_all import build_ispan_cpu, fp_cpu
            times = []
            for i in range(n_repeats + 1):
                t0 = time.perf_counter()
                U_lo, U_hi, li = build_ispan_cpu(vol_whd)
                _ = fp_cpu(U_lo, U_hi, li)[1:-1, 1:-1, 1:-1]
                elapsed = time.perf_counter() - t0
                if i > 0:
                    times.append(elapsed)
            t_cpu = float(np.median(times))
            print(f"  cpu={t_cpu:.1f}s", end="", flush=True)
        except Exception as e:
            print(f"  cpu=ERR({e})", end="", flush=True)

    # GPU single-pass
    t_gpu1 = None
    if _CUPY_OK:
        try:
            t_gpu1 = _time_gpu1(vol_whd, n_repeats)
            print(f"  gpu1={t_gpu1:.3f}s", end="", flush=True)
        except Exception as e:
            print(f"  gpu1=ERR({e})", end="", flush=True)

    # DD-FP IBI
    t_dd, r_star, viols_dd = None, 2, 0
    if _CUPY_OK:
        try:
            t_dd, r_star, viols_dd = _time_dd(vol_whd, n_repeats, K=K)
            print(f"  dd={t_dd:.2f}s r*={r_star} viols={viols_dd}", end="", flush=True)
        except Exception as e:
            print(f"  dd=ERR({e})", end="", flush=True)

    print()

    dwc_ok = (viols_dd == 0)

    def _tips(t, guaranteed):
        if not t or t == 0:
            return None
        return round((2*vol_whd.shape[0]-1)*(2*vol_whd.shape[1]-1)*(2*vol_whd.shape[2]-1) / t, 1)

    speedup_gpu1 = round(t_cpu / t_gpu1, 1) if (t_cpu and t_gpu1) else None
    speedup_dd   = round(t_cpu / t_dd,   1) if (t_cpu and t_dd)   else None

    return {
        "subject_id":        sid,
        "t_cpu_s":           round(t_cpu,  2) if t_cpu  else None,
        "t_gpu1_s":          round(t_gpu1, 3) if t_gpu1 else None,
        "t_dd_s":            round(t_dd,   2) if t_dd   else None,
        "speedup_gpu1":      speedup_gpu1,
        "speedup_dd":        speedup_dd,
        "r_star":            r_star,
        "viols_dd":          viols_dd,
        "dwc_guaranteed_dd": dwc_ok,
        "tips_cpu":          _tips(t_cpu,  True),
        "tips_gpu1":         0,
        "tips_dd":           _tips(t_dd, dwc_ok),
    }


# ── Summary stats ─────────────────────────────────────────────────────────────

def append_summary_stats(rows: list[dict]) -> list[dict]:
    numeric = ["t_cpu_s", "t_gpu1_s", "t_dd_s",
               "speedup_gpu1", "speedup_dd",
               "r_star", "viols_dd",
               "tips_cpu", "tips_gpu1", "tips_dd"]
    data = [r for r in rows if not str(r.get("subject_id","")).startswith("__")]

    def _stat(name, fn):
        row = {"subject_id": name}
        for col in FIELDNAMES[1:]:
            if col not in numeric:
                row[col] = ""
                continue
            vals = [float(r[col]) for r in data
                    if r.get(col) not in (None, "", "None")]
            row[col] = round(fn(vals), 3) if vals else ""
        return row

    return rows + [
        _stat("__STAT_MEAN__",   np.mean),
        _stat("__STAT_STD__",    np.std),
        _stat("__STAT_MIN__",    np.min),
        _stat("__STAT_MAX__",    np.max),
        _stat("__STAT_MEDIAN__", np.median),
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    if not _NIB_OK:
        sys.exit("nibabel required: pip install nibabel")
    if not _CUPY_OK:
        sys.exit("CuPy required for GPU timing.")

    brats_dir = Path(args.brats_dir)
    if not brats_dir.exists():
        sys.exit(f"BraTS dir not found: {brats_dir}")

    print(f"\n{'='*65}")
    print(f"  gen_timing_n100.py  v3 (K={args.K})")
    print(f"  BraTS  : {brats_dir}")
    print(f"  N      : {args.n_subjects}  seed={args.random_seed}")
    print(f"  K      : {args.K}  delta=1")
    print(f"  Repeats: {args.n_repeats}  (warmup=1 discarded)")
    print(f"  skip_cpu: {args.skip_cpu}")
    print(f"{'='*65}\n")

    subject_dirs = collect_subjects(brats_dir, args.n_subjects, args.random_seed)
    print(f"Sampled {len(subject_dirs)} subjects\n")

    rows: list[dict] = []
    processed = 0

    for subj_dir in subject_dirs:
        sid = subj_dir.name
        flair_file = find_flair_file(subj_dir)
        if flair_file is None:
            print(f"  [SKIP] {sid}: no flair file")
            continue
        row = process_subject(sid, flair_file, args.n_repeats, args.skip_cpu, K=args.K)
        if row is not None:
            rows.append(row)
            processed += 1

    if not rows:
        sys.exit("No subjects processed.")

    rows = append_summary_stats(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    data_rows = [r for r in rows if not str(r["subject_id"]).startswith("__")]
    t_dd_vals  = [float(r["t_dd_s"])  for r in data_rows if r.get("t_dd_s")]
    t_g1_vals  = [float(r["t_gpu1_s"]) for r in data_rows if r.get("t_gpu1_s")]
    dwc_pass   = sum(1 for r in data_rows if r.get("dwc_guaranteed_dd") in (True,"True"))

    print(f"\n{'='*65}")
    print(f"  DONE — {processed} subjects")
    print(f"  Output : {out}")
    if t_dd_vals:
        print(f"  DD-FP  : mean={np.mean(t_dd_vals):.2f}s  "
              f"median={np.median(t_dd_vals):.2f}s")
    if t_g1_vals:
        print(f"  GPU1   : mean={np.mean(t_g1_vals):.3f}s  "
              f"median={np.median(t_g1_vals):.3f}s")
    print(f"  DWC pass: {dwc_pass}/{processed}")
    print(f"\n  → A2 BRATS_REF_ROWS values updated:")
    if t_dd_vals:
        print(f"      t_dd_s  median = {np.median(t_dd_vals):.3f}")
    if t_g1_vals:
        print(f"      t_gpu1_s median = {np.median(t_g1_vals):.3f}")
    print(f"      note = 'BraTS2021 N={processed} median (timing_n100.csv)'")
    print(f"{'='*65}\n")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate timing CSV (K-configurable)")
    p.add_argument("--brats-dir", required=True)
    p.add_argument("--output", default="timing_n100.csv")
    p.add_argument("--n-subjects",  type=int, default=100)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--n-repeats",   type=int, default=1,
                   help="Measurement repeats per subject (default=1; warmup is always discarded)")
    p.add_argument("--skip-cpu", action="store_true",
                   help="Skip sequential CPU timing (~184s/subject)")
    p.add_argument("--K", type=int, default=16,
                   help="DD-FP subdomain count (default=16 for K=16)")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())