"""
scripts/part_a/exp_a4_scalability.py
======================================
Exp A4: Scalability Analysis

Shows that DD-FP GPU scales linearly (b ~ 1.0) with input size.

Procedure
---------
  GPU only (CuPy front_propagation_gpu)
  2D: 256^2, 512^2, 1024^2, 2048^2, 4096^2
  3D: 64^3, 128^3, 256^3  (512^3 attempted, OOM caught)
  Measurements per volume:
    - Wall-clock time: median of 5 runs (1 warmup discarded)
    - Peak GPU memory: cupy.get_default_memory_pool().used_bytes()

Analysis
--------
  Fit log(time) = a + b * log(N)  where N = total voxels
  Report b: expect b ~ 1.0 (linear scaling)
  If b > 1.2: flag as super-linear

BraTS reference point
---------------------
  Loaded from timing_n100.csv (--timing-n100-csv).
  Column used: t_gpu1_s  (GPU single-pass K=1, N subjects, median).
  Shape: 240x240x155 = 8.9M voxels.

Output
------
  results/part_a/a4_scalability.csv
  results/part_a/a4_scalability_loglog.png
    - Log-log plot: x=N (voxels), y=time
    - Fitted line overlaid, slope b annotated
    - Separate series for 2D and 3D
    - 300 DPI
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.benchmark_utils import generate_synthetic_volume, time_function

try:
    import cupy as cp
    from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu
    _CUPY_OK = True
except Exception as _e:
    _CUPY_OK = False
    print(f"[WARN] CuPy unavailable ({_e}). Exp A4 requires GPU.")


# ---------------------------------------------------------------------------
# Test sizes
# ---------------------------------------------------------------------------
SIZES_2D = [256, 512, 1024, 2048, 4096]
SIZES_3D = [64, 128, 256]      # 512^3 attempted with OOM guard
N_REPEATS = 5
WARMUP    = 1


# ---------------------------------------------------------------------------
# Load BraTS reference point from timing_n100.csv
# ---------------------------------------------------------------------------

def _load_brats_ref(timing_n100_csv: Path) -> dict:
    """
    Load the BraTS reference data point from timing_n100.csv.
    Uses the median of t_gpu1_s (GPU single-pass K=1) across all subjects.
    Raises FileNotFoundError if the file is missing.
    """
    if not timing_n100_csv.exists():
        raise FileNotFoundError(
            f"timing_n100.csv not found: {timing_n100_csv}\n"
            "Generate it with scripts/walltime/gen_timing_n100.py"
        )
    with open(timing_n100_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    gpu1_vals = [
        float(r["t_gpu1_s"]) for r in rows
        if r.get("t_gpu1_s") not in (None, "", "None")
    ]
    if not gpu1_vals:
        raise ValueError(f"No valid t_gpu1_s values in {timing_n100_csv}")

    t_gpu1_median = round(statistics.median(gpu1_vals), 3)
    print(f"  [A4] Loaded {timing_n100_csv.name}: "
          f"BraTS t_gpu1_s median={t_gpu1_median:.3f}s (N={len(gpu1_vals)})")

    return {
        "ndim": 3,
        "size_str": "240x240x155",
        "n_voxels": 240 * 240 * 155,
        "time_s": t_gpu1_median,
        "peak_memory_mb": None,
        "source": f"{timing_n100_csv.name} (BraTS2021, N={len(gpu1_vals)} median)",
    }


# ---------------------------------------------------------------------------
# GPU runner
# ---------------------------------------------------------------------------

def _ensure_3d(vol: np.ndarray) -> np.ndarray:
    """Promote 2D (H,W) to 3D (H,W,1) for the 3D-only GPU pipeline."""
    return vol[:, :, np.newaxis] if vol.ndim == 2 else vol


def _run_gpu_full(vol: np.ndarray) -> None:
    """Full pipeline including host->device and device->host transfers."""
    vol3 = _ensure_3d(vol)
    U_lo, U_hi, l_inf = build_ispan_gpu(vol3)
    u_pad = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)
    cp.cuda.Stream.null.synchronize()
    _ = cp.asnumpy(u_pad)


def _measure_peak_memory_mb(vol: np.ndarray) -> float:
    """Measure peak GPU memory during a single FP run."""
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    before = pool.used_bytes()

    vol3 = _ensure_3d(vol)
    U_lo, U_hi, l_inf = build_ispan_gpu(vol3)
    _ = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)
    cp.cuda.Stream.null.synchronize()

    peak = pool.used_bytes()
    del U_lo, U_hi, _
    pool.free_all_blocks()

    return (peak - before) / (1024 ** 2)


# ---------------------------------------------------------------------------
# Linear regression on log-log scale
# ---------------------------------------------------------------------------

def _fit_scaling_exponent(n_voxels: list[int], times: list[float]) -> tuple[float, float]:
    """
    Fit log(t) = a + b * log(N) via ordinary least squares.
    Returns (a, b) where b is the scaling exponent.
    b ~ 1.0 => linear scaling
    b > 1.2 => super-linear (flag)
    """
    x = np.log(np.array(n_voxels, dtype=float))
    y = np.log(np.array(times,    dtype=float))

    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan"), float("nan")

    x, y = x[mask], y[mask]
    A = np.column_stack([np.ones_like(x), x])
    result = np.linalg.lstsq(A, y, rcond=None)
    a, b = result[0]
    return float(a), float(b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_a4(
    out_dir: Path,
    timing_n100_csv: Path | None = None,
) -> None:
    """
    Parameters
    ----------
    out_dir         : directory for CSV and PNG outputs
    timing_n100_csv : path to timing_n100.csv (default: ROOT/timing_n100.csv)
    """
    if not _CUPY_OK:
        print("[ERROR] Exp A4 requires CuPy/GPU. Skipping.")
        return

    if timing_n100_csv is None:
        timing_n100_csv = ROOT / "timing_n100.csv"

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"\n{'='*70}")
    print("  Exp A4: Scalability (GPU only)")
    print(f"{'='*70}")
    print(f"  {'shape':<20} {'n_voxels':>12} {'time_s':>8} {'mem_MB':>8}  {'vox/s':>12}")
    print(f"  {'-'*64}")

    def _run_one(ndim: int, size_or_shape, seed: int = 42):
        if isinstance(size_or_shape, int):
            shape = (size_or_shape,) * ndim
        else:
            shape = size_or_shape

        n_voxels = int(np.prod(shape))
        shape_str = "x".join(str(s) for s in shape)
        size_val  = shape[0]

        vol = generate_synthetic_volume(shape, seed=seed)
        vol_gpu_input = vol[:, :, np.newaxis] if ndim == 2 else vol

        try:
            t = time_function(_run_gpu_full, vol_gpu_input,
                              n_repeats=N_REPEATS, warmup=WARMUP,
                              use_cuda_events=True)
            t_med = t["median_s"]
        except Exception as ex:
            oom = "OOM" if "memory" in str(ex).lower() else str(ex)[:40]
            print(f"  {shape_str:<20} {n_voxels:>12,}  skipped ({oom})")
            rows.append({
                "ndim": ndim, "size": size_val, "shape": shape_str,
                "n_voxels": n_voxels, "time_s": None,
                "peak_memory_mb": None, "voxels_per_sec": None,
                "source": "measured",
            })
            return

        try:
            peak_mb = _measure_peak_memory_mb(vol_gpu_input)
        except Exception:
            peak_mb = None

        vps = n_voxels / t_med if t_med > 0 else None

        print(f"  {shape_str:<20} {n_voxels:>12,} {t_med:>8.3f} "
              f"{peak_mb if peak_mb else 0:>7.1f}  "
              f"{vps/1e6 if vps else 0:>10.2f}M")

        rows.append({
            "ndim":            ndim,
            "size":            size_val,
            "shape":           shape_str,
            "n_voxels":        n_voxels,
            "time_s":          round(t_med, 4),
            "peak_memory_mb":  round(peak_mb, 2) if peak_mb is not None else None,
            "voxels_per_sec":  round(vps) if vps is not None else None,
            "source":          "measured",
        })

    # 2D
    print("\n  --- 2D ---")
    for s in SIZES_2D:
        _run_one(2, s)

    # 3D
    print("\n  --- 3D ---")
    for s in SIZES_3D:
        _run_one(3, s)

    # 3D 512^3 with OOM guard
    print(f"\n  [3D 512^3 attempt]")
    try:
        _run_one(3, 512)
    except (MemoryError, cp.cuda.memory.OutOfMemoryError):
        print("  skipped (OOM)")
        rows.append({
            "ndim": 3, "size": 512, "shape": "512x512x512",
            "n_voxels": 512**3, "time_s": None,
            "peak_memory_mb": None, "voxels_per_sec": None,
            "source": "skipped_OOM",
        })

    # Append BraTS reference point (from timing_n100.csv)
    brats_ref = _load_brats_ref(timing_n100_csv)
    rows.append({
        "ndim":            3,
        "size":            240,
        "shape":           brats_ref["size_str"],
        "n_voxels":        brats_ref["n_voxels"],
        "time_s":          brats_ref["time_s"],
        "peak_memory_mb":  brats_ref["peak_memory_mb"],
        "voxels_per_sec":  round(brats_ref["n_voxels"] / brats_ref["time_s"]),
        "source":          brats_ref["source"],
    })
    print(f"\n  (+ BraTS ref: {brats_ref['size_str']}, "
          f"t={brats_ref['time_s']}s from {timing_n100_csv.name})")

    # CSV
    csv_path = out_dir / "a4_scalability.csv"
    fieldnames = ["ndim", "size", "shape", "n_voxels",
                  "time_s", "peak_memory_mb", "voxels_per_sec", "source"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  CSV -> {csv_path}")

    # Fit scaling exponent per dimensionality
    print(f"\n  --- Scaling exponent b (log-log fit) ---")
    for ndim in [2, 3]:
        valid = [(r["n_voxels"], r["time_s"])
                 for r in rows
                 if r["ndim"] == ndim
                 and r["time_s"] is not None
                 and r["source"] == "measured"]
        if len(valid) < 2:
            print(f"  {ndim}D: insufficient data points")
            continue
        ns, ts = zip(*valid)
        a, b = _fit_scaling_exponent(list(ns), list(ts))
        flag = " ** SUPER-LINEAR **" if b > 1.2 else ""
        print(f"  {ndim}D: b = {b:.3f}  (a = {a:.3f})"
              f"  => {'linear' if b <= 1.2 else 'super-linear'}{flag}")

    _plot_loglog(rows, out_dir)
    print(f"{'='*70}\n")


def _plot_loglog(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib not available, skipping log-log plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    styles = {
        2: {"color": "#0072B2", "marker": "o", "label": "2D"},
        3: {"color": "#D55E00", "marker": "s", "label": "3D"},
    }

    for ndim in [2, 3]:
        measured = [(r["n_voxels"], r["time_s"])
                    for r in rows
                    if r["ndim"] == ndim
                    and r["time_s"] is not None
                    and r["source"] == "measured"]
        ref_pts  = [(r["n_voxels"], r["time_s"])
                    for r in rows
                    if r["ndim"] == ndim
                    and r["time_s"] is not None
                    and r["source"] != "measured"]

        st = styles[ndim]

        if measured:
            ns, ts = zip(*measured)
            ax.scatter(ns, ts, color=st["color"], marker=st["marker"],
                       s=60, zorder=5, label=st["label"])

            a, b = _fit_scaling_exponent(list(ns), list(ts))
            if np.isfinite(b):
                ns_arr = np.array(sorted(ns))
                ts_fit = np.exp(a) * ns_arr ** b
                ax.plot(ns_arr, ts_fit, color=st["color"],
                        linestyle="--", linewidth=1.5,
                        label=f"{ndim}D fit: $b={b:.2f}$")

        for n, t in ref_pts:
            ax.scatter([n], [t], color=st["color"], marker="*",
                       s=150, zorder=6, label=f"3D ref (BraTS)")

    all_n = [r["n_voxels"] for r in rows if r["time_s"] is not None]
    if all_n:
        n_min, n_max = min(all_n), max(all_n)
        ns_ref = np.array([n_min, n_max], dtype=float)
        valid_t = [r["time_s"] for r in rows
                   if r["time_s"] is not None and r["n_voxels"] == n_min]
        if valid_t:
            t_anchor = np.median(valid_t)
            scale = t_anchor / n_min
            ax.plot(ns_ref, scale * ns_ref, color="gray",
                    linestyle=":", linewidth=1.2, label="Ideal $b=1.0$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of voxels $N$", fontsize=12)
    ax.set_ylabel("Wall-clock time (s, log scale)", fontsize=12)
    ax.set_title("Exp A4: DD-FP GPU Scalability (log-log)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(which="both", linestyle="--", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "a4_scalability_loglog.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot -> {path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp A4: Scalability Analysis")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "part_a",
                   help="Output directory (default: ROOT/results/part_a)")
    p.add_argument("--timing-n100-csv", type=Path, default=None,
                   help="Path to timing_n100.csv  [default: ROOT/timing_n100.csv]")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_a4(
        out_dir=args.out_dir,
        timing_n100_csv=args.timing_n100_csv,
    )
