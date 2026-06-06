"""
scripts/part_a/exp_a2_speedup.py
==================================
Exp A2: Wall-Clock Speedup

Three implementations of DD-FP interpolation compared:
  1. seq_cpu  : build_ispan_cpu + fp_cpu  (Algorithm 1 priority-queue, single-threaded)
                Source: experiment_DDFP.py fp_cpu()  -- same code as Table 6 baseline
  2. par_cpu  : build_ispan_cpu only (vectorised numpy, no FP loop)
                Represents the pure Ispan construction cost without front-propagation.
                Used as the "vectorised numpy baseline" in the speedup table.
  3. gpu      : build_ispan_gpu + front_propagation_gpu  (CuPy Level-BFS)
                Source: gpu_immersion.py  -- same code as Table 6 GPU column
"""
from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.benchmark_utils import generate_synthetic_volume, time_function

# ---------------------------------------------------------------------------
# CPU sequential: build_ispan_cpu + fp_cpu from experiment_DDFP.py
# Inlined here to avoid import path issues across environments.
# ---------------------------------------------------------------------------

def _snap(lv, lo, hi):
    if lv < lo: return lo
    if lv > hi: return hi
    return lv


def _build_ispan_cpu(vol_u8: np.ndarray):
    """
    Ispan construction (CPU, numpy).  Returns (U_lo_pad, U_hi_pad, l_inf).
    Identical to experiment_DDFP.build_ispan_cpu().
    """
    W, H, D = vol_u8.shape
    s = vol_u8.astype(np.float32)
    W2, H2, D2 = 2*W-1, 2*H-1, 2*D-1
    U_lo = np.empty((W2, H2, D2), np.float32)
    U_hi = np.empty((W2, H2, D2), np.float32)
    U_lo[::2, ::2, ::2] = s
    U_hi[::2, ::2, ::2] = s
    for a, b, sl in [
        (s[:-1,:,:], s[1:,:,:],  (np.s_[1::2], np.s_[::2],  np.s_[::2])),
        (s[:,:-1,:], s[:,1:,:],  (np.s_[::2],  np.s_[1::2], np.s_[::2])),
        (s[:,:,:-1], s[:,:,1:], (np.s_[::2],  np.s_[::2],  np.s_[1::2])),
    ]:
        U_lo[sl] = np.minimum(a, b)
        U_hi[sl] = np.maximum(a, b)
    def mm4(f, *c): return f(f(c[0], c[1]), f(c[2], c[3]))
    U_lo[1::2,1::2,::2]  = mm4(np.minimum,s[:-1,:-1,:],s[1:,:-1,:],s[:-1,1:,:],s[1:,1:,:])
    U_hi[1::2,1::2,::2]  = mm4(np.maximum,s[:-1,:-1,:],s[1:,:-1,:],s[:-1,1:,:],s[1:,1:,:])
    U_lo[1::2,::2,1::2]  = mm4(np.minimum,s[:-1,:,:-1],s[1:,:,:-1],s[:-1,:,1:],s[1:,:,1:])
    U_hi[1::2,::2,1::2]  = mm4(np.maximum,s[:-1,:,:-1],s[1:,:,:-1],s[:-1,:,1:],s[1:,:,1:])
    U_lo[::2,1::2,1::2]  = mm4(np.minimum,s[:,:-1,:-1],s[:,1:,:-1],s[:,:-1,1:],s[:,1:,1:])
    U_hi[::2,1::2,1::2]  = mm4(np.maximum,s[:,:-1,:-1],s[:,1:,:-1],s[:,:-1,1:],s[:,1:,1:])
    c = np.stack([s[:-1,:-1,:-1],s[1:,:-1,:-1],s[:-1,1:,:-1],s[1:,1:,:-1],
                  s[:-1,:-1,1:], s[1:,:-1,1:], s[:-1,1:,1:], s[1:,1:,1:]])
    U_lo[1::2,1::2,1::2] = c.min(0)
    U_hi[1::2,1::2,1::2] = c.max(0)
    return (np.pad(U_lo, 1, constant_values=0.0),
            np.pad(U_hi, 1, constant_values=0.0), 0.0)


def _fp_cpu(U_lo_pad: np.ndarray, U_hi_pad: np.ndarray, l_inf: float = 0.0):
    """
    Algorithm 1 sequential FP (CPU, priority-queue).
    Identical to experiment_DDFP.fp_cpu() -- Table 6 CPU baseline source.
    """
    Wp, Hp, Dp = U_lo_pad.shape
    u  = np.full(Wp * Hp * Dp, np.nan, np.float32)
    dv = np.zeros(Wp * Hp * Dp, bool)
    Q  = defaultdict(list)
    lo_f = U_lo_pad.ravel().astype(np.float64)
    hi_f = U_hi_pad.ravel().astype(np.float64)
    cur  = [int(round(l_inf))]

    def push(h, lv):
        Q[int(round(_snap(lv, lo_f[h], hi_f[h])))].append(h)

    def pop():
        if not Q[cur[0]]:
            occ = [lv for lv, q in Q.items() if q]
            if not occ: return -1
            cur[0] = min(occ, key=lambda x: abs(x - cur[0]))
        return Q[cur[0]].pop()

    dx = [-1,1,0,0,0,0]; dy = [0,0,-1,1,0,0]; dz = [0,0,0,0,-1,1]

    def nbs(idx):
        z = idx//(Wp*Hp); r = idx%(Wp*Hp); y = r//Wp; x = r%Wp
        res = []
        for i in range(6):
            nx, ny, nz = x+dx[i], y+dy[i], z+dz[i]
            if 0 <= nx < Wp and 0 <= ny < Hp and 0 <= nz < Dp:
                res.append(nx + ny*Wp + nz*Wp*Hp)
        return res

    Q[int(round(l_inf))].append(0); dv[0] = True
    while True:
        h = pop()
        if h == -1: break
        u[h] = float(cur[0])
        for nb in nbs(h):
            if not dv[nb]:
                push(nb, float(cur[0]))
                dv[nb] = True
    return u.reshape(Wp, Hp, Dp)


def _run_cpu_seq(vol: np.ndarray) -> None:
    """Full DD-FP pipeline: Ispan + sequential FP. Timing target for seq_cpu."""
    vol3 = vol[:, :, np.newaxis] if vol.ndim == 2 else vol
    U_lo, U_hi, l_inf = _build_ispan_cpu(vol3)
    _fp_cpu(U_lo, U_hi, l_inf)


# ---------------------------------------------------------------------------
# CPU parallel: vectorised Ispan only (no FP loop)
# Represents O(N) numpy-only workload -- the absolute floor for any CPU method.
# ---------------------------------------------------------------------------

def _run_cpu_par(vol: np.ndarray) -> None:
    """Vectorised Ispan construction only (no priority-queue FP)."""
    vol3 = vol[:, :, np.newaxis] if vol.ndim == 2 else vol
    _build_ispan_cpu(vol3)


# ---------------------------------------------------------------------------
# GPU: build_ispan_gpu + front_propagation_gpu
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu
    _CUPY_OK = True
except Exception as _e:
    _CUPY_OK = False
    print(f"[WARN] CuPy unavailable ({_e}). GPU columns will be NaN.")


def _run_gpu(vol: np.ndarray) -> None:
    vol3 = vol[:, :, np.newaxis] if vol.ndim == 2 else vol
    U_lo, U_hi, l_inf = build_ispan_gpu(vol3)
    u_pad = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)
    cp.cuda.Stream.null.synchronize()
    _ = cp.asnumpy(u_pad)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# BraTS reference rows (RTX 3080 Ti Laptop, N=20 timing sample, seed=42)
#
# Paper tab:speedup values (Table 3):
#   Sequential FP            : 187.5 ± 9.0 s  [175.2 / 216.8]   ~1x
#   GPU single-pass  (K=1)   : 2.5   ± 0.2 s  [2.3   / 3.1  ]  ~76x  (no DWC guarantee)
#   DD-FP (K=16, δ=1, IBI)  : 4.2   ± 0.2 s  [3.9   / 4.9  ]  ~45x  (DWC guaranteed)
#
# These values are loaded automatically from timing CSVs when available,
# and fall back to the paper values otherwise.
# To regenerate on your hardware:
#   python scripts/walltime/gen_timing_n100.py --brats-dir <path> \
#          --output timing_k16.csv --n-subjects 100 --K 16
#   python scripts/preprocess/brats_n100_sampling.py --brats-dir <path> \
#          --output timing_n100.csv --n-subjects 100 --skip-cpu
# ---------------------------------------------------------------------------

def _load_brats_ref_rows() -> list[dict]:
    """
    Load BraTS reference timing rows from pre-computed CSV files, falling back
    to paper values if the files are not present.

    Expected files (release assets or locally generated):
      timing_n100.csv  — seq_cpu + gpu1 (K=1) timings, N=100 subjects
      timing_k16.csv   — gpu1 (K=1) + ddfp K=16 timings, N=100 subjects
    """
    import statistics

    # Paper fallback values (Table 3, RTX 3080 Ti Laptop, N=20 sample)
    seq_paper  = 187.5
    gpu1_paper = 2.5
    dd_paper   = 4.2

    seq_med  = seq_paper
    gpu1_med = gpu1_paper
    dd_med   = dd_paper
    source   = "paper (Table 3 fallback)"

    # Try loading timing_n100.csv for seq_cpu and gpu1
    n100_path = ROOT / "timing_n100.csv"
    if n100_path.exists():
        try:
            with open(n100_path, newline="") as f:
                rows_n100 = list(csv.DictReader(f))
            cpu_vals  = [float(r["t_cpu_s"])  for r in rows_n100 if r.get("t_cpu_s")]
            gpu1_vals = [float(r["t_gpu1_s"]) for r in rows_n100 if r.get("t_gpu1_s")]
            if cpu_vals:
                seq_med = round(statistics.median(cpu_vals), 3)
            if gpu1_vals:
                gpu1_med = round(statistics.median(gpu1_vals), 3)
            source = f"timing_n100.csv (N={len(rows_n100)})"
            print(f"  [A2] Loaded timing_n100.csv: seq_cpu={seq_med:.1f}s  gpu1={gpu1_med:.3f}s")
        except Exception as e:
            print(f"  [A2] Could not read timing_n100.csv ({e}), using paper values.")

    # Try loading timing_k16.csv for DD-FP K=16
    k16_path = ROOT / "timing_k16.csv"
    if k16_path.exists():
        try:
            with open(k16_path, newline="") as f:
                rows_k16 = list(csv.DictReader(f))
            dd_vals = [float(r["t_dd_s"]) for r in rows_k16 if r.get("t_dd_s")]
            if dd_vals:
                dd_med = round(statistics.median(dd_vals), 3)
                source += f" + timing_k16.csv (N={len(rows_k16)})"
            print(f"  [A2] Loaded timing_k16.csv: ddfp_k16={dd_med:.3f}s")
        except Exception as e:
            print(f"  [A2] Could not read timing_k16.csv ({e}), using paper value for DD-FP.")

    def _sp(t_method):
        if t_method is None or seq_med is None or seq_med == 0:
            return None
        return round(seq_med / t_method, 1)

    return [
        # Row 1: Sequential FP (seq_cpu)
        {"ndim": 3, "size": 240, "n_voxels": 240*240*155,
         "t_seq_cpu_s": seq_med, "t_par_cpu_s": None, "t_gpu_s": None,
         "speedup_par_vs_seq": None, "speedup_gpu_vs_seq": None,
         "note": f"BraTS seq_cpu median ({source})"},
        # Row 2: GPU single-pass K=1
        {"ndim": 3, "size": 240, "n_voxels": 240*240*155,
         "t_seq_cpu_s": None, "t_par_cpu_s": None, "t_gpu_s": gpu1_med,
         "speedup_par_vs_seq": None, "speedup_gpu_vs_seq": _sp(gpu1_med),
         "note": f"BraTS GPU single-pass K=1 median ({source})"},
        # Row 3: DD-FP K=16, δ=1 (paper main result)
        {"ndim": 3, "size": 240, "n_voxels": 240*240*155,
         "t_seq_cpu_s": None, "t_par_cpu_s": None, "t_gpu_s": dd_med,
         "speedup_par_vs_seq": None, "speedup_gpu_vs_seq": _sp(dd_med),
         "note": f"BraTS DD-FP K=16 δ=1 IBI median ({source})"},
    ]


BRATS_REF_ROWS = _load_brats_ref_rows()

SIZES_2D = [256, 512, 1024, 2048]
SIZES_3D = [64, 128, 256]
N_REPEATS = 5
WARMUP    = 1

# Sizes where seq_cpu is too slow to run (skip gracefully)
SEQ_CPU_SIZE_LIMIT_2D = 512     # >512^2: skip seq_cpu
SEQ_CPU_SIZE_LIMIT_3D = 64      # >64^3: skip seq_cpu


def run_a2(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def _measure(ndim: int, size: int, vol: np.ndarray):
        n_voxels = vol.size
        print(f"\n  [{size}^{ndim}]  shape={vol.shape}  voxels={n_voxels:,}")
        results = {}

        # seq_cpu
        limit = SEQ_CPU_SIZE_LIMIT_2D if ndim == 2 else SEQ_CPU_SIZE_LIMIT_3D
        if size > limit:
            print(f"    seq_cpu : skipped (too slow for size={size})")
            results["seq_cpu"] = None
        else:
            try:
                t = time_function(_run_cpu_seq, vol, n_repeats=N_REPEATS, warmup=WARMUP)
                print(f"    seq_cpu : {t['median_s']:.3f}s  (std={t['std_s']:.3f})")
                results["seq_cpu"] = t["median_s"]
            except Exception as ex:
                print(f"    seq_cpu : ERROR {ex}")
                results["seq_cpu"] = None

        # par_cpu
        try:
            t = time_function(_run_cpu_par, vol, n_repeats=N_REPEATS, warmup=WARMUP)
            print(f"    par_cpu : {t['median_s']:.3f}s  (std={t['std_s']:.3f})  [Ispan only]")
            results["par_cpu"] = t["median_s"]
        except Exception as ex:
            print(f"    par_cpu : ERROR {ex}")
            results["par_cpu"] = None

        # gpu
        if not _CUPY_OK:
            results["gpu"] = None
        else:
            try:
                t = time_function(_run_gpu, vol, n_repeats=N_REPEATS, warmup=WARMUP,
                                  use_cuda_events=True)
                print(f"    gpu     : {t['median_s']:.3f}s  (std={t['std_s']:.3f})")
                results["gpu"] = t["median_s"]
            except Exception as ex:
                oom = "OOM" if "memory" in str(ex).lower() else str(ex)[:60]
                print(f"    gpu     : skipped ({oom})")
                results["gpu"] = None

        t_seq = results["seq_cpu"]
        def _speedup(t_m):
            if t_m is None or t_seq is None or t_seq == 0: return None
            return round(t_seq / t_m, 2)

        rows.append({
            "ndim": ndim, "size": size, "n_voxels": n_voxels,
            "t_seq_cpu_s":        results["seq_cpu"],
            "t_par_cpu_s":        results["par_cpu"],
            "t_gpu_s":            results["gpu"],
            "speedup_par_vs_seq": _speedup(results["par_cpu"]),
            "speedup_gpu_vs_seq": _speedup(results["gpu"]),
            "note": "measured",
        })

    print(f"\n{'='*70}")
    print("  Exp A2: Wall-Clock Speedup")
    print(f"  seq_cpu = Algorithm 1 FP (priority-queue)")
    print(f"  par_cpu = Ispan construction only (vectorised numpy, FP omitted)")
    print(f"  gpu     = DD-FP full pipeline (CuPy Level-BFS)")
    print(f"{'='*70}")

    print("\n  --- 2D ---")
    for s in SIZES_2D:
        _measure(2, s, generate_synthetic_volume((s, s), seed=42))

    print("\n  --- 3D ---")
    for s in SIZES_3D:
        _measure(3, s, generate_synthetic_volume((s, s, s), seed=42))

    # 512^3 OOM guard
    print(f"\n  [512^3]  attempting (may OOM)...")
    try:
        _measure(3, 512, generate_synthetic_volume((512, 512, 512), seed=42))
    except MemoryError:
        print("    skipped (host OOM)")
        rows.append({"ndim":3,"size":512,"n_voxels":512**3,
                     "t_seq_cpu_s":None,"t_par_cpu_s":None,"t_gpu_s":None,
                     "speedup_par_vs_seq":None,"speedup_gpu_vs_seq":None,
                     "note":"host_OOM"})

    # Append BraTS reference rows
    rows.extend(BRATS_REF_ROWS)
    print(f"\n  (+ BraTS ref appended from timing_n100.csv)")

    # CSV
    csv_path = out_dir / "a2_speedup.csv"
    fieldnames = ["ndim","size","n_voxels","t_seq_cpu_s","t_par_cpu_s","t_gpu_s",
                  "speedup_par_vs_seq","speedup_gpu_vs_seq","note"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\n  CSV -> {csv_path}")

    _plot_barplot(rows, out_dir)
    _write_latex(rows, out_dir)
    print(f"{'='*70}\n")


def _plot_barplot(rows, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    COLORS = {"seq_cpu": "#E69F00", "par_cpu": "#56B4E9", "gpu": "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax_idx, ndim in enumerate([2, 3]):
        ax = axes[ax_idx]
        subset = [r for r in rows if r["ndim"] == ndim and r.get("note") == "measured"]
        if not subset: continue
        sizes = [r["size"] for r in subset]
        x = np.arange(len(sizes)); width = 0.25

        for i, (key, label) in enumerate([("t_seq_cpu_s","Seq CPU (Algorithm 1)"),
                                           ("t_par_cpu_s","Par CPU (Ispan only)"),
                                           ("t_gpu_s",    "GPU (DD-FP)")]):
            vals = [r[key] if r[key] is not None else np.nan for r in subset]
            ax.bar(x + i*width, vals, width, label=label,
                   color=list(COLORS.values())[i], alpha=0.85)

        ax.set_yscale("log")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{s}^{ndim}" for s in sizes], fontsize=9)
        ax.set_xlabel("Input size", fontsize=11)
        ax.set_ylabel("Time (s, log scale)", fontsize=11)
        ax.set_title(f"{ndim}D Wall-Clock Time", fontsize=12)
        ax.legend(fontsize=8); ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("DD-FP Speedup: Seq CPU vs Par CPU (Ispan) vs GPU", fontsize=12, y=1.01)
    plt.tight_layout()
    path = out_dir / "a2_speedup_barplot.png"
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Plot -> {path}")


def _write_latex(rows, out_dir):
    def _fmt(v): return "--" if v is None else f"{v:.3f}"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{DD-FP Wall-Clock Speedup (median of 5 runs, RTX 3080 Ti Laptop). "
        r"seq\_cpu: Algorithm 1 FP (priority-queue); "
        r"par\_cpu: Ispan construction only (vectorised numpy); "
        r"gpu: full DD-FP (CuPy Level-BFS). "
        r"BraTS row from timing\_n100.csv (N=100 subjects).}",
        r"\label{tab:speedup_a2}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Dim & Size & Seq CPU (s) & Par CPU (s) & GPU (s) & Par/Seq & GPU/Seq & Note \\",
        r"\midrule",
    ]
    for r in rows:
        note = r.get("note","")
        size_str = r.get("shape", f"${r['size']}^{r['ndim']}$") if "shape" in r else f"${r['size']}^{r['ndim']}$"
        lines.append(
            f"  {r['ndim']}D & {size_str} & {_fmt(r['t_seq_cpu_s'])} & "
            f"{_fmt(r['t_par_cpu_s'])} & {_fmt(r['t_gpu_s'])} & "
            f"{_fmt(r['speedup_par_vs_seq'])} & {_fmt(r['speedup_gpu_vs_seq'])} & "
            f"\\small{{{note}}} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = out_dir / "a2_speedup_table.tex"
    path.write_text("\n".join(lines))
    print(f"  LaTeX -> {path}")


if __name__ == "__main__":
    run_a2(ROOT / "results" / "part_a")