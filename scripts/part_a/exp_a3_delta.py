"""
scripts/part_a/exp_a3_delta.py
================================
Exp A3: delta (overlap) Sensitivity

Key hypothesis (Theorem 3.1 / Lemma 4):
  delta>=1: boundary violations == 0  (DWC preserved at subdomain boundaries)

IBI implementation: run_ibi_v10() from experiment_DDFP.py.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import label, generate_binary_structure

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.benchmark_utils import generate_synthetic_volume

try:
    import cupy as cp
    from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu
    _CUPY_OK = True
except Exception as e:
    _CUPY_OK = False
    print(f"[WARN] CuPy unavailable ({e}). Experiment requires GPU.")


# ---------------------------------------------------------------------------
# Helpers inlined verbatim from experiment_DDFP.py
# ---------------------------------------------------------------------------

def _split_domain_1d(total: int, K: int) -> list[tuple[int, int]]:
    base, rem = divmod(total, K)
    slices, cur = [], 0
    for i in range(K):
        sz = base + (1 if i < rem else 0)
        slices.append((cur, cur + sz)); cur += sz
    return slices


def _pin_and_seed_expanded(
    U_lo_pad, U_hi_pad, seeds_pad, assembled_g,
    sub_z_start, g_ez_start, g_ez_end, W2s, H2s,
):
    n = g_ez_end - g_ez_start
    if n <= 0: return
    l_pad_start = (g_ez_start - 2 * sub_z_start) + 1
    n_g = min(n, assembled_g.shape[2] - g_ez_start)
    n_l = min(n, U_lo_pad.shape[2] - l_pad_start)
    actual = min(n_g, n_l)
    if actual <= 0: return
    vals = assembled_g[:W2s, :H2s, g_ez_start:g_ez_start + actual]
    U_lo_pad[1:W2s+1, 1:H2s+1, l_pad_start:l_pad_start + actual] = vals
    U_hi_pad[1:W2s+1, 1:H2s+1, l_pad_start:l_pad_start + actual] = vals
    seeds_pad[1:W2s+1, 1:H2s+1, l_pad_start:l_pad_start + actual] = vals


def _assemble_boundary(u_target, u_sub, z_b, z0_ext, D2):
    """Verbatim from experiment_DDFP.py -- writes 2 cells: gz=2*z_b-1, gz=2*z_b.
    """
    local_zb = z_b - z0_ext
    ez_0 = 2 * local_zb;  ez_1 = ez_0 - 1
    gz_0 = 2 * z_b;       gz_1 = gz_0 - 1

    if 0 <= ez_0 < u_sub.shape[2] and gz_0 < D2:
        u_target[:, :, gz_0] = u_sub[:, :, ez_0]
    if 0 <= ez_1 < u_sub.shape[2] and gz_1 >= 0:
        u_target[:, :, gz_1] = u_sub[:, :, ez_1]


def run_ibi_v10(vol_u8: np.ndarray, K: int, delta: int,
                max_rounds: int = 8, l_inf: float = 0.0,
                verbose: bool = False) -> dict:
    """
    Verbatim copy of experiment_DDFP.run_ibi_v10().

    IBI convergence (Jacobi) guarantees DWC at subdomain boundaries for delta>=1.
    Boundary cells: _assemble_boundary writes gz=2*z_b-1 and gz=2*z_b from sub k.
    Face-block [gz=2*z_b, gz=2*z_b+1] is correct because in IBI rounds sub k+1
    is PINNED at gz=2*z_b (from the assembled value), so sub k+1's FP computes
    gz=2*z_b+1 consistently with the boundary (Lemma 4 + IBI pinning logic).

    D<=1 short-circuit: 2D-as-3D volumes (D=1 or D=3 from repeat) with K=1
    have no subdomain boundaries, so bdry_viol=0 by definition.
    """
    W, H, D = vol_u8.shape
    W2, H2, D2 = 2*W - 1, 2*H - 1, 2*D - 1
    W2s, H2s = W2, H2
    t0 = time.time()

    K_eff = min(K, D)
    slices_z = [(z0, z1) for z0, z1 in _split_domain_1d(D, K_eff) if z1 > z0]
    boundary_z_orig = [z1 for _, z1 in slices_z[:-1]]

    # Round 0: independent FP per subdomain
    # _assemble_boundary is called immediately after each sub's inner assembly.
    # This is the original single-pass from experiment_DDFP.py ("idempotent"
    # because later subs' inner assembly starts at gz_s=2*z0, not at gz=2*z_b).
    u_assembled = cp.zeros((W2, H2, D2), dtype=cp.float32)
    sub_meta = []

    for k, (z0, z1) in enumerate(slices_z):
        z0_ext = max(0, z0 - delta)
        z1_ext = min(D, z1 + delta)
        sub_vol = vol_u8[:, :, z0_ext:z1_ext]

        U_lo, U_hi, _ = build_ispan_gpu(sub_vol)
        u_sub_pad = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)
        u_sub = u_sub_pad[1:-1, 1:-1, 1:-1]

        inner_loc_z0 = z0 - z0_ext
        inner_loc_z1 = inner_loc_z0 + (z1 - z0)
        # k>0: start one cell past the boundary cell (gz=2*z0) so that
        # _assemble_boundary's write to gz=2*z0 is not overwritten by
        # this sub's inner assembly.  Lemma 4 requires gz=2*z0 to remain
        # the value written by the previous sub's _assemble_boundary.
        ez_s = (2 * inner_loc_z0 + 1) if k > 0 else 0
        gz_s = (2 * z0 + 1)           if k > 0 else 0
        ez_e = 2 * inner_loc_z1 - 1 if k < len(slices_z) - 1 else u_sub.shape[2]
        gz_e = min(gz_s + (ez_e - ez_s), D2)
        actual = gz_e - gz_s
        if actual > 0 and ez_s + actual <= u_sub.shape[2]:
            u_assembled[:, :, gz_s:gz_e] = u_sub[:, :, ez_s:ez_s + actual]

        if k < len(slices_z) - 1:
            _assemble_boundary(u_assembled, u_sub, z1, z0_ext, D2)

        sub_meta.append({"k": k, "z0": z0, "z1": z1,
                         "z0_ext": z0_ext, "z1_ext": z1_ext})

    # IBI rounds (Jacobi)
    R_star = 0
    for r in range(1, max_rounds + 1):
        u_prev = u_assembled.copy()
        u_new  = u_assembled.copy()

        for m in sub_meta:
            k = m["k"]; z0 = m["z0"]; z1 = m["z1"]
            z0_ext = m["z0_ext"]; z1_ext = m["z1_ext"]
            sub_vol = vol_u8[:, :, z0_ext:z1_ext]

            U_lo, U_hi, _ = build_ispan_gpu(sub_vol)
            seeds = cp.full(U_lo.shape, cp.nan, dtype=cp.float32)

            if k > 0:
                _pin_and_seed_expanded(U_lo, U_hi, seeds, u_prev,
                    z0_ext, 2*z0, 2*z0+1, W2s, H2s)
            if k < len(slices_z) - 1:
                _pin_and_seed_expanded(U_lo, U_hi, seeds, u_prev,
                    z0_ext, 2*z1, 2*z1_ext, W2s, H2s)

            u_sub_pad = front_propagation_gpu(U_lo, U_hi, l_inf,
                                               boundary_seeds=seeds, verbose=False)
            u_sub = u_sub_pad[1:-1, 1:-1, 1:-1]

            inner_loc_z0 = z0 - z0_ext
            inner_loc_z1 = inner_loc_z0 + (z1 - z0)
            # Same boundary-cell preservation fix as Round 0:
            # skip gz=2*z0 so _assemble_boundary's value survives.
            ez_s = (2 * inner_loc_z0 + 1) if k > 0 else 0
            gz_s = (2 * z0 + 1)           if k > 0 else 0
            ez_e = 2 * inner_loc_z1 - 1 if k < len(slices_z) - 1 else u_sub.shape[2]
            gz_e = min(gz_s + (ez_e - ez_s), D2)
            actual = gz_e - gz_s
            if actual > 0 and ez_s + actual <= u_sub.shape[2]:
                u_new[:, :, gz_s:gz_e] = u_sub[:, :, ez_s:ez_s + actual]

            if k < len(slices_z) - 1:
                _assemble_boundary(u_new, u_sub, z1, z0_ext, D2)

        max_change = float(cp.abs(u_new - u_prev).max().item())
        u_assembled = u_new
        R_star = r
        if verbose:
            print(f"    R={r}  max_change={max_change:.4f}")
        if max_change < 1e-3:
            break

    return {
        "u_dd": cp.asnumpy(u_assembled),
        "boundary_z_orig": boundary_z_orig,
        "R_star": R_star,
        "t_total_s": time.time() - t0,
    }

def _count_boundary_violations(u_inner: np.ndarray,
                                boundary_z_orig: list[int]) -> int:
    """Verbatim from experiment_DDFP.count_boundary_violations()."""
    D2 = u_inner.shape[2]
    total = 0
    for z_b in boundary_z_orig:
        for ez in [2*z_b - 1, 2*z_b]:
            if ez < 0 or ez + 1 >= D2: continue
            for ax in [0, 1]:
                if ax == 0:
                    v00 = u_inner[:-1,:,ez];  v11 = u_inner[1:,:,ez+1]
                    v10 = u_inner[1:,:,ez];   v01 = u_inner[:-1,:,ez+1]
                else:
                    v00 = u_inner[:,:-1,ez];  v11 = u_inner[:,1:,ez+1]
                    v10 = u_inner[:,1:,ez];   v01 = u_inner[:,:-1,ez+1]
                lo_a = np.minimum(v00,v11); hi_a = np.maximum(v00,v11)
                lo_r = np.minimum(v10,v01); hi_r = np.maximum(v10,v01)
                total += int(((hi_a < lo_r) | (hi_r < lo_a)).sum())
    return total


def _betti_numbers(u: np.ndarray, threshold: float = 0.5) -> tuple[int, int]:
    binary = (u >= threshold).astype(np.int32)
    _, beta0 = label(binary)
    if binary.ndim == 2:
        struct = generate_binary_structure(2, 1)
        _, b0 = label(binary, structure=struct)
        chi = (int(binary.sum())
               - int((binary[:-1,:] & binary[1:,:]).sum())
               - int((binary[:,:-1] & binary[:,1:]).sum())
               + int((binary[:-1,:-1] & binary[1:,:-1]
                      & binary[:-1,1:] & binary[1:,1:]).sum()))
        beta1 = max(0, b0 - chi)
    else:
        beta1 = -1
    return int(beta0), int(beta1)


# ---------------------------------------------------------------------------
# Test images
# ---------------------------------------------------------------------------
DELTAS = [1, 2, 3]  
K      = 16

TEST_IMAGES = [
    {"id": "synth_2d_s0", "vol": generate_synthetic_volume((256, 256), seed=0)},
    {"id": "synth_2d_s1", "vol": generate_synthetic_volume((256, 256), seed=1)},
    {"id": "synth_3d_64_s0",  "vol": generate_synthetic_volume((64,  64,  64),  seed=10)},
    {"id": "synth_3d_64_s1",  "vol": generate_synthetic_volume((64,  64,  64),  seed=11)}
]


def run_a3(out_dir: Path) -> None:
    if not _CUPY_OK:
        print("[ERROR] Exp A3 requires CuPy/GPU. Skipping.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    print(f"\n{'='*70}")
    print("  Exp A3: delta Sensitivity  (K=16, IBI max_rounds=8, tol=1e-3)")
    print(f"{'='*70}")
    print(f"  {'image_id':<22} {'delta':>5} {'bdry_viol':>10} {'time_s':>8} {'R*':>4} {'beta0':>6}")
    print(f"  {'-'*60}")

    for img in TEST_IMAGES:
        iid   = img["id"]
        vol   = img["vol"]
        is_3d = vol.ndim == 3

        for d in DELTAS:
            if not is_3d:
                vol3  = np.repeat(vol[:, :, np.newaxis], 3, axis=2)
                K_run = 1
            else:
                vol3  = vol
                K_run = K

            try:
                result    = run_ibi_v10(vol3, K=K_run, delta=d,
                                        max_rounds=8, verbose=False)
                u_out     = result["u_dd"]
                bdry_viol = _count_boundary_violations(u_out, result["boundary_z_orig"])
                u_2d      = u_out[:, :, 1] if not is_3d else u_out
                beta0, _  = _betti_numbers(u_2d)
                elapsed   = result["t_total_s"]
                R_star    = result["R_star"]
            except Exception as ex:
                print(f"  {iid:<22} {d:>5}  ERROR: {ex}")
                continue

            print(f"  {iid:<22} {d:>5} {bdry_viol:>10,} "
                  f"{elapsed:>8.3f} {R_star:>4}  {beta0:>6}")

            rows.append({
                "image_id":        iid,
                "ndim":            3 if is_3d else 2,
                "delta":           d,
                "K":               K_run,
                "R_star":          R_star,
                "bdry_violations": bdry_viol,
                "time_s":          round(elapsed, 4),
                "beta0":           beta0,
                "dwc_ok":          bdry_viol == 0,
            })

    csv_path = out_dir / "a3_delta_sweep.csv"
    fieldnames = ["image_id","ndim","delta","K","R_star",
                  "bdry_violations","time_s","beta0","dwc_ok"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(rows)
    print(f"\n  CSV -> {csv_path}")

    _plot_violations(rows, out_dir)
    _plot_time(rows, out_dir)
    print(f"{'='*70}\n")


def _plot_violations(rows, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax_i, (ndim, title) in enumerate([(2,"2D"),(3,"3D")]):
        ax = axes[ax_i]
        d_rows = {d: [r for r in rows if r["delta"]==d and r["ndim"]==ndim]
                  for d in DELTAS}
        means = [np.mean([r["bdry_violations"] for r in d_rows[d]]) if d_rows[d] else 0
                 for d in DELTAS]
        stds  = [np.std([r["bdry_violations"] for r in d_rows[d]], ddof=1)
                 if len(d_rows[d]) > 1 else 0 for d in DELTAS]
        ax.bar(DELTAS, means, yerr=stds, capsize=5, color="#0072B2", alpha=0.85, width=0.5)
        ax.set_xlabel("delta"); ax.set_ylabel("Mean boundary violations")
        ax.set_title(f"Exp A3: Boundary DWC Violations ({title})")
        ax.set_xticks(DELTAS); ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()
    path = out_dir / "a3_delta_violations.png"
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Plot -> {path}")


def _plot_time(rows, out_dir):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax_i, (ndim, title) in enumerate([(2,"2D"),(3,"3D")]):
        ax = axes[ax_i]
        r_by_d = {d: [r["time_s"] for r in rows if r["delta"]==d and r["ndim"]==ndim]
                  for d in DELTAS}
        means = [np.mean(r_by_d[d]) if r_by_d[d] else 0 for d in DELTAS]
        stds  = [np.std(r_by_d[d], ddof=1) if len(r_by_d[d]) > 1 else 0
                 for d in DELTAS]
        ax.plot(DELTAS, means, marker="o", color="#009E73", lw=2)
        ax.fill_between(DELTAS, [m-s for m,s in zip(means,stds)],
                                 [m+s for m,s in zip(means,stds)],
                        alpha=0.25, color="#009E73")
        ax.set_xlabel("delta"); ax.set_ylabel("Mean wall-clock time (s)")
        ax.set_title(f"Exp A3: Runtime vs delta ({title})")
        ax.set_xticks(DELTAS); ax.grid(ls="--", alpha=0.4)
    plt.tight_layout()
    path = out_dir / "a3_delta_time.png"
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    print(f"  Plot -> {path}")

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exp A3: Synthetic Data delta sensitivity")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results" / "part_a",
                   help="Output directory (default: ROOT/results/part_a)")
    return p.parse_args()
    
if __name__ == "__main__":
    args = _parse_args()
    run_a3(args.out_dir)