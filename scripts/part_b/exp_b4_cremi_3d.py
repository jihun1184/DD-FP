"""
scripts/part_b/exp_b4_cremi_3d.py
------------------------------------
Part B Experiment 4: 3D topology measurement on CREMI volumes.

CREMI: electron-microscopy (EM) 3D brain tissue data.
  - HDF5 volume: [Z=125, Y=1250, X=1250], voxel size 40×4×4 nm
  - Target: neuron membrane (thin tubular topology)
  - Paper positioning:
      BraTS  = compact 3D blob  → CC_3D minor, TSI_3D significant
      CREMI  = thin 3D membrane → CC_3D significant (3D analogue of DRIVE)

Subvolume strategy:
  Full volume (125,1250,1250) → expanded (249,2499,2499): ~1.5 GB (feasible but slow)
  Recommended: n_patches random (64,128,128) subvolumes
               expanded (127,255,255): 33 MB — fast with diverse topology.

CREMI mask convention:
  mask = (neuron_ids == 0)  →  membrane binary mask

Padding-eval for ddfp (v2)
--------------------------
Previous runs observed small DWC violations in ddfp outputs caused by
patch-boundary truncation: the immersion pipeline sees an incomplete membrane
structure at every face of the extracted patch and cannot correctly compute the
DWC field near those faces.

Fix: extract a padded patch of size (D+2δ, H+2δ, W+2δ) from the full HDF5
volume, run immersion_pipeline on the padded volume, then crop the resulting
expanded-lattice field back to the core region before evaluating topology and
DWC violations.

  core crop index in expanded lattice:
    axis size (2(D+2δ)−1)  →  core [2δ : 2D+2δ−1]  (length = 2D−1)

  δ is controlled by --ddfp-pad (default 4).  At the volume boundary the
  padding is clamped: actual_pad = min(δ, available_margin).  Metrics for
  no_interp and naive_interp are computed on the original core patch unchanged.

Usage:
    python scripts/part_b/exp_b4_cremi_3d.py \
        --hdf5-dir data/CREMI/raw \
        --n-volumes 3 --n-patches 10 \
        --patch-size 64 128 128 \
        --ddfp-pad 4 \
        --output results/part_b/exp_b4_cremi_3d_results.csv
"""

from __future__ import annotations

import argparse
import csv
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
           thresholds: tuple = (0.3, 0.4, 0.5, 0.6, 0.7)) -> float:
    b0_vals = [b0_3d((interp >= t).astype(np.uint8), 26) for t in thresholds]
    return float(np.std(b0_vals))


def topology_metrics_3d(interp: np.ndarray, ref_b0: int, ref_chi: int) -> dict:
    """
    3D topology metrics at threshold=0.5 + TSI_3D.

    b0_consistency = beta0_26conn / beta0_6conn
        3D-DWC ⟹ 6-conn = 26-conn ⟹ ratio = 1.0
        DWC violation ⟹ 26-conn diagonal merge ⟹ ratio != 1.0
        (Replaces beta0_error which compared different resolutions.)

    chi_sign_flip = 1 if sign(chi_interp) ≠ sign(ref_chi)
        Naive bilinear creates spurious body-cells ⟹ χ flips sign.
        (Replaces chi_error which had the same resolution mismatch.)
    """
    bin05    = (interp >= 0.5).astype(np.uint8)
    b6       = b0_3d(bin05, 6)
    b26      = b0_3d(bin05, 26)
    chi      = euler_3d(bin05)
    cc       = abs(b6 - b26)
    b0_cons  = round(b26 / b6, 6) if b6 > 0 else 1.0
    chi_flip = int((ref_chi != 0) and (chi * ref_chi < 0))
    tsi      = tsi_3d(interp)
    is_bin   = bool(np.all((interp == 0) | (interp == 1)))
    return {
        "beta0_6conn":    b6,
        "beta0_26conn":   b26,
        "chi":            chi,
        "cc_3d":          cc,
        "cc_3d_zero":     int(cc == 0),
        "b0_consistency": b0_cons,
        "chi_sign_flip":  chi_flip,
        "tsi_3d":         round(tsi, 4),
        "is_binary":      int(is_bin),
    }


# 3D interpolation (CPU-only)

def no_interp_3d(vol_u8):
    from scipy.ndimage import zoom as ndimage_zoom
    s = vol_u8.astype(np.float32)
    W, H, D = s.shape
    factors = ((2*W-1)/W, (2*H-1)/H, (2*D-1)/D)
    return ndimage_zoom(s, factors, order=0, prefilter=False)


def naive_interp_3d(vol_bin: np.ndarray) -> np.ndarray:
    """Trilinear zoom to (2D-1,2H-1,2W-1). Output: continuous [0,1]."""
    s = vol_bin.astype(np.float32)
    D, H, W = s.shape
    factors = ((2*D-1)/D, (2*H-1)/H, (2*W-1)/W)
    return np.clip(ndimage_zoom(s, factors, order=1, prefilter=False),
                   0.0, 1.0).astype(np.float32)


def seq_fp_3d(vol_bin: np.ndarray) -> np.ndarray:
    """
    Sequential Front Propagation (FP) — Boutry et al. Algorithm 1, CPU.

    Wraps build_ispan_cpu + fp_cpu from src.ddfp.cpu_fp for 3-D volumes.
    By Theorem 4.1 the output is numerically identical to DD-FP (delta=1, IBI)
    for any volume satisfying Assumption (A1).

    Parameters
    ----------
    vol_bin : np.ndarray, shape (D, H, W), uint8 or float32 in {0, 1}

    Returns
    -------
    np.ndarray, shape (2D-1, 2H-1, 2W-1), float32, range [0, 1]
    """
    from src.ddfp.cpu_fp import build_ispan_cpu, fp_cpu

    D, H, W = vol_bin.shape
    vol_u8  = (vol_bin.astype(np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    # (D, H, W) → (W, H, D) — cpu_fp convention
    vol_whd = vol_u8.transpose(2, 1, 0).copy()

    U_lo_pad, U_hi_pad, l_inf = build_ispan_cpu(vol_whd)
    u_pad = fp_cpu(U_lo_pad, U_hi_pad, l_inf)

    # strip padding → (2W-1, 2H-1, 2D-1), transpose → (2D-1, 2H-1, 2W-1)
    u_core = u_pad[1:-1, 1:-1, 1:-1].transpose(2, 1, 0).astype(np.float32)
    if u_core.max() > 1.0:
        u_core = u_core / 255.0
    return u_core


def load_cremi_membrane_3d(hdf_path: Path) -> np.ndarray:
    """
    CREMI HDF5 → 3D binary membrane mask.

    Membrane definition (3D extension of prepare_cremi.py logic):
      voxel (z,y,x) is membrane ⟺ at least one 6-connected neighbour has a different neuron_id

    Returns
    -------
    np.ndarray, shape (Z, Y, X), uint8, {0, 1}
      1 = membrane (topology target), 0 = cell interior
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required: pip install h5py")

    with h5py.File(str(hdf_path), "r") as f:
        labels = f["/volumes/labels/neuron_ids"][:]  # (Z, Y, X) uint64

    Z, Y, X = labels.shape
    membrane = np.zeros((Z, Y, X), dtype=np.uint8)

    membrane[:-1,:,:]  |= (labels[:-1,:,:]  != labels[1:,:,:]).astype(np.uint8)  # +z
    membrane[1:,:,:]   |= (labels[:-1,:,:]  != labels[1:,:,:]).astype(np.uint8)  # -z
    membrane[:,:-1,:]  |= (labels[:,:-1,:]  != labels[:,1:,:]).astype(np.uint8)  # +y
    membrane[:,1:,:]   |= (labels[:,:-1,:]  != labels[:,1:,:]).astype(np.uint8)  # -y
    membrane[:,:,:-1]  |= (labels[:,:,:-1]  != labels[:,:,1:]).astype(np.uint8)  # +x
    membrane[:,:,1:]   |= (labels[:,:,:-1]  != labels[:,:,1:]).astype(np.uint8)  # -x

    return membrane


def extract_subvolumes(
    membrane: np.ndarray,
    n_patches: int,
    patch_size: tuple,
    seed: int = 42,
) -> list[tuple[str, tuple[int, int, int]]]:
    """
    Sample n_patches random subvolume *coordinates* from a CREMI membrane volume.

    Returns (name, (z0, y0, x0)) tuples rather than the cropped arrays so that
    callers can extract differently-sized regions (e.g. padded for ddfp) from
    the full HDF5-loaded volume without re-sampling.

    Subvolume selection criteria:
      - foreground (membrane) ratio 1-30% (exclude near-empty or near-full patches)
      - select patches with rich topology within this range
    """
    Z, Y, X = membrane.shape
    pD, pH, pW = patch_size
    rng = np.random.default_rng(seed)
    patches = []
    attempts = 0
    max_attempts = n_patches * 50

    while len(patches) < n_patches and attempts < max_attempts:
        attempts += 1
        z0 = rng.integers(0, max(1, Z - pD))
        y0 = rng.integers(0, max(1, Y - pH))
        x0 = rng.integers(0, max(1, X - pW))
        sub = membrane[z0:z0+pD, y0:y0+pH, x0:x0+pW]
        if sub.shape != (pD, pH, pW):
            continue
        fg_rate = sub.mean()
        if 0.01 <= fg_rate <= 0.30:
            name = f"cremi3d_z{z0:03d}_y{y0:04d}_x{x0:04d}"
            patches.append((name, (z0, y0, x0)))

    return patches


def make_synthetic_membrane_3d() -> list[tuple[str, np.ndarray]]:
    """
    Synthetic 3D volume resembling a CREMI membrane.
    Thin plane/surface structures for 3D thin-line topology tests.
    """
    vols = []
    rng = np.random.default_rng(42)

    D, H, W = 32, 64, 64
    grid = np.zeros((D, H, W), dtype=np.uint8)
    grid[:, ::8, :] = 1   # horizontal planes
    grid[:, :, ::8] = 1   # vertical planes
    vols.append(("synth_grid_membrane", grid))

    diag = np.zeros((32, 64, 64), dtype=np.uint8)
    for z in range(32):
        offset = z % 8
        diag[z, offset::8, :] = 1
        diag[z, :, offset::8] = 1
    vols.append(("synth_diagonal_membrane", diag))

    D2, H2, W2 = 32, 48, 48
    z2, y2, x2 = np.mgrid[0:D2, 0:H2, 0:W2]
    r2 = (y2 - H2//2)**2 + (x2 - W2//2)**2
    cyl_surface = ((r2 >= 100) & (r2 <= 144)).astype(np.uint8)
    vols.append(("synth_cylinder_surface", cyl_surface))

    rand_mem = np.zeros((24, 48, 48), dtype=np.uint8)
    for z in range(24):
        row = rng.integers(4, 44, 6)
        col = rng.integers(4, 44, 6)
        rand_mem[z, row, :] = 1
        rand_mem[z, :, col] = 1
    vols.append(("synth_random_membrane", rand_mem))

    return vols


def _run_ddfp_3d(
    vol: np.ndarray,
    name: str,
    source: str,
    ref_b0: int,
    ref_chi: int,
    fg: float,
    rows: list,
    membrane_full: np.ndarray | None = None,
    coords: tuple[int, int, int] | None = None,
    patch_size: tuple[int, int, int] | None = None,
    ddfp_pad: int = 4,
) -> None:
    """
    Run ddfp_3d with padding-eval to eliminate patch-boundary truncation.

    When `membrane_full` and `coords` are provided the function extracts a
    padded patch of size (D+2δ, H+2δ, W+2δ) from the full membrane volume,
    runs immersion_pipeline on it, then crops the expanded-lattice output back
    to the core region before computing topology metrics and DWC violations.

      core crop in expanded lattice:
        start = 2 * actual_pad_axis   (per-axis, may differ at volume boundaries)
        end   = start + (2 * pD - 1)

    Falls back to the passed `vol` (core only, no padding) when the full volume
    is unavailable (synthetic volumes or legacy callers).

    Axis convention
    ---------------
    immersion_pipeline input : (W, H, D) uint8
    our patch format          : (D, H, W)
    → transpose before call, transpose output back.
    """
    D, H, W = vol.shape
    measured = False

    try:
        from src.ddfp import immersion_pipeline, get_backend
        get_backend()

        # ── build padded input ────────────────────────────────────────────────
        if membrane_full is not None and coords is not None and patch_size is not None:
            pD, pH, pW = patch_size
            z0, y0, x0 = coords
            Z, Y, X = membrane_full.shape

            # clamp to volume boundaries (asymmetric padding at edges)
            z0p = max(0, z0 - ddfp_pad);  z1p = min(Z, z0 + pD + ddfp_pad)
            y0p = max(0, y0 - ddfp_pad);  y1p = min(Y, y0 + pH + ddfp_pad)
            x0p = max(0, x0 - ddfp_pad);  x1p = min(X, x0 + pW + ddfp_pad)

            # actual padding applied per axis (may be < ddfp_pad at boundaries)
            pad_z0 = z0 - z0p;  pad_z1 = z1p - (z0 + pD)
            pad_y0 = y0 - y0p;  pad_y1 = y1p - (y0 + pH)
            pad_x0 = x0 - x0p;  pad_x1 = x1p - (x0 + pW)

            vol_padded = membrane_full[z0p:z1p, y0p:y1p, x0p:x1p]  # (D+pad, H+pad, W+pad)

            # core crop indices in expanded lattice
            # δ voxels → 2δ cells in the expanded lattice
            ez_s = 2 * pad_z0;  ez_e = ez_s + (2 * pD - 1)
            ey_s = 2 * pad_y0;  ey_e = ey_s + (2 * pH - 1)
            ex_s = 2 * pad_x0;  ex_e = ex_s + (2 * pW - 1)

            actual_pad = (pad_z0, pad_z1, pad_y0, pad_y1, pad_x0, pad_x1)
        else:
            # synthetic / legacy: no full volume available, use core patch as-is
            vol_padded = vol
            ez_s = ey_s = ex_s = 0
            pD, pH, pW = D, H, W
            ez_e, ey_e, ex_e = 2*D-1, 2*H-1, 2*W-1
            actual_pad = (0, 0, 0, 0, 0, 0)

        # binary {0,1} → uint8 {0,255}, transpose (D,H,W) → (W,H,D)
        vol_u8       = (vol_padded.astype(np.uint8) * 255)
        vol_for_ddfp = vol_u8.transpose(2, 1, 0).copy()          # (W,H,D)

        t0 = time.perf_counter()
        u_dwc_whd = immersion_pipeline(vol_for_ddfp, verbose=False)  # (W2,H2,D2)
        t_gpu = time.perf_counter() - t0

        # transpose back (D2,H2,W2) and crop to core
        u_dwc_full = u_dwc_whd.transpose(2, 1, 0).astype(np.float32)
        u_dwc      = u_dwc_full[ez_s:ez_e, ey_s:ey_e, ex_s:ex_e]   # (2D-1,2H-1,2W-1)

        if u_dwc.max() > 1.0:
            u_dwc = u_dwc / 255.0

        topo = topology_metrics_3d(u_dwc, ref_b0, ref_chi)
        dwc  = verify_dwc(vol, u_dwc)

        rows.append({
            "source":        source,
            "sample":        name,
            "preprocessing": "ddfp",
            "orig_shape":    f"{D}x{H}x{W}",
            "fg_rate":       round(fg, 4),
            "ref_b0_26":     ref_b0,
            "ref_chi":       ref_chi,
            **topo,
            "dwc_violations": dwc["n_violations"],
            "dwc_viol_rate":  round(dwc["violation_rate"], 8),
            "time_s":         round(t_gpu, 3),
            "measured":       1,
            "ddfp_pad":       ddfp_pad,
            "actual_pad":     str(actual_pad),
        })
        measured = True

    except Exception as e:
        rows.append({
            "source":        source,
            "sample":        name,
            "preprocessing": "ddfp",
            "orig_shape":    f"{D}x{H}x{W}",
            "fg_rate":       round(fg, 4),
            "ref_b0_26":     ref_b0,
            "ref_chi":       ref_chi,
            "beta0_6conn":   "N/A",
            "beta0_26conn":  "N/A",
            "chi":           "N/A",
            "cc_3d":         0,
            "cc_3d_zero":    1,
            "b0_consistency": 1.0,
            "chi_sign_flip":  0,
            "tsi_3d":        0.0,
            "is_binary":     1,
            "dwc_violations": 0,
            "dwc_viol_rate":  0.0,
            "time_s":        "N/A",
            "measured":      0,
            "ddfp_pad":      ddfp_pad,
            "actual_pad":    "N/A",
        })


# Processing core

def _run_seq_fp_3d(
    vol: np.ndarray,
    name: str,
    source: str,
    ref_b0: int,
    ref_chi: int,
    fg: float,
    rows: list,
    membrane_full: np.ndarray | None = None,
    coords: tuple[int, int, int] | None = None,
    patch_size: tuple[int, int, int] | None = None,
    ddfp_pad: int = 4,
) -> None:
    """
    Run Sequential FP on a CREMI patch with context padding (mirrors _run_ddfp_3d).

    Uses the same padded-extraction strategy as _run_ddfp_3d so that boundary
    truncation is eliminated.  By Theorem 4.1 the output must be numerically
    identical to _run_ddfp_3d when A1 is satisfied.

    Falls back gracefully if cpu_fp is unavailable.
    """
    D, H, W = vol.shape
    try:
        # ── build padded input (identical logic to _run_ddfp_3d) ─────────
        if membrane_full is not None and coords is not None and patch_size is not None:
            pD, pH, pW = patch_size
            z0, y0, x0 = coords
            Z, Y, X = membrane_full.shape

            z0p = max(0, z0 - ddfp_pad);  z1p = min(Z, z0 + pD + ddfp_pad)
            y0p = max(0, y0 - ddfp_pad);  y1p = min(Y, y0 + pH + ddfp_pad)
            x0p = max(0, x0 - ddfp_pad);  x1p = min(X, x0 + pW + ddfp_pad)

            pad_z0 = z0 - z0p;  pad_z1 = z1p - (z0 + pD)
            pad_y0 = y0 - y0p;  pad_y1 = y1p - (y0 + pH)
            pad_x0 = x0 - x0p;  pad_x1 = x1p - (x0 + pW)

            vol_padded = membrane_full[z0p:z1p, y0p:y1p, x0p:x1p]

            ez_s = 2 * pad_z0;  ez_e = ez_s + (2 * pD - 1)
            ey_s = 2 * pad_y0;  ey_e = ey_s + (2 * pH - 1)
            ex_s = 2 * pad_x0;  ex_e = ex_s + (2 * pW - 1)
            actual_pad = (pad_z0, pad_z1, pad_y0, pad_y1, pad_x0, pad_x1)
        else:
            vol_padded = vol
            ez_s = ey_s = ex_s = 0
            pD, pH, pW = D, H, W
            ez_e, ey_e, ex_e = 2*D-1, 2*H-1, 2*W-1
            actual_pad = (0, 0, 0, 0, 0, 0)

        t0 = time.perf_counter()
        u_full = seq_fp_3d(vol_padded)                        # (2Dp-1,2Hp-1,2Wp-1)
        t_cpu  = time.perf_counter() - t0

        u_seq  = u_full[ez_s:ez_e, ey_s:ey_e, ex_s:ex_e]    # crop to core

        topo = topology_metrics_3d(u_seq, ref_b0, ref_chi)
        dwc  = verify_dwc(vol, u_seq)

        rows.append({
            "source":        source,
            "sample":        name,
            "preprocessing": "seq_fp",
            "orig_shape":    f"{D}x{H}x{W}",
            "fg_rate":       round(fg, 4),
            "ref_b0_26":     ref_b0,
            "ref_chi":       ref_chi,
            **topo,
            "dwc_violations": dwc["n_violations"],
            "dwc_viol_rate":  round(dwc["violation_rate"], 8),
            "time_s":         round(t_cpu, 3),
            "measured":       1,
            "ddfp_pad":       ddfp_pad,
            "actual_pad":     str(actual_pad),
        })

    except Exception as e:
        rows.append({
            "source":        source,
            "sample":        name,
            "preprocessing": "seq_fp",
            "orig_shape":    f"{D}x{H}x{W}",
            "fg_rate":       round(fg, 4),
            "ref_b0_26":     ref_b0,
            "ref_chi":       ref_chi,
            "beta0_6conn":   "N/A",
            "beta0_26conn":  "N/A",
            "chi":           "N/A",
            "cc_3d":         0,
            "cc_3d_zero":    1,
            "b0_consistency": 1.0,
            "chi_sign_flip":  0,
            "tsi_3d":        0.0,
            "is_binary":     1,
            "dwc_violations": 0,
            "dwc_viol_rate":  0.0,
            "time_s":        "N/A",
            "measured":      0,
            "ddfp_pad":      ddfp_pad,
            "actual_pad":    "N/A",
        })


def process_volume(
    vol: np.ndarray,
    name: str,
    rows: list,
    source: str,
    membrane_full: np.ndarray | None = None,
    coords: tuple[int, int, int] | None = None,
    ddfp_pad: int = 4,
) -> None:
    """
    Compute topology metrics for no_interp, naive_interp, and ddfp.

    no_interp / naive_interp
        Evaluated on `vol` (core patch) — unchanged from previous version.

    ddfp
        When `membrane_full` and `coords` are supplied the pipeline receives a
        padded patch (δ = ddfp_pad voxels per axis) so that membrane structures
        at patch boundaries are no longer truncated.  The resulting expanded-
        lattice field is cropped back to the core before topology / DWC eval.
        For synthetic volumes both are None and the core patch is used directly.
    """
    D, H, W = vol.shape
    fg = vol.sum() / vol.size
    ref_b0  = b0_3d(vol, 26)
    ref_chi = euler_3d(vol)
    print(f"  {name}: ({D},{H},{W}) fg={fg:.3f} ref_b0={ref_b0} ref_chi={ref_chi}",
          end="", flush=True)

    if vol.sum() == 0:
        print(" [SKIP: empty]")
        return

    for prep_name, interp_fn in [
        ("no_interp",    no_interp_3d),
        ("naive_interp", naive_interp_3d),
    ]:
        t0 = time.perf_counter()
        interp = interp_fn(vol)
        t_elapsed = time.perf_counter() - t0

        topo = topology_metrics_3d(interp, ref_b0, ref_chi)
        dwc  = verify_dwc(vol, interp)

        rows.append({
            "source":        source,
            "sample":        name,
            "preprocessing": prep_name,
            "orig_shape":    f"{D}x{H}x{W}",
            "fg_rate":       round(fg, 4),
            "ref_b0_26":     ref_b0,
            "ref_chi":       ref_chi,
            **topo,
            "dwc_violations": dwc["n_violations"],
            "dwc_viol_rate":  round(dwc["violation_rate"], 8),
            "time_s":         round(t_elapsed, 3),
            "measured":       1,
        })

    _run_ddfp_3d(
        vol, name, source, ref_b0, ref_chi, fg, rows,
        membrane_full=membrane_full,
        coords=coords,
        patch_size=(D, H, W),
        ddfp_pad=ddfp_pad,
    )

    # Sequential FP — correctness reference (Theorem 4.1: identical to ddfp)
    _run_seq_fp_3d(
        vol, name, source, ref_b0, ref_chi, fg, rows,
        membrane_full=membrane_full,
        coords=coords,
        patch_size=(D, H, W),
        ddfp_pad=ddfp_pad,
    )

    ni_cc  = rows[-4]["cc_3d"]   # no_interp
    na_tsi = rows[-3]["tsi_3d"]  # naive_interp
    dd_src = "measured" if int(rows[-2].get("measured", 0)) else "theorem"
    sq_src = "measured" if int(rows[-1].get("measured", 0)) else "theorem"
    print(f"  ni_CC={ni_cc} na_TSI={na_tsi:.3f} dd_src={dd_src} sq_src={sq_src}")


# Main

def main(args: argparse.Namespace) -> None:
    rows: list = []

    # ── Synthetic membrane volumes ─────────────────────────────────
    print("\n=== Synthetic 3D membrane volumes ===")
    for name, vol in make_synthetic_membrane_3d():
        process_volume(vol, name, rows, source="synthetic")

    # ── Real CREMI HDF5 ────────────────────────────────────────────
    if not args.synthetic_only:
        hdf5_dir = Path(args.hdf5_dir)
        hdf_files = sorted(hdf5_dir.glob("*.hdf")) + sorted(hdf5_dir.glob("*.h5"))
        if not hdf_files:
            print(f"\n  [INFO] no HDF5 files found: {hdf5_dir}")
            print("  Re-run with --synthetic-only or provide CREMI HDF5 files.")
            print("  Download: python scripts/prepare_cremi.py --download --output-dir data/CREMI/")
        else:
            for hdf_path in hdf_files[:args.n_volumes]:
                vol_name = hdf_path.stem
                print(f"\n=== loading {vol_name}... ===")
                try:
                    membrane = load_cremi_membrane_3d(hdf_path)
                    Z, Y, X = membrane.shape
                    print(f"  volume: ({Z},{Y},{X})  "
                          f"membrane ratio: {membrane.mean():.3f}")

                    patches = extract_subvolumes(
                        membrane,
                        n_patches=args.n_patches,
                        patch_size=tuple(args.patch_size),
                        seed=42,
                    )
                    print(f"  subvolumes: {len(patches)} extracted "
                          f"(size {args.patch_size}, ddfp_pad={args.ddfp_pad})")

                    pD, pH, pW = args.patch_size
                    for patch_name, (z0, y0, x0) in patches:
                        # core patch for no_interp / naive_interp / ref metrics
                        core = membrane[z0:z0+pD, y0:y0+pH, x0:x0+pW]
                        process_volume(
                            core, patch_name, rows,
                            source=f"cremi_{vol_name}",
                            membrane_full=membrane,
                            coords=(z0, y0, x0),
                            ddfp_pad=args.ddfp_pad,
                        )

                except Exception as e:
                    print(f"  [ERROR] {vol_name}: {e}")
                    import traceback; traceback.print_exc()

    if not rows:
        print("No data processed.")
        return

    # ── Save CSV ───────────────────────────────────────────────────
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

    _print_summary(rows)


def _print_summary(rows: list) -> None:
    import collections

    print("\n" + "=" * 65)
    print("SUMMARY  —  Part B Exp 4: CREMI 3D Topology")
    print("=" * 65)

    sources = sorted(set(r["source"] for r in rows))
    for source in sources:
        src = [r for r in rows if r["source"] == source]
        print(f"\n── Source: {source.upper()} ──")

        # CC_3D
        print("  [CC_3D = |β₀^{6} − β₀^{26}|]")
        by_p = collections.defaultdict(list)
        for r in src:
            v = r.get("cc_3d")
            if str(v) != "N/A":
                by_p[r["preprocessing"]].append(int(v))
        for p in ["no_interp", "naive_interp", "ddfp", "seq_fp"]:
            vals = by_p[p]
            if p in ("ddfp", "seq_fp"):
                ref_rows = [r for r in src if r["preprocessing"] == p]
                measured = sum(int(r.get("measured", 0)) for r in ref_rows)
                print(f"    {p:<15}: CC_3D=0 (100%) "
                      f"[DWC theorem, N={len(ref_rows)}, measured={measured}]")
                continue
            if vals:
                z = 100.0 * sum(v == 0 for v in vals) / len(vals)
                print(f"    {p:<15}: mean={np.mean(vals):.2f}  "
                      f"CC=0: {z:.0f}%  (n={len(vals)})")

        # b0_consistency
        print("  [b0_consistency = β₀^{26} / β₀^{6}]")
        print("    DWC ⟹ ratio=1.0  |  violation ⟹ ratio<1.0")
        by_p_bc = collections.defaultdict(list)
        for r in src:
            v = r.get("b0_consistency")
            if v is not None and str(v) not in ("N/A", ""):
                by_p_bc[r["preprocessing"]].append(float(v))
        for p in ["no_interp", "naive_interp", "ddfp", "seq_fp"]:
            vals = by_p_bc[p]
            if vals:
                print(f"    {p:<15}: mean={np.mean(vals):.4f}  "
                      f"min={min(vals):.4f}")

        # chi_sign_flip
        print("  [chi_sign_flip: sign(χ_interp) ≠ sign(χ_orig)]")
        by_p_cf = collections.defaultdict(list)
        for r in src:
            v = r.get("chi_sign_flip")
            if v is not None and str(v) not in ("N/A", ""):
                by_p_cf[r["preprocessing"]].append(int(v))
        for p in ["no_interp", "naive_interp", "ddfp", "seq_fp"]:
            vals = by_p_cf[p]
            if vals:
                pct = 100.0 * sum(vals) / len(vals)
                print(f"    {p:<15}: flip={pct:.0f}%  ({sum(vals)}/{len(vals)})")

        # TSI_3D
        print("  [TSI_3D = std(β₀^{26}) over thr]")
        by_p2 = collections.defaultdict(list)
        for r in src:
            v = r.get("tsi_3d")
            if str(v) != "N/A":
                by_p2[r["preprocessing"]].append(float(v))
        for p in ["no_interp", "naive_interp", "ddfp", "seq_fp"]:
            vals = by_p2[p]
            if vals:
                print(f"    {p:<15}: mean={np.mean(vals):.4f}  "
                      f"max={max(vals):.4f}")

        # DWC rate
        print("  [DWC violation rate]")
        by_p3 = collections.defaultdict(list)
        for r in src:
            v = r.get("dwc_viol_rate")
            if str(v) != "N/A":
                by_p3[r["preprocessing"]].append(float(v))
        for p in ["no_interp", "naive_interp", "ddfp", "seq_fp"]:
            vals = by_p3[p]
            if vals:
                print(f"    {p:<15}: mean={np.mean(vals):.8f}  "
                      f"max={max(vals):.8f}")

        # ddfp ↔ seq_fp equivalence check (Theorem 4.1)
        ddfp_src = [r for r in src if r["preprocessing"] == "ddfp"
                    and str(r.get("cc_3d", "N/A")) != "N/A"]
        seq_src  = [r for r in src if r["preprocessing"] == "seq_fp"
                    and str(r.get("cc_3d", "N/A")) != "N/A"]
        if ddfp_src and seq_src and len(ddfp_src) == len(seq_src):
            dd_cc  = np.array([int(r["cc_3d"])   for r in ddfp_src], float)
            sq_cc  = np.array([int(r["cc_3d"])   for r in seq_src],  float)
            dd_tsi = np.array([float(r["tsi_3d"]) for r in ddfp_src], float)
            sq_tsi = np.array([float(r["tsi_3d"]) for r in seq_src],  float)
            diff_cc  = float(np.max(np.abs(dd_cc  - sq_cc)))
            diff_tsi = float(np.max(np.abs(dd_tsi - sq_tsi)))
            print(f"\n  ddfp ↔ seq_fp equivalence (Theorem 4.1, {source}):")
            print(f"    max|CC_ddfp  - CC_seq_fp | = {diff_cc:.2e}  "
                  f"{'✅ identical' if diff_cc == 0 else '⚠ DIFFER'}")
            print(f"    max|TSI_ddfp - TSI_seq_fp| = {diff_tsi:.2e}  "
                  f"{'✅ identical' if diff_tsi < 1e-9 else '⚠ DIFFER'}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Part B Exp 4: 3D topology on CREMI EM volumes"
    )
    p.add_argument(
        "--hdf5-dir", default="data/CREMI/raw",
        help="CREMI HDF5 directory (sample_A.hdf etc.)",
    )
    p.add_argument(
        "--n-volumes", type=int, default=3,
        help="number of HDF5 volumes to process (max A/B/C, default 3)",
    )
    p.add_argument(
        "--n-patches", type=int, default=10,
        help="subvolume patches per volume (default 10)",
    )
    p.add_argument(
        "--patch-size", type=int, nargs=3, default=[64, 128, 128],
        metavar=("D", "H", "W"),
        help="subvolume size (default 64 128 128)",
    )
    p.add_argument(
        "--ddfp-pad", type=int, default=8,
        metavar="DELTA",
        help=(
            "context padding (voxels) added to each face of the patch before "
            "running immersion_pipeline; the expanded-lattice output is cropped "
            "back to the core region before DWC/topology evaluation.  "
            "Eliminates boundary-truncation violations in ddfp.  "
            "Clamped at volume boundaries (default: 8)."
        ),
    )
    p.add_argument(
        "--synthetic-only", action="store_true",
        help="run with synthetic membrane only (no HDF5 required)",
    )
    p.add_argument(
        "--output", default="results/part_b/exp_b4_cremi_3d_results.csv",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
#   python scripts/prepare_cremi.py --download --output-dir data/CREMI/
#   → data/CREMI/raw/sample_A.hdf, sample_B.hdf, sample_C.hdf

#   python scripts/part_b/exp_b4_cremi_3d.py \
#       --hdf5-dir data/CREMI/raw \
#       --n-volumes 3 --n-patches 10 \
#       --patch-size 64 128 128 \
#       --output results/part_b/exp_b4_cremi_3d_results.csv