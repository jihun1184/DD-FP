"""
src/benchmark_utils.py
======================
Shared utilities for Part A (DD-FP preprocessing validation) experiments.

Three public functions
----------------------
  verify_dwc(original, interpolated)  ->  dict
  time_function(fn, *args, **kwargs)  ->  dict
  generate_synthetic_volume(size, seed) -> np.ndarray

All functions support both 2D (H, W) and 3D (D, H, W) inputs.
"""
from __future__ import annotations

import time
from typing import Callable, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter

# ---------------------------------------------------------------------------
# Optional GPU support
# ---------------------------------------------------------------------------
try:
    import cupy as cp
    _CUPY_OK = True
except ImportError:
    _CUPY_OK = False


# ===========================================================================
# 1.  verify_dwc
# ===========================================================================

def verify_dwc(
    original: np.ndarray,
    interpolated: np.ndarray,
) -> dict:
    """
    Verify the Digital Well-Composedness (DWC) property of an interpolated
    Khalimsky-grid image (Boutry 2015, Property 1).

    Works for both 2D (H, W) and 3D (D, H, W) inputs.

    Boutry 2015 Property 1 (exact definition)
    ------------------------------------------
    u is DWC  iff  for every k-dimensional face-block S and every antagonist
    pair (p, p') in S:
        intvl(u[p], u[p']) ∩ span{ u[r] | r in S \\ {p,p'} } != empty

    For a 2x2 face-block with corners v00, v10, v01, v11:
      Pair A: (v00, v11) antagonists -- their interval must overlap span{v10,v01}
      Pair B: (v10, v01) antagonists -- their interval must overlap span{v00,v11}
    Both must hold; a block fails if either pair fails.

    For 3D: additionally check 2x2x2 body-blocks with 4 space-diagonal pairs.

    This is a direct vectorised CPU port of the CUDA kernel
    `dwc_check_all_blocks` from gpu_immersion.py (validated on N=20 BraTS
    subjects, all violations==0 after DD-FP with delta>=1).

    Parameters
    ----------
    original      : uint8 ndarray (H,W) or (D,H,W) -- unused in check, kept
                    for API compatibility with calling code.
    interpolated  : float32 ndarray (2H-1,2W-1) or (2D-1,2H-1,2W-1)

    Returns
    -------
    dict:
        n_violations  : int   -- number of violating face/body-blocks
        max_abs_error : float -- max violation margin across all blocks
        violation_rate: float -- n_violations / total_blocks_checked
    """
    u = interpolated.astype(np.float32)
    ndim = u.ndim

    def _intersects(a, b, rest):
        """Does [min(a,b), max(a,b)] intersect span(rest)?  Vectorised."""
        lo_ab = np.minimum(a, b)
        hi_ab = np.maximum(a, b)
        lo_r = rest[0].copy()
        hi_r = rest[0].copy()
        for r in rest[1:]:
            np.minimum(lo_r, r, out=lo_r)
            np.maximum(hi_r, r, out=hi_r)
        return (lo_ab <= hi_r) & (lo_r <= hi_ab)

    def _margin(a, b, rest):
        """Signed gap between the two intervals (>0 means no intersection)."""
        lo_ab = np.minimum(a, b);  hi_ab = np.maximum(a, b)
        lo_r = rest[0].copy();     hi_r = rest[0].copy()
        for r in rest[1:]:
            np.minimum(lo_r, r, out=lo_r)
            np.maximum(hi_r, r, out=hi_r)
        return np.maximum(lo_ab - hi_r, lo_r - hi_ab)

    n_violations = 0
    total_blocks = 0
    max_margin   = 0.0

    # ── 2D face-blocks: every adjacent pair at step=1 ────────────────────────
    # Matches CUDA kernel dwc_check_all_blocks: block index (x,y) ranges over
    # ALL consecutive pairs 0..Wp-2, covering 0-cells, 1-cells, and 2-cells.
    # The violation check is: do the two diagonal intervals intersect?
    axis_pairs = [(0, 1)] if ndim == 2 else [(0, 1), (0, 2), (1, 2)]

    for ax0, ax1 in axis_pairs:
        s = [slice(None)] * ndim

        def _c(d0, d1, _s=s, _ax0=ax0, _ax1=ax1):
            sl = list(_s)
            sl[_ax0] = slice(0, -1) if d0 == 0 else slice(1, None)
            sl[_ax1] = slice(0, -1) if d1 == 0 else slice(1, None)
            return u[tuple(sl)]

        v00, v10, v01, v11 = _c(0,0), _c(1,0), _c(0,1), _c(1,1)

        okA = _intersects(v00, v11, [v10, v01])
        okB = _intersects(v10, v01, [v00, v11])
        viol = ~(okA & okB)

        n_violations += int(viol.sum())
        total_blocks  += viol.size

        mg = np.where(viol, np.maximum(_margin(v00, v11, [v10, v01]),
                                        _margin(v10, v01, [v00, v11])), 0.0)
        max_margin = max(max_margin, float(mg.max()))

    # ── 3D body-blocks (2x2x2): every adjacent triple at step=1 ─────────────
    if ndim == 3:
        c = {}
        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    sl = (slice(0,-1) if dz==0 else slice(1,None),
                          slice(0,-1) if dy==0 else slice(1,None),
                          slice(0,-1) if dx==0 else slice(1,None))
                    c[(dz,dy,dx)] = u[sl]

        all_keys = [(dz,dy,dx) for dz in range(2)
                                for dy in range(2)
                                for dx in range(2)]
        diag_pairs = [
            ((0,0,0),(1,1,1)), ((1,0,0),(0,1,1)),
            ((0,1,0),(1,0,1)), ((0,0,1),(1,1,0)),
        ]
        block_viol = np.zeros(c[(0,0,0)].shape, dtype=bool)

        for pa, pb in diag_pairs:
            rest = [c[k] for k in all_keys if k != pa and k != pb]
            block_viol |= ~_intersects(c[pa], c[pb], rest)

        n_violations += int(block_viol.sum())
        total_blocks  += block_viol.size

    violation_rate = n_violations / max(total_blocks, 1)

    return {
        "n_violations":   n_violations,
        "max_abs_error":  round(max_margin, 4),
        "violation_rate": violation_rate,
    }


# ===========================================================================
# 2.  time_function
# ===========================================================================

def time_function(
    fn: Callable,
    *args,
    n_repeats: int = 5,
    warmup: int = 1,
    use_cuda_events: bool = False,
    **kwargs,
) -> dict:
    """
    Measure wall-clock time of fn(*args, **kwargs).

    Parameters
    ----------
    fn            : callable to time
    *args         : positional arguments forwarded to fn
    n_repeats     : number of timed runs (after warmup)
    warmup        : number of warm-up runs (discarded)
    use_cuda_events: if True and CuPy available, use torch.cuda.Event timing
                    (more accurate for GPU kernels)
    **kwargs      : keyword arguments forwarded to fn

    Returns
    -------
    dict:
        median_s  : float  -- median time in seconds
        std_s     : float  -- std dev of times
        all_times : list   -- all n_repeats measured times (seconds)
    """
    # Warmup
    for _ in range(warmup):
        fn(*args, **kwargs)

    times = []
    for _ in range(n_repeats):
        if use_cuda_events and _CUPY_OK:
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            cp.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - t0)
        else:
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            times.append(time.perf_counter() - t0)

    times_arr = np.array(times)
    return {
        "median_s":  float(np.median(times_arr)),
        "std_s":     float(np.std(times_arr, ddof=1)) if len(times_arr) > 1 else 0.0,
        "all_times": times_arr.tolist(),
    }


# ===========================================================================
# 3.  generate_synthetic_volume
# ===========================================================================

def generate_synthetic_volume(
    size: Union[Tuple[int, ...], int],
    seed: int = 42,
    n_blobs: int = 5,
    noise_std: float = 8.0,
) -> np.ndarray:
    """
    Generate a deterministic synthetic volume for benchmarking.

    Creates non-trivial topology (multiple Gaussian blobs + noise) so that
    the DWC violation check is meaningful.  Works for 2D and 3D.

    Parameters
    ----------
    size  : tuple (H, W) or (D, H, W), or a single int for a cubic volume
    seed  : random seed for reproducibility
    n_blobs     : number of Gaussian foreground blobs
    noise_std   : additive Gaussian noise std (in uint8 [0,255] scale)

    Returns
    -------
    np.ndarray, dtype=uint8, shape=size
    """
    if isinstance(size, int):
        # Infer dimensionality: default 3D cube
        size = (size, size, size)

    ndim = len(size)
    rng = np.random.default_rng(seed)
    vol = np.zeros(size, dtype=np.float32)

    # Place Gaussian blobs at random centres
    for _ in range(n_blobs):
        centre = tuple(rng.integers(s // 4, 3 * s // 4) for s in size)
        sigma  = rng.uniform(min(size) * 0.08, min(size) * 0.25)
        blob   = np.zeros(size, dtype=np.float32)
        blob[centre] = 1.0
        blob = gaussian_filter(blob, sigma=sigma)
        if blob.max() > 0:
            blob /= blob.max()
        intensity = rng.uniform(100, 220)
        vol += blob * intensity

    # Add Gaussian noise
    vol += rng.normal(0, noise_std, size).astype(np.float32)

    # Normalise to [0, 255] uint8
    vol = np.clip(vol, 0, None)
    if vol.max() > 0:
        vol = vol / vol.max() * 255.0

    return vol.astype(np.uint8)


# ---------------------------------------------------------------------------
# Convenience: naive bilinear interpolation (for A1 baseline comparison)
# ---------------------------------------------------------------------------

def naive_interpolate(vol: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Produce a (2H-1, 2W-1) or (2D-1, 2H-1, 2W-1) interpolated volume
    using scipy.ndimage.zoom with the given spline order.

    order=1 -> bilinear/trilinear  (baseline "naive_interp")
    order=0 -> nearest-neighbour   (baseline "no_interp" equivalent)

    Output is cast to float32 and clipped to [0, 255].
    """
    from scipy.ndimage import zoom as ndimage_zoom

    ndim = vol.ndim
    # zoom factor per axis: (2*s - 1) / s = 2 - 1/s
    factors = tuple((2 * s - 1) / s for s in vol.shape)
    out = ndimage_zoom(vol.astype(np.float32), zoom=factors, order=order)
    return np.clip(out, 0, 255).astype(np.float32)