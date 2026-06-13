"""
src/ddfp/cpu_fp.py
------------------
CPU sequential Flat Persistence (FP) — Algorithm 1 (Boutry et al.).

This module provides a CPU baseline implementation
to verify numerical equivalence (Theorem 1 Step A) with the GPU IBI v10 results
from the E-NEW-3 experiment in `experiment_DDFP_all.py`.

Public API
----------
build_ispan_cpu(vol_u8) -> (U_lo_pad, U_hi_pad, l_inf)
    Builds interval span lattice on the CPU (preprocessing for Algorithm 1).

fp_cpu(U_lo_pad, U_hi_pad, l_inf) -> np.ndarray
    Runs sequential FP (Algorithm 1) on the CPU.

Usage
-----
    from src.ddfp.cpu_fp import build_ispan_cpu, fp_cpu

    U_lo, U_hi, l_inf = build_ispan_cpu(vol_u8)
    u_cpu = fp_cpu(U_lo, U_hi, l_inf)[1:-1, 1:-1, 1:-1]   # padding 제거
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────────

def _snap(lv: float, lo: float, hi: float) -> float:
    """Snap level `lv` into the range `[lo, hi]`.

    This single operation ensures that the level value does not exceed
    the interval span of each voxel just before it is inserted into
    the FP queue.

    Parameters
    ----------
    lv : float         Current propagation level
    lo : float         U_lo of the voxel
    hi : float         U_hi of the voxel

    Returns
    -------
    float  Clipped level value
    """
    return min(max(lv, lo), hi)


# ── Public API ─────────────────────────────────────────────────────────────────

def build_ispan_cpu(vol_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Builds the interval span lattice (U_lo, U_hi) on the CPU.

    This is a preprocessing step for Algorithm 1:
    The original grid (W×H×D) is expanded to an expanded lattice (2W-1 × 2H-1 × 2D-1),
    and then U_lo and U_hi are filled with the min/max values of adjacent voxels
    for each cell type (0-cell / 1-cell / 2-cell / 3-cell).
    The result is returned with boundary padding (1 cell) added.

    Parameters
    ----------
    vol_u8 : np.ndarray, shape (W, H, D), dtype uint8
        3-D volume normalized to the range 0~255.

    Returns
    -------
    U_lo_pad : np.ndarray, shape (W2+2, H2+2, D2+2), dtype float32
        Lower bound lattice including padding.  W2 = 2W-1 etc.
    U_hi_pad : np.ndarray, shape (W2+2, H2+2, D2+2), dtype float32
        Upper bound lattice including padding.
    l_inf : float
        Starting propagation level (always 0.0).
    """
    W, H, D = vol_u8.shape
    s = vol_u8.astype(np.float32)

    W2, H2, D2 = 2 * W - 1, 2 * H - 1, 2 * D - 1
    U_lo = np.empty((W2, H2, D2), np.float32)
    U_hi = np.empty((W2, H2, D2), np.float32)

    # ── 0-cell: original voxel positions (even indices) ──────────────────
    U_lo[::2, ::2, ::2] = s
    U_hi[::2, ::2, ::2] = s

    # ── 1-cell: min/max of two adjacent voxels (one odd index) ───────────
    for a, b, sl in [
        (s[:-1, :, :], s[1:, :, :], (np.s_[1::2], np.s_[::2],  np.s_[::2])),
        (s[:, :-1, :], s[:, 1:, :], (np.s_[::2],  np.s_[1::2], np.s_[::2])),
        (s[:, :, :-1], s[:, :, 1:], (np.s_[::2],  np.s_[::2],  np.s_[1::2])),
    ]:
        U_lo[sl] = np.minimum(a, b)
        U_hi[sl] = np.maximum(a, b)

    # ── 2-cell: min/max of four adjacent voxels (two odd indices) ─────────
    def mm4(f, *c):
        return f(f(c[0], c[1]), f(c[2], c[3]))

    U_lo[1::2, 1::2, ::2] = mm4(
        np.minimum, s[:-1, :-1, :], s[1:, :-1, :], s[:-1, 1:, :], s[1:, 1:, :]
    )
    U_hi[1::2, 1::2, ::2] = mm4(
        np.maximum, s[:-1, :-1, :], s[1:, :-1, :], s[:-1, 1:, :], s[1:, 1:, :]
    )
    U_lo[1::2, ::2, 1::2] = mm4(
        np.minimum, s[:-1, :, :-1], s[1:, :, :-1], s[:-1, :, 1:], s[1:, :, 1:]
    )
    U_hi[1::2, ::2, 1::2] = mm4(
        np.maximum, s[:-1, :, :-1], s[1:, :, :-1], s[:-1, :, 1:], s[1:, :, 1:]
    )
    U_lo[::2, 1::2, 1::2] = mm4(
        np.minimum, s[:, :-1, :-1], s[:, 1:, :-1], s[:, :-1, 1:], s[:, 1:, 1:]
    )
    U_hi[::2, 1::2, 1::2] = mm4(
        np.maximum, s[:, :-1, :-1], s[:, 1:, :-1], s[:, :-1, 1:], s[:, 1:, 1:]
    )

    # ── 3-cell: min/max of eight adjacent voxels (three odd indices) ──────
    c = np.stack([
        s[:-1, :-1, :-1], s[1:, :-1, :-1], s[:-1, 1:, :-1], s[1:, 1:, :-1],
        s[:-1, :-1, 1:],  s[1:, :-1, 1:],  s[:-1, 1:, 1:],  s[1:, 1:, 1:],
    ])
    U_lo[1::2, 1::2, 1::2] = c.min(0)
    U_hi[1::2, 1::2, 1::2] = c.max(0)

    # ── add boundary padding (required for the FP algorithm's edge handling) ──
    return (
        np.pad(U_lo, 1, constant_values=0.0),
        np.pad(U_hi, 1, constant_values=0.0),
        0.0,  # l_inf: starting propagation level
    )


def fp_cpu(
    U_lo_pad: np.ndarray,
    U_hi_pad: np.ndarray,
    l_inf: float = 0.0,
) -> np.ndarray:
    """Algorithm 1 Sequential Flat Persistence (FP) — CPU baseline.

    Performs level-set BFS on the interval span lattice.
    Each voxel is visited with a level value that is snapped into its [U_lo, U_hi]
    range using _snap().

    Used to verify numerical equivalence with GPU IBI v10 (front_propagation_gpu)
    in the E-NEW-3 experiment (Theorem 1 Step A).

    Parameters
    ----------
    U_lo_pad : np.ndarray  Lower bound lattice including padding (output of build_ispan_cpu)
    U_hi_pad : np.ndarray  Upper bound lattice including padding (output of build_ispan_cpu)
    l_inf    : float        Starting propagation level (default 0.0)

    Returns
    -------
    np.ndarray, shape == U_lo_pad.shape, dtype float32
        FP level assigned to each voxel. After calling, removing padding
        at [1:-1,1:-1,1:-1] results in the internal expanded lattice.

    Notes
    -----
    - Time complexity: O(N log N) — using a dict-based bucket queue.
    - This implementation is for accuracy verification only; use the GPU
      version for large volumes.
    """
    Wp, Hp, Dp = U_lo_pad.shape
    N = Wp * Hp * Dp

    u  = np.full(N, np.nan, np.float32)   # Result array (unvisited = NaN)
    dv = np.zeros(N, bool)                 # Visited flag
    Q: dict[int, list[int]] = defaultdict(list)  # Bucket queue {level: [voxel indices]}

    lo_f = U_lo_pad.ravel().astype(np.float64)
    hi_f = U_hi_pad.ravel().astype(np.float64)
    cur  = [int(round(l_inf))]             # Current propagation level (wrapped in a list for closure sharing)

    # ── Internal helper functions ──────────────────────────────────────────

    def push(h: int, lv: float) -> None:
        """Insert voxel h into the bucket for level snap(lv, lo[h], hi[h])."""
        Q[int(round(_snap(lv, lo_f[h], hi_f[h])))].append(h)

    def pop() -> int:
        """Pop a voxel from the current-level bucket.

        If the bucket is empty, advance cur[0] to the nearest level with
        non-empty buckets.  Return -1 if the entire queue is empty.
        """
        if not Q[cur[0]]:
            occ = [lv for lv, q in Q.items() if q]
            if not occ:
                return -1
            cur[0] = min(occ, key=lambda x: abs(x - cur[0]))
        return Q[cur[0]].pop()

    # 6-directional offsets for 26-connected neighbors (face-adjacency)
    dx = [-1, 1, 0, 0, 0, 0]
    dy = [ 0, 0,-1, 1, 0, 0]
    dz = [ 0, 0, 0, 0,-1, 1]

    def nbs(idx: int) -> list[int]:
        """Return the list of 6-neighbor linear indices for linear index idx."""
        z = idx // (Wp * Hp)
        r = idx  % (Wp * Hp)
        y = r // Wp
        x = r  % Wp
        res = []
        for i in range(6):
            nx, ny, nz = x + dx[i], y + dy[i], z + dz[i]
            if 0 <= nx < Wp and 0 <= ny < Hp and 0 <= nz < Dp:
                res.append(nx + ny * Wp + nz * Wp * Hp)
        return res

    # ── BFS propagation ──────────────────────────────────────────────────────
    Q[int(round(l_inf))].append(0)  # Start from the origin (voxel 0)
    dv[0] = True

    while True:
        h = pop()
        if h == -1:
            break
        u[h] = float(cur[0])
        for nb in nbs(h):
            if not dv[nb]:
                push(nb, float(cur[0]))
                dv[nb] = True

    return u.reshape(Wp, Hp, Dp)