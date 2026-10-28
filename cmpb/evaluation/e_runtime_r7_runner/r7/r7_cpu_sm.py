"""R7 same-tie CPU state machine: `fp_cpu` (frozen, lib/cpu_fp.py sha 8eae554f...) copied verbatim except for the tie policy.
The only differences to the frozen function are marked by the diff stored in R7 deliverables (diffs/r7_cpu_sm_vs_cpu_fp.diff).
Equivalence to `seedref.seed_aware_cpu_reference` (tie lower/higher) and to `fp_cpu` (tie insertion) is checked by r7_selftest.py (CPU only)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from cpu_fp import _snap  # frozen helper (lib/cpu_fp.py)


def fp_cpu_tie(
    U_lo_pad: np.ndarray,
    U_hi_pad: np.ndarray,
    l_inf: float = 0.0,
    tie: str = "insertion",
    stats: dict | None = None,
) -> np.ndarray:
    """R7 copy of `fp_cpu` (Algorithm 1 Sequential FP) with an explicit equidistant-level tie policy.

    tie="insertion" reproduces the frozen `fp_cpu` bit for bit (dict insertion order decides ties).
    tie="lower" / "higher" = the CUDA-native tie policies, so this is the same-tie CPU state machine.
    `stats` (optional dict) receives tie_events, level_transitions, backward_transitions.

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
    if tie not in ("insertion", "lower", "higher"):
        raise ValueError("tie must be 'insertion', 'lower' or 'higher'")
    st = stats if stats is not None else {}
    st.update(tie_events=0, level_transitions=[], backward_transitions=0)

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
            prev = cur[0]
            if tie == "insertion":
                cur[0] = min(occ, key=lambda x: abs(x - cur[0]))
            else:
                d = min(abs(x - prev) for x in occ)
                cand = [x for x in occ if abs(x - prev) == d]
                st["tie_events"] += int(len(cand) > 1)
                cur[0] = min(cand) if tie == "lower" else max(cand)
            st["level_transitions"].append([int(prev), int(cur[0])])
            st["backward_transitions"] += int(cur[0] < prev)
        return Q[cur[0]].pop()

    # 6-directional offsets for 26-connected neighbors (face-adjacency)
    dx = [-1, 1, 0, 0, 0, 0]
    dy = [ 0, 0,-1, 1, 0, 0]
    dz = [ 0, 0, 0, 0,-1, 1]

    def nbs(idx: int) -> list[int]:
        """Return the list of 6-neighbor linear indices for linear index idx."""
        # NumPy ravel() is C-order for shape (Wp, Hp, Dp):
        # z is contiguous, then y, then x.
        x = idx // (Hp * Dp)
        r = idx % (Hp * Dp)
        y = r // Dp
        z = r % Dp
        res = []
        for i in range(6):
            nx, ny, nz = x + dx[i], y + dy[i], z + dz[i]
            if 0 <= nx < Wp and 0 <= ny < Hp and 0 <= nz < Dp:
                res.append((nx * Hp + ny) * Dp + nz)
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