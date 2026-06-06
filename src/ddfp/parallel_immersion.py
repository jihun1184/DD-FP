"""
parallel_immersion.py  (v2)
============================
Boutry, Géraud, Najman — "How to Make nD Functions Digitally Well-Composed
in a Self-dual Way" (ISMM 2015, hal-01168723)

Implements the full u_DWC construction and F_ord pipeline from §2-§5.

Pipeline:
  [input] uint8 3D volume  u: Ω ⊂ ℤ³ → V

  ── §5.1 Ispan interpolation ─────────────────────────────────────────────────
  build_ispan(u):
    0-cell:  U(z) = [u(z), u(z)]
    k≥1-cell: U(z) = [min(N(z)∩D), max(N(z)∩D)]  ← interval span
    outer border l_inf = median(inner boundary of u)
    → Ispan(u) is DWC (Prop 4) + Self-dual

  ── §4 Front propagation ─────────────────────────────────────────────────────
  front_propagation(U_lo, U_hi, l_inf):
    Hierarchical queue propagation (Algorithm 1)
    priority_push: snap current level ℓ to nearest point in [lo, hi]
    → u_DWC: DWC (Thm 2) + Self-dual (Prop 2) + Interpolation (Prop 5)

  ── §3 F_ord computation ─────────────────────────────────────────────────────
  compute_ford_from_immersion(u_dwc):
    stable rank(u_DWC) → total order rank
    → Max-Tree(F_ord) = ToS(u)  (Carlinet & Géraud Prop 3.2)

Usage:
  from parallel_immersion import immersion_pipeline
  vol_u8 = preprocess(raw_volume)
  u_dwc = immersion_pipeline(vol_u8)

Dependencies: numpy, scipy, collections
"""


import numpy as np
from scipy.ndimage import generic_filter
import time
from typing import Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# §A  Khalimsky grid cell-type map
# ═══════════════════════════════════════════════════════════════════════════════

# For each position (ix,iy,iz) in the expanded grid Ω̂:
#   all even      → 0-cell (original pixel)
#   one odd       → 1-cell (edge)
#   two odd       → 2-cell (face)
#   all odd       → 3-cell (voxel interior)
# Expanded grid size: (2W-1) × (2H-1) × (2D-1)

def cell_type_map(shape_hat: Tuple[int,int,int]) -> np.ndarray:
    """
    Return k-cell type (0-3) array for each expanded grid position.
    0: original pixel / 1: edge / 2: face / 3: voxel
    """
    W2, H2, D2 = shape_hat
    ix = np.arange(W2) % 2
    iy = np.arange(H2) % 2
    iz = np.arange(D2) % 2
    ktype = (ix[:, None, None] + iy[None, :, None] + iz[None, None, :]).astype(np.int8)
    return ktype


# ═══════════════════════════════════════════════════════════════════════════════
# §B  Self-dual interpolation  (Boutry 2015, §2.2 Definition 2.4)
# ═══════════════════════════════════════════════════════════════════════════════

def _embed_original(vol_u8: np.ndarray) -> np.ndarray:
    """
    Embed the original volume in the expanded grid Ω̂.
    Place original values at 0-cell positions (even indices); leave the rest zero.
    """
    W, H, D = vol_u8.shape
    vol_hat = np.zeros((2*W-1, 2*H-1, 2*D-1), dtype=np.float32)
    vol_hat[::2, ::2, ::2] = vol_u8.astype(np.float32)
    return vol_hat


def _fill_1cells(vol_hat: np.ndarray) -> None:
    """
    Fill 1-cells (edges): median of 2 adjacent 0-cells. In-place.
    """
    vol_hat[1::2, ::2, ::2] = (vol_hat[0::2, ::2, ::2][:-1] +
                                vol_hat[2::2, ::2, ::2]) / 2.0
    vol_hat[::2, 1::2, ::2] = (vol_hat[::2, 0::2, ::2][:, :-1] +
                                vol_hat[::2, 2::2, ::2]) / 2.0
    vol_hat[::2, ::2, 1::2] = (vol_hat[::2, ::2, 0::2][:, :, :-1] +
                                vol_hat[::2, ::2, 2::2]) / 2.0


def _fill_2cells(vol_hat: np.ndarray) -> None:
    """
    Fill 2-cells (faces): median of 4 corner 0-cells. In-place.

    xy-face: (odd, odd, even) → median of 4 corners
    xz-face: (odd, even, odd) → median of 4 corners
    yz-face: (even, odd, odd) → median of 4 corners
    """
    a = vol_hat[0::2, 0::2, ::2][:-1, :-1, :]   # (W-1, H-1, D')
    b = vol_hat[2::2, 0::2, ::2][:, :-1, :]
    c = vol_hat[0::2, 2::2, ::2][:-1, :, :]
    d = vol_hat[2::2, 2::2, ::2]
    vol_hat[1::2, 1::2, ::2] = _median4(a, b, c, d)

    a = vol_hat[0::2, ::2, 0::2][:-1, :, :-1]
    b = vol_hat[2::2, ::2, 0::2][:, :, :-1]
    c = vol_hat[0::2, ::2, 2::2][:-1, :, :]
    d = vol_hat[2::2, ::2, 2::2]
    vol_hat[1::2, ::2, 1::2] = _median4(a, b, c, d)

    a = vol_hat[::2, 0::2, 0::2][:, :-1, :-1]
    b = vol_hat[::2, 2::2, 0::2][:, :, :-1]
    c = vol_hat[::2, 0::2, 2::2][:, :-1, :]
    d = vol_hat[::2, 2::2, 2::2]
    vol_hat[::2, 1::2, 1::2] = _median4(a, b, c, d)


def _fill_3cells(vol_hat: np.ndarray) -> None:
    """
    Fill 3-cells (voxel interiors): median of 8 adjacent 0-cells. In-place.
    """
    corners = np.stack([
        vol_hat[0::2, 0::2, 0::2][:-1, :-1, :-1],
        vol_hat[2::2, 0::2, 0::2][:, :-1, :-1],
        vol_hat[0::2, 2::2, 0::2][:-1, :, :-1],
        vol_hat[2::2, 2::2, 0::2][:, :, :-1],
        vol_hat[0::2, 0::2, 2::2][:-1, :-1, :],
        vol_hat[2::2, 0::2, 2::2][:, :-1, :],
        vol_hat[0::2, 2::2, 2::2][:-1, :, :],
        vol_hat[2::2, 2::2, 2::2],
    ], axis=0)  # (8, W-1, H-1, D-1)
    vol_hat[1::2, 1::2, 1::2] = np.median(corners, axis=0)


def _median4(a, b, c, d):
    """Element-wise median of 4 arrays, optimised with min/max without sorting."""
    # 4-sample median = mean of middle two
    stacked = np.stack([a, b, c, d], axis=0)
    stacked.sort(axis=0)
    return (stacked[1] + stacked[2]) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# §C  Self-dual correction  (Boutry 2015, §2.3 Proposition 2.6)
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_self_dual_correction(vol_hat: np.ndarray,
                                 ktype: np.ndarray,
                                 n_iter: int = 3) -> np.ndarray:
    """
    Self-dual correction: clamp interpolated cells (k≥1) to the min/max range
    of neighbouring 0-cells so that Î satisfies Proposition 2.6.

    Proposition 2.6:
      Î(p) ∈ [min_{q∈N(p)} I(q),  max_{q∈N(p)} I(q)]   for all k≥1 cells p

    where N(p) is the set of 0-cells adjacent to p.

    n_iter: number of iterations (1 is sufficient; extra iterations improve boundary stability)
    """
    result = vol_hat.copy()
    W2, H2, D2 = vol_hat.shape

    interp_mask = ktype >= 1

    for _ in range(n_iter):
        neighbors = []
        pad = np.pad(result, 1, mode='edge')
        for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
            shifted = pad[1+dx:1+dx+W2, 1+dy:1+dy+H2, 1+dz:1+dz+D2]
            neighbors.append(shifted)

        nbr_min = np.minimum.reduce(neighbors)
        nbr_max = np.maximum.reduce(neighbors)

        result = np.where(interp_mask,
                          np.clip(result, nbr_min, nbr_max),
                          result)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# §D  Parallel immersion main function
# ═══════════════════════════════════════════════════════════════════════════════

def parallel_immerse(vol_u8: np.ndarray,
                     self_dual_iters: int = 1,
                     verbose: bool = True) -> np.ndarray:
    """
    Run full Parallel Immersion (Paper 2 §2).

    Args:
        vol_u8: preprocessed 3D volume, shape (W, H, D), dtype uint8
        self_dual_iters: self-dual correction iterations (default 1)
        verbose: print progress

    Returns:
        vol_hat: self-dual interpolation on the expanded grid,
                 shape (2W-1, 2H-1, 2D-1), dtype float32
    """
    t0 = time.time()
    W, H, D = vol_u8.shape
    W2, H2, D2 = 2*W-1, 2*H-1, 2*D-1

    if verbose:
        print(f"\n  [Immersion] input: {W}×{H}×{D}  "
              f"→ expanded: {W2}×{H2}×{D2}  "
              f"(×{W2*H2*D2/(W*H*D):.1f}x)")

    vol_hat = _embed_original(vol_u8)
    if verbose:
        print(f"  [A] 0-cell embedding done  ({time.time()-t0:.2f}s)")

    t1 = time.time()
    _fill_1cells(vol_hat)
    _fill_2cells(vol_hat)
    _fill_3cells(vol_hat)
    if verbose:
        print(f"  [B] k-cell interpolation done  (1-/2-/3-cell, {time.time()-t1:.2f}s)")

    t2 = time.time()
    ktype = cell_type_map((W2, H2, D2))
    vol_hat = _apply_self_dual_correction(vol_hat, ktype, n_iter=self_dual_iters)
    if verbose:
        unique_levels = len(np.unique(vol_hat))
        print(f"  [C] Self-dual correction done  (unique levels: {unique_levels:,}, {time.time()-t2:.2f}s)")

    if verbose:
        print(f"  [Immersion] total: {time.time()-t0:.2f}s  range=[{vol_hat.min():.1f}, {vol_hat.max():.1f}]")

    return vol_hat


# ═══════════════════════════════════════════════════════════════════════════════
# §E  Ispan interpolation  (Boutry et al. 2015, §5.1)
# ═══════════════════════════════════════════════════════════════════════════════

def build_ispan(vol_u8: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    §5.1 Proposition 4: construct Ispan(u).

    Build an interval-valued map on the expanded grid D2 ⊂ (Z/2)^n:
        - 0-cell (original pixel): U(z) = [u(z), u(z)]  (point interval)
        - k≥1-cell (interpolated): U(z) = [min(N(z)∩D), max(N(z)∩D)]
                                   where N(z) = B_z ∩ D, B_z = {-½,0,½}^n translated to z

    DWC guarantee: since bIspan(u)c = Imin(u) and dIspan(u)e = Imax(u) are both DWC,
                   Ispan(u) itself is DWC (Property 2).
    Self-dual: Ispan(-u) = -Ispan(u).

    Args:
        vol_u8: shape (W, H, D), dtype uint8  — original image u

    Returns:
        U_lo: shape (W2+2, H2+2, D2+2), float32  ← bIspan(u)c = Imin(u)  (with border)
        U_hi: shape (W2+2, H2+2, D2+2), float32  ← dIspan(u)e = Imax(u)  (with border)
        l_inf: outer border value
    """
    W, H, D = vol_u8.shape
    W2, H2, D2 = 2*W - 1, 2*H - 1, 2*D - 1

    src = vol_u8.astype(np.float32)

    U_lo = np.empty((W2, H2, D2), dtype=np.float32)
    U_hi = np.empty((W2, H2, D2), dtype=np.float32)
    U_lo[::2, ::2, ::2] = src
    U_hi[::2, ::2, ::2] = src

    a, b = src[:-1, :, :], src[1:, :, :]
    U_lo[1::2, ::2, ::2] = np.minimum(a, b)
    U_hi[1::2, ::2, ::2] = np.maximum(a, b)
    a, b = src[:, :-1, :], src[:, 1:, :]
    U_lo[::2, 1::2, ::2] = np.minimum(a, b)
    U_hi[::2, 1::2, ::2] = np.maximum(a, b)
    a, b = src[:, :, :-1], src[:, :, 1:]
    U_lo[::2, ::2, 1::2] = np.minimum(a, b)
    U_hi[::2, ::2, 1::2] = np.maximum(a, b)

    def _corners4_minmax(c00, c10, c01, c11):
        lo = np.minimum(np.minimum(c00, c10), np.minimum(c01, c11))
        hi = np.maximum(np.maximum(c00, c10), np.maximum(c01, c11))
        return lo, hi

    lo, hi = _corners4_minmax(src[:-1,:-1,:], src[1:,:-1,:],
                               src[:-1,1:,:],  src[1:,1:,:])
    U_lo[1::2, 1::2, ::2] = lo
    U_hi[1::2, 1::2, ::2] = hi
    lo, hi = _corners4_minmax(src[:-1,:,:-1], src[1:,:,:-1],
                               src[:-1,:,1:],  src[1:,:,1:])
    U_lo[1::2, ::2, 1::2] = lo
    U_hi[1::2, ::2, 1::2] = hi
    lo, hi = _corners4_minmax(src[:,:-1,:-1], src[:,1:,:-1],
                               src[:,:-1,1:],  src[:,1:,1:])
    U_lo[::2, 1::2, 1::2] = lo
    U_hi[::2, 1::2, 1::2] = hi

    c = np.stack([src[:-1,:-1,:-1], src[1:,:-1,:-1],
                  src[:-1,1:,:-1],  src[1:,1:,:-1],
                  src[:-1,:-1,1:],  src[1:,:-1,1:],
                  src[:-1,1:,1:],   src[1:,1:,1:]], axis=0)
    U_lo[1::2, 1::2, 1::2] = c.min(axis=0)
    U_hi[1::2, 1::2, 1::2] = c.max(axis=0)

    # l_inf = median{ u(z) | z ∈ inner boundary of D }
    inner_vals = np.concatenate([
        src[0,:,:].ravel(),  src[-1,:,:].ravel(),
        src[:,0,:].ravel(),  src[:,-1,:].ravel(),
        src[:,:,0].ravel(),  src[:,:,-1].ravel(),
    ])
    l_inf = float(np.median(inner_vals))

    def _pad_border(arr, val):
        return np.pad(arr, 1, mode='constant', constant_values=val)

    U_lo_pad = _pad_border(U_lo, l_inf)
    U_hi_pad = _pad_border(U_hi, l_inf)

    return U_lo_pad, U_hi_pad, l_inf


# ═══════════════════════════════════════════════════════════════════════════════
# §F  Front propagation  (Algorithm 1, Boutry et al. 2015)
# ═══════════════════════════════════════════════════════════════════════════════

def front_propagation(U_lo: np.ndarray,
                      U_hi: np.ndarray,
                      l_inf: float,
                      verbose: bool = True) -> np.ndarray:
    """
    §4.2 Algorithm 1: Front Propagation FP(U) → u_DWC.

    priority_push(Q, h, U, ℓ):
        lo, hi = U(h)
        if   ℓ < lo:  enqueue h at level lo   (snap up)
        elif ℓ > hi:  enqueue h at level hi   (snap down)
        else:         enqueue h at level ℓ    (keep current)

    priority_pop(Q, ℓ):
        if Q[ℓ] empty → search bidirectionally for nearest non-empty bucket

    Result:
        u_DWC(h) = level ℓ when h is popped  → always ∈ U(h)

    Args:
        U_lo, U_hi: output of build_ispan() (with outer border), shape (W2+2, H2+2, D2+2)
        l_inf: initial level

    Returns:
        u_dwc: same shape, dtype float32
    """
    from collections import deque
    t0 = time.time()
    shape = U_lo.shape
    W2p, H2p, D2p = shape
    total = W2p * H2p * D2p

    # ── Hierarchical queue ────────────────────────────────────────────────────
    V_MIN = int(np.floor(U_lo.min()))
    V_MAX = int(np.ceil(U_hi.max()))
    n_levels = V_MAX - V_MIN + 1
    buckets: list = [deque() for _ in range(n_levels)]

    def _lv2i(lv: float) -> int:
        return max(0, min(n_levels - 1, int(round(lv)) - V_MIN))

    def _priority_push(coord, cur_lv: float) -> None:
        lo = float(U_lo[coord])
        hi = float(U_hi[coord])
        if   cur_lv < lo:  enq = lo
        elif cur_lv > hi:  enq = hi
        else:               enq = cur_lv
        buckets[_lv2i(enq)].append(coord)

    u_dwc   = np.empty(shape, dtype=np.float32)
    u_dwc.fill(np.nan)
    visited = np.zeros(shape, dtype=bool)

    _DIRS = ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1))

    p_inf = (0, 0, 0)
    visited[p_inf] = True
    _priority_push(p_inf, l_inf)
    cur_idx = _lv2i(l_inf)

    processed = 0

    while processed < total:
        if not buckets[cur_idx]:
            found = False
            for delta in range(1, n_levels):
                up   = cur_idx + delta
                down = cur_idx - delta
                if up < n_levels and buckets[up]:
                    cur_idx = up;   found = True; break
                if down >= 0      and buckets[down]:
                    cur_idx = down; found = True; break
            if not found:
                break

        coord = buckets[cur_idx].popleft()
        cur_lv = float(V_MIN + cur_idx)

        u_dwc[coord] = cur_lv
        processed += 1

        x, y, z = coord
        for dx, dy, dz in _DIRS:
            nx, ny, nz = x+dx, y+dy, z+dz
            if 0 <= nx < W2p and 0 <= ny < H2p and 0 <= nz < D2p:
                nb = (nx, ny, nz)
                if not visited[nb]:
                    visited[nb] = True
                    _priority_push(nb, cur_lv)

    if verbose:
        nan_cnt = int(np.isnan(u_dwc).sum())
        status  = "✅" if nan_cnt == 0 else "⚠️"
        print(f"  [FP] {status} propagation done  processed={processed:,}/{total:,}  "
              f"NaN={nan_cnt}  ({time.time()-t0:.2f}s)")

    return u_dwc


# ═══════════════════════════════════════════════════════════════════════════════
# §G  u_DWC construction and F_ord computation  (§5.2 Proposition 5)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_udwc(vol_u8: np.ndarray,
                 verbose: bool = True) -> np.ndarray:
    """
    §5.2 Proposition 5: u_DWC = (FP ∘ Ispan)(u)

    Steps:
      1) Ispan(u): interval-valued map on expanded grid (DWC, Self-dual)
      2) FP(Ispan(u)): Front Propagation flattens to a single-valued function
         → result u_DWC is DWC, Self-dual, and interpolates u

    Args:
        vol_u8: shape (W, H, D), dtype uint8

    Returns:
        u_dwc: shape (2W-1, 2H-1, 2D-1), dtype float32
               outer border stripped; u_DWC|_D = u guaranteed
    """
    t0 = time.time()
    W, H, D = vol_u8.shape

    if verbose:
        print(f"\n  [u_DWC] §5.1 Ispan interpolation start...")

    t1 = time.time()
    U_lo, U_hi, l_inf = build_ispan(vol_u8)
    W2p = U_lo.shape[0]
    if verbose:
        W2 = 2*W-1
        print(f"  [Ispan] done  grid {W2}×{2*H-1}×{2*D-1}"
              f" + outer border → {U_lo.shape}  "
              f"l_inf={l_inf:.1f}  ({time.time()-t1:.2f}s)")

    if verbose:
        print(f"  [u_DWC] §4.2 Front Propagation start...")
    t2 = time.time()
    u_dwc_pad = front_propagation(U_lo, U_hi, l_inf, verbose=verbose)

    u_dwc = u_dwc_pad[1:-1, 1:-1, 1:-1]

    if verbose:
        orig_vals  = vol_u8.astype(np.float32)
        reconst    = u_dwc[::2, ::2, ::2]
        max_err    = np.abs(reconst - orig_vals).max()
        n_unique   = len(np.unique(u_dwc))
        print(f"  [u_DWC] done  shape={u_dwc.shape}  "
              f"unique={n_unique:,}  "
              f"interp_err(max)={max_err:.4f}  "
              f"total: {time.time()-t0:.2f}s")
        if max_err > 0.5:
            print(f"  [u_DWC] ⚠️  original pixel value mismatch (max_err={max_err})")

    return u_dwc


# def compute_ford_from_immersion(vol_hat: np.ndarray,
#                                  verbose: bool = True) -> np.ndarray:
#     """


#     → Max-Tree(F_ord) = ToS(u)

#     Args:

#     Returns:
#     """
#     t0 = time.time()
#     flat  = vol_hat.ravel().astype(np.float32)
#     N     = len(flat)

#     order     = np.argsort(flat, kind='stable')
#     ford_flat = np.empty(N, dtype=np.int32)
#     ford_flat[order] = np.arange(N, dtype=np.int32)
#     ford = ford_flat.reshape(vol_hat.shape)

#     if verbose:
#         n_unique = len(np.unique(ford_flat))
#               f"({time.time()-t0:.2f}s)")

#     return ford


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def export_for_cuda(vol_u8: np.ndarray,
                    ford: np.ndarray,
                    out_prefix: str = "/tmp/tos_input") -> dict:
    """
    Write binary files for the GPU pipeline (gpu_tos_3d.cu).

    Files:
        {prefix}_ford.bin  : int32 array, shape (W2, H2, D2), row-major
        {prefix}_meta.txt  : metadata text

    Args:
        vol_u8: preprocessed volume (W, H, D)
        ford:   F_ord array (2W-1, 2H-1, 2D-1)
        out_prefix: output file path prefix

    Returns:
        metadata dict
    """
    W, H, D = vol_u8.shape
    W2, H2, D2 = ford.shape

    ford_path = f"{out_prefix}_ford.bin"
    meta_path = f"{out_prefix}_meta.txt"

    ford.astype(np.int32).tofile(ford_path)

    meta = {
        "orig_shape": f"{W} {H} {D}",
        "hat_shape":  f"{W2} {H2} {D2}",
        "N_orig":     W * H * D,
        "N_hat":      W2 * H2 * D2,
        "ford_path":  ford_path,
        "ford_dtype": "int32",
        "ford_order": "C (row-major)",
    }
    with open(meta_path, 'w') as f:
        for k, v in meta.items():
            f.write(f"{k} = {v}\n")

    return meta


# ═══════════════════════════════════════════════════════════════════════════════
# §G  Drop-in replacement for brats_tos_pipeline.py Step 2
# ═══════════════════════════════════════════════════════════════════════════════

def immersion_pipeline(vol_u8: np.ndarray,
                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop-in replacement for brats_tos_pipeline.py Step 2 (compute_ford).

    v1: parallel_immerse (simple k-cell interpolation) → argsort F_ord
    v2: Ispan (interval map) → FP (Front Propagation) → u_DWC
        (full implementation of Boutry et al. 2015 §5.2 Proposition 5)

    Returns:
        u_dwc: (FP∘Ispan)(u), shape (2W-1,2H-1,2D-1), DWC guaranteed
    """
    u_dwc = compute_udwc(vol_u8, verbose=verbose)
    # ford  = compute_ford_from_immersion(u_dwc, verbose=verbose)
    return u_dwc


# ═══════════════════════════════════════════════════════════════════════════════
# §H  Validation utilities
# ═══════════════════════════════════════════════════════════════════════════════

def verify_self_dual(vol_hat: np.ndarray,
                     ktype: np.ndarray,
                     tol: float = 1e-4) -> bool:
    """
    Verify Proposition 2.6: check that all interpolated cells lie
    within the [min, max] range of adjacent 0-cells.
    """
    W2, H2, D2 = vol_hat.shape
    interp_mask = ktype >= 1

    pad = np.pad(vol_hat, 1, mode='edge')
    nbr_min = np.full_like(vol_hat, np.inf)
    nbr_max = np.full_like(vol_hat, -np.inf)
    for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
        shifted = pad[1+dx:1+dx+W2, 1+dy:1+dy+H2, 1+dz:1+dz+D2]
        nbr_min = np.minimum(nbr_min, shifted)
        nbr_max = np.maximum(nbr_max, shifted)

    violations = interp_mask & (
        (vol_hat < nbr_min - tol) | (vol_hat > nbr_max + tol)
    )
    n_viol = violations.sum()
    if n_viol > 0:
        print(f"  [Verify] ❌ Self-dual violated: {n_viol:,} cells")
        return False
    print(f"  [Verify] ✅ Self-dual condition satisfied (tol={tol})")
    return True


# def verify_ford_total_order(ford: np.ndarray) -> bool:
#     """
#     """
#     flat = ford.ravel()
#     n_unique = len(np.unique(flat))
#     if n_unique == len(flat):
#         return True
#     else:
#         dupes = len(flat) - n_unique
#         return False


# ═══════════════════════════════════════════════════════════════════════════════
# §I  Standalone test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    print("=" * 65)
    print("Parallel Immersion v2  —  FP∘Ispan  (Boutry et al. 2015)")
    print("=" * 65)

    print("\n[Test 1] Paper Fig.2/3 2D example (3×3×1 slice)")
    u_2d = np.array([[[9],[11],[15]],
                     [[7],[1], [13]],
                     [[3],[5], [3]]], dtype=np.uint8)   # shape (3,3,1)
    u_dwc = immersion_pipeline(u_2d, verbose=True)
    reconst = u_dwc[::2, ::2, ::2]
    print(f"\n  u (original):\n{u_2d[:,:,0]}")
    print(f"  u_DWC[0-cell] (== u guaranteed):\n{reconst[:,:,0].astype(int)}")
    orig_ok = np.allclose(reconst, u_2d.astype(np.float32))
    print(f"  u_DWC|_D == u: {'OK' if orig_ok else 'MISMATCH'}")

    print(f"\n  u_DWC expanded grid (5×5×1 central slice):")
    print(u_dwc[:,:,0].astype(int))

    print("\n[Test 2] Self-dual check: FP(Ispan(-u)) == -FP(Ispan(u))")
    u_neg = (255 - u_2d.astype(np.int16)).astype(np.uint8)
    # self-dual: (FP∘Ispan)(−u) = −(FP∘Ispan)(u)
    u_dwc_neg = immersion_pipeline(u_neg, verbose=False)
    sym_err = np.abs((u_dwc + u_dwc_neg) - 255).max()
    print(f"  symmetry error (max): {sym_err:.4f}  {'OK' if sym_err < 0.5 else 'WARN'}")

    print("\n[Test 3] 3D synthetic volume (16×16×8)")
    from scipy.ndimage import gaussian_filter
    np.random.seed(42)
    W, H, D = 16, 16, 8
    raw = gaussian_filter(np.random.rand(W,H,D).astype(np.float32), 2)
    vol_u8 = (raw * 255).astype(np.uint8)
    u_dwc3 = immersion_pipeline(vol_u8, verbose=True)

    print(f"\n  u_DWC shape: {u_dwc3.shape}  "
          f"range=[{u_dwc3.min():.0f}, {u_dwc3.max():.0f}]")
    # print(f"  F_ord shape: {ford3.shape}  "
    #       f"range=[{ford3.min()}, {ford3.max()}]")

    ktype = cell_type_map(u_dwc3.shape)
    verify_self_dual(u_dwc3, ktype)
    # verify_ford_total_order(ford3)

    reconst3 = u_dwc3[::2, ::2, ::2]
    err3 = np.abs(reconst3 - vol_u8.astype(np.float32)).max()
    print(f"  u_DWC|_D == u: {'✅' if err3 < 0.5 else '❌'} (max err={err3:.4f})")

    print("\n" + "=" * 65)
    print("done  —  ready for gpu_tos_3d.cu / brats_tos_pipeline_v3.py")
    print("=" * 65)