"""R8-F1J independent CPU oracle (NumPy only; no repository imports).
Definitions are written from the mathematical statements, not from the CUDA kernel or the repo CPU code:
  * ISpan: separable min/max over the voxels incident to each expanded-lattice cell.
  * DWC (Boutry Property 1): every 2x2 face block and 2x2x2 cube; for each antagonist pair (p,p'),
    [min(u_p,u_p'), max(...)] meets [min, max] of the remaining block points.
  * Owner map (Round 0): owner(g)=k  <=>  g in [2*z0_k, 2*z1_k-1] (first tile from 0, last tile to 2D-2).
"""
from __future__ import annotations
import itertools
import numpy as np

SHAPES = [(3, 3, 8), (2, 2, 12), (4, 4, 10), (5, 5, 9), (6, 6, 12), (1, 3, 6), (2, 5, 3)]
FAMILIES = ["iid", "randomwalk", "checker_noise", "zlayer_noise", "box", "binary_density"]


# ---------------------------------------------------------------- generators
def levels(A: int) -> np.ndarray:
    """Alphabet of A evenly spaced integer levels in [0,255] (A=256 -> identity)."""
    if not 2 <= A <= 256:
        raise ValueError(f"alphabet size out of range: {A}")
    return np.round(np.linspace(0, 255, A)).astype(np.uint8)


def gen_volume(family: str, shape, A: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic uint8 volume of `shape` drawn from `family` over an A-level alphabet."""
    W, H, D = shape
    n = W * H * D
    x, y, z = np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing="ij")
    if family == "iid":
        idx = rng.integers(0, A, size=shape)
    elif family == "randomwalk":
        idx = np.clip(np.cumsum(rng.integers(-1, 2, size=n)) + A // 2, 0, A - 1).reshape(shape)
    elif family == "checker_noise":
        idx = ((x + y + z) % 2) * (A - 1)
        m = rng.random(shape) < 0.1
        idx = np.where(m, rng.integers(0, A, size=shape), idx)
    elif family == "zlayer_noise":
        base = rng.integers(0, A, size=D)[None, None, :] + np.zeros(shape, int)
        idx = np.clip(base + np.where(rng.random(shape) < 0.2, rng.integers(-1, 2, size=shape), 0), 0, A - 1)
    elif family == "box":
        bg, fg = rng.integers(0, A, size=2)
        lo = [int(rng.integers(0, s)) for s in shape]
        hi = [int(rng.integers(l + 1, s + 1)) for l, s in zip(lo, shape)]
        idx = np.full(shape, bg)
        idx[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = fg
    elif family == "binary_density":
        p = rng.uniform(0.1, 0.9)
        idx = (rng.random(shape) < p).astype(int) * (A - 1)
    else:
        raise ValueError(f"unknown family {family}")
    return levels(A)[np.asarray(idx, dtype=np.int64)].astype(np.uint8)


# ---------------------------------------------------------------- ISpan oracle
def _ex(a: np.ndarray, axis: int, op) -> np.ndarray:
    """Expand one axis: even slots copy a[i/2]; odd slots take op(a[i], a[i+1])."""
    n = a.shape[axis]
    sh = list(a.shape); sh[axis] = 2 * n - 1
    out = np.empty(sh, a.dtype)
    sl = lambda s: tuple(s if i == axis else slice(None) for i in range(a.ndim))
    out[sl(slice(0, None, 2))] = a
    out[sl(slice(1, None, 2))] = op(a[sl(slice(0, -1))], a[sl(slice(1, None))])
    return out


def ispan_oracle(vol_u8: np.ndarray):
    """Padded (lo, hi) float32 ISpan with a one-cell 0 border, built by separable expansion."""
    v = np.asarray(vol_u8, np.float32)
    lo, hi = v, v
    for ax in range(3):
        lo, hi = _ex(lo, ax, np.minimum), _ex(hi, ax, np.maximum)
    return np.pad(lo, 1).astype(np.float32), np.pad(hi, 1).astype(np.float32)


# ---------------------------------------------------------------- DWC oracle
def _meet(a, b, rest):
    """intvl(a,b) meets span(rest) (closed intervals)."""
    lo_ab, hi_ab = np.minimum(a, b), np.maximum(a, b)
    lo_r, hi_r = rest[0], rest[0]
    for r in rest[1:]:
        lo_r, hi_r = np.minimum(lo_r, r), np.maximum(hi_r, r)
    return (lo_ab <= hi_r) & (lo_r <= hi_ab)


def dwc_count_np(u: np.ndarray) -> dict:
    """Vectorised DWC counts. NaN-containing blocks are skipped and counted, like the GPU kernel."""
    u = np.asarray(u, np.float32)
    if u.ndim != 3:
        raise ValueError("expected 3-D array")
    Wp, Hp, Dp = u.shape
    sx = lambda a, b: (slice(a, Wp - b),)
    out = {"face_xy": 0, "face_xz": 0, "face_yz": 0, "cube": 0, "nan_skipped": 0, "total_blocks": 0}
    # faces: (v00, v10, v01, v11) with antagonist pairs (v00,v11) and (v10,v01)
    def face(a, b, c, d, key):
        if a.size == 0:
            return
        nan = np.isnan(a) | np.isnan(b) | np.isnan(c) | np.isnan(d)
        bad = ~(_meet(a, d, [b, c]) & _meet(b, c, [a, d])) & ~nan
        out[key] += int(bad.sum()); out["nan_skipped"] += int(nan.sum()); out["total_blocks"] += a.size
    face(u[:-1, :-1, :], u[1:, :-1, :], u[:-1, 1:, :], u[1:, 1:, :], "face_xy")
    face(u[:-1, :, :-1], u[1:, :, :-1], u[:-1, :, 1:], u[1:, :, 1:], "face_xz")
    face(u[:, :-1, :-1], u[:, 1:, :-1], u[:, :-1, 1:], u[:, 1:, 1:], "face_yz")
    # cubes
    c = {(i, j, k): u[i:Wp - 1 + i, j:Hp - 1 + j, k:Dp - 1 + k] for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    if c[(0, 0, 0)].size:
        nan = np.zeros(c[(0, 0, 0)].shape, bool)
        for v in c.values():
            nan |= np.isnan(v)
        bad = np.zeros_like(nan)
        for p in itertools.product((0, 1), repeat=3):
            q = tuple(1 - t for t in p)
            if p > q:  # each of the 4 antagonist pairs once
                continue
            bad |= ~_meet(c[p], c[q], [v for key, v in c.items() if key not in (p, q)])
        out["cube"] += int((bad & ~nan).sum()); out["nan_skipped"] += int(nan.sum()); out["total_blocks"] += nan.size
    out["violations"] = out["face_xy"] + out["face_xz"] + out["face_yz"] + out["cube"]
    return out


def dwc_count_bruteforce(u: np.ndarray) -> dict:
    """Scalar loop implementation of Property 1 with antagonists = coordinate complement. Tiny arrays only."""
    u = np.asarray(u, np.float32)
    if u.size > 4000:
        raise ValueError("bruteforce oracle limited to tiny arrays")
    viol = skipped = total = 0
    for axes in itertools.chain(itertools.combinations(range(3), 2), [(0, 1, 2)]):
        ranges = [range(u.shape[a] - 1) if a in axes else range(u.shape[a]) for a in range(3)]
        for base in itertools.product(*ranges):
            pts = []
            for off in itertools.product((0, 1), repeat=len(axes)):
                q = list(base)
                for a, o in zip(axes, off):
                    q[a] += o
                pts.append((off, float(u[tuple(q)])))
            total += 1
            if any(np.isnan(v) for _, v in pts):
                skipped += 1
                continue
            bad = False
            for off, v in pts:
                anti = tuple(1 - o for o in off)
                if off > anti:
                    continue
                va = dict(pts)[anti]
                rest = [w for o, w in pts if o not in (off, anti)]
                lo_i, hi_i = min(v, va), max(v, va)
                if not (lo_i <= max(rest) and min(rest) <= hi_i):
                    bad = True
            viol += bad
    return {"violations": int(viol), "nan_skipped": int(skipped), "total_blocks": int(total)}


# ---------------------------------------------------------------- Round-0 assembly / seam predicates
def split_domain_1d(total: int, K: int):
    if not 1 <= K <= total:
        raise ValueError(f"need 1<=K<=D (K={K}, D={total})")
    base, rem = divmod(total, K)
    out, cur = [], 0
    for i in range(K):
        sz = base + (1 if i < rem else 0); out.append((cur, cur + sz)); cur += sz
    return out


def tile_windows(D: int, K: int, delta: int = 1):
    """[(z0, z1, z0e, z1e)] for the z-slab decomposition with halo delta."""
    return [(z0, z1, max(0, z0 - delta), min(D, z1 + delta)) for z0, z1 in split_domain_1d(D, K)]


def owner_map(D: int, K: int):
    """owner[g] for g in 0..2D-2 from the stated rule (independent of the repo assembly code)."""
    sl = split_domain_1d(D, K)
    own = np.full(2 * D - 1, -1, np.int64)
    for k, (z0, z1) in enumerate(sl):
        g0 = 0 if k == 0 else 2 * z0
        g1 = 2 * D - 2 if k == K - 1 else 2 * z1 - 1
        own[g0:g1 + 1] = k
    return own


def assemble_round0(tile_interiors, D: int, K: int, delta: int = 1) -> np.ndarray:
    """Assemble B from per-tile interior outputs (shape (2W-1,2H-1,2*(z1e-z0e)-1)) by owner map."""
    wins = tile_windows(D, K, delta)
    own = owner_map(D, K)
    if (own < 0).any():
        raise AssertionError("owner map has holes")
    t0 = tile_interiors[0]
    B = np.empty(t0.shape[:2] + (2 * D - 1,), np.float32)
    for g, k in enumerate(own):
        B[:, :, g] = tile_interiors[k][:, :, g - 2 * wins[k][2]]
    return B


def seam_predicates(B: np.ndarray, lo_glob: np.ndarray, hi_glob: np.ndarray, D: int, K: int):
    """Per-seam column predicates on B and the global interval field (interior, unpadded)."""
    res = {"seam_mono": 0, "seam_in_interval": 0, "seams": 0}
    for z0, z1 in split_domain_1d(D, K)[:-1]:
        p1, p2 = 2 * z1 - 1, 2 * z1
        res["seams"] += 1
        res["seam_mono"] += int((B[:, :, p1] > B[:, :, p2]).sum())
    return res


def bits_equal(a, b) -> bool:
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32)
    return a.shape == b.shape and bool(np.array_equal(np.ascontiguousarray(a).view(np.uint32), np.ascontiguousarray(b).view(np.uint32)))
