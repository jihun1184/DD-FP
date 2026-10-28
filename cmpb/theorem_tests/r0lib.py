"""Independent re-implementation used ONLY to re-check the proof steps of Theorem R0.

Nothing here is taken from the frozen project code: the first-discovery front propagation (FP) is
written from the contract P1-P5 alone, the DWC checker is written from the threshold-form definition
and is itself cross-checked against a brute-force block enumerator (see check_checker in r0_verify.py).

Notation: lattice cells are triples of ints; voxel v <-> cell 2v; star(2i)={i}, star(2i+1)={i,i+1}.
"""
from collections import defaultdict
import itertools
import numpy as np

# ----------------------------------------------------------------------------- interval map
def _expand(a, axis, fn):
    n = a.shape[axis]
    out_shape = list(a.shape); out_shape[axis] = 2 * n - 1
    out = np.empty(out_shape, dtype=a.dtype)
    sl = lambda s: tuple(s if i == axis else slice(None) for i in range(a.ndim))
    out[sl(slice(0, None, 2))] = a
    if n > 1:
        lo_part = np.take(a, range(0, n - 1), axis=axis)
        hi_part = np.take(a, range(1, n), axis=axis)
        out[sl(slice(1, None, 2))] = fn(lo_part, hi_part)
    return out

def ispan(u):
    """lo(c)=min over star(c) of u, hi(c)=max over star(c); min/max are separable over the product star."""
    lo, hi = u.astype(np.int64), u.astype(np.int64)
    for ax in range(3):
        lo = _expand(lo, ax, np.minimum)
        hi = _expand(hi, ax, np.maximum)
    return lo, hi

def pad_frame(a, alpha=0):
    return np.pad(a, 1, mode="constant", constant_values=alpha)

# ----------------------------------------------------------------------------- DWC checker
ANTIPODE = [((0, 0, 0), (1, 1, 1)), ((0, 0, 1), (1, 1, 0)), ((0, 1, 0), (1, 0, 1)), ((0, 1, 1), (1, 0, 0))]

def _corner(X, i, j, k):
    nx, ny, nz = X.shape
    return X[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k]

def critical_blocks(X):
    """Number of critical face blocks (xy,xz,yz) and cube blocks of the boolean array X (threshold form)."""
    X = X.astype(bool)
    nf = 0
    # faces: block = 2x2 in plane (a,b), third coordinate fixed
    for (a, b) in ((0, 1), (0, 2), (1, 2)):
        if X.shape[a] < 2 or X.shape[b] < 2:
            continue
        def c(i, j):
            sl = [slice(None)] * 3
            sl[a] = slice(i, X.shape[a] - 1 + i)
            sl[b] = slice(j, X.shape[b] - 1 + j)
            return X[tuple(sl)]
        p00, p01, p10, p11 = c(0, 0), c(0, 1), c(1, 0), c(1, 1)
        n = p00.astype(int) + p01 + p10 + p11
        crit = (n == 2) & ((p00 & p11) | (p01 & p10))
        nf += int(crit.sum())
    nc = 0
    if min(X.shape) >= 2:
        cs = {(i, j, k): _corner(X, i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
        n = sum(v.astype(int) for v in cs.values())
        crit = np.zeros(n.shape, bool)
        for p, q in ANTIPODE:
            crit |= (n == 2) & cs[p] & cs[q]
            crit |= (n == 6) & (~cs[p]) & (~cs[q])
        nc = int(crit.sum())
    return nf, nc

def dwc_violations(f):
    """(#critical face blocks, #critical cube blocks) summed over all thresholds t in values(f)."""
    nf = nc = 0
    for t in np.unique(f):
        a, b = critical_blocks(f >= t)
        nf += a; nc += b
    return nf, nc

def brute_critical(X):
    """Slow block enumerator written separately (loops), used to validate critical_blocks."""
    X = X.astype(bool); nx, ny, nz = X.shape
    nf = nc = 0
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                for (a, b) in ((0, 1), (0, 2), (1, 2)):
                    d = [0, 0, 0]
                    pts = []
                    for i in (0, 1):
                        for j in (0, 1):
                            q = [x, y, z]; q[a] += i; q[b] += j
                            pts.append(tuple(q))
                    if all(0 <= q[t] < s for q in pts for t, s in enumerate((nx, ny, nz))):
                        v = [X[q] for q in pts]  # order 00,01,10,11
                        if sum(v) == 2 and ((v[0] and v[3]) or (v[1] and v[2])):
                            nf += 1
                pts = [(x + i, y + j, z + k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
                if all(q[0] < nx and q[1] < ny and q[2] < nz for q in pts):
                    S = {q: X[q] for q in pts}
                    ins = {q for q in pts if S[q]}
                    for q in pts:
                        anti = (2 * x + 1 - q[0], 2 * y + 1 - q[1], 2 * z + 1 - q[2])
                        if q < anti:
                            if ins == {q, anti} or ins == set(pts) - {q, anti}:
                                nc += 1
    return nf, nc

# ----------------------------------------------------------------------------- FP with an admissible-run scheduler
_NB = {}
def neighbours(shape):
    if shape in _NB:
        return _NB[shape]
    nx, ny, nz = shape
    idx = np.arange(nx * ny * nz).reshape(shape)
    nb = [[] for _ in range(nx * ny * nz)]
    for ax in range(3):
        a = np.take(idx, range(0, shape[ax] - 1), axis=ax).ravel()
        b = np.take(idx, range(1, shape[ax]), axis=ax).ravel()
        for p, q in zip(a.tolist(), b.tolist()):
            nb[p].append(q); nb[q].append(p)
    _NB[shape] = nb
    return nb

class Run:
    __slots__ = ("uhat", "ldisc", "order", "levels", "root", "shape", "branches")

def fp_run(lo, hi, root, l_inf, rng, sched="rand", within="rand", forced=None):
    """One run of first-discovery FP satisfying P1-P5 (sched in {'rand','near','bad'}).

    P1: a cell is reserved at clamp(level at discovery, lo, hi) and processed at that level.
    P2: the level changes only when its bucket is empty.
    P3: a level change goes to the nearest occupied level above or below (never skipping an occupied level).
        'rand' picks above/below arbitrarily (the weakest contract), 'near' picks the nearer (ties random),
        'bad' violates P3 (jumps to a random non-nearest occupied level when one exists).
    P4: processing a cell discovers all undiscovered 6-neighbours.  P5: start level l_inf at `root`.
    `forced`: optional list of 0/1 decisions (0=down,1=up) for exhaustive enumeration; consumed in order.
    """
    shape = lo.shape
    nb = neighbours(shape)
    L, H = lo.ravel(), hi.ravel()
    n = L.size
    disc = np.zeros(n, bool)
    uhat = np.full(n, -1, np.int64)
    ldisc = np.full(n, -1, np.int64)
    buckets = defaultdict(list)
    disc[root] = True; ldisc[root] = l_inf
    buckets[l_inf].append(root)
    cur = l_inf
    order, levels = [], []
    branches = 0
    done = 0
    while done < n:
        b = buckets[cur]
        if b:
            if within == "rand":
                i = rng.integers(len(b)); b[i], b[-1] = b[-1], b[i]
            c = b.pop()
            uhat[c] = cur; order.append(c); levels.append(cur); done += 1
            for q in nb[c]:
                if not disc[q]:
                    disc[q] = True; ldisc[q] = cur
                    s = min(max(cur, L[q]), H[q])
                    buckets[s].append(q)
        else:
            occ = sorted(k for k, v in buckets.items() if v)
            if not occ:
                break
            below = [k for k in occ if k < cur]; above = [k for k in occ if k > cur]
            if sched == "bad":
                far = [k for k in occ if (below and k < below[-1]) or (above and k > above[0])]
                cur = far[rng.integers(len(far))] if far else (below[-1] if below else above[0])
            else:
                if not below: cur = above[0]
                elif not above: cur = below[-1]
                else:
                    if forced is not None:
                        go_up = bool(forced[branches]) if branches < len(forced) else False
                    elif sched == "near":
                        du, dd = above[0] - cur, cur - below[-1]
                        go_up = (du < dd) if du != dd else bool(rng.integers(2))
                    else:
                        go_up = bool(rng.integers(2))
                    branches += 1
                    cur = above[0] if go_up else below[-1]
    r = Run(); r.uhat = uhat.reshape(shape); r.ldisc = ldisc.reshape(shape); r.order = order
    r.levels = levels; r.root = root; r.shape = shape
    r.branches = branches
    return r

# ----------------------------------------------------------------------------- admissibility validator (reads only a trace)
def validate_admissible(run, lo, hi, l_inf):
    """Re-derive P1-P5 from the trace.  Returns list of violated contract items."""
    shape = lo.shape; nb = neighbours(shape)
    L, H = lo.ravel(), hi.ravel()
    n = L.size
    bad = set()
    if len(run.order) != n: bad.add("P5-termination")
    if run.order[0] != run.root: bad.add("P5-root")
    if run.levels[0] != l_inf: bad.add("P5-start")
    disc_time = {run.root: 0}
    reserved = {run.root: l_inf}
    pending = defaultdict(int); pending[l_inf] = 1
    prev = None
    for t, (c, lev) in enumerate(zip(run.order, run.levels)):
        if c not in reserved: bad.add("P1-undiscovered-processed"); continue
        if reserved[c] != lev: bad.add("P1-level")
        if prev is not None and lev != prev:
            if pending[prev] > 0: bad.add("P2")
            lo_, hi_ = min(prev, lev), max(prev, lev)
            if any(v > 0 and lo_ < k < hi_ for k, v in pending.items()): bad.add("P3")
        pending[lev] -= 1
        for q in nb[c]:
            if q not in reserved:
                s = min(max(lev, L[q]), H[q]); reserved[q] = s; pending[s] += 1
        prev = lev
    return sorted(bad)
