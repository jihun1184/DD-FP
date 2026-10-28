"""DWC predicate on a scalar field over the expanded lattice (numpy only).
2-D face-blocks: all 2x2 neighbourhoods in xy, xz, yz planes. 3-D blocks: 2x2x2 cubes with the
Boutry-style antagonist-vertex condition checked via 2-D faces only (see P1_THEOREM audit)."""
import numpy as np

def _viol(a, b, c, d):
    # block corners a=(0,0) d=(1,1) antagonists; b,c the other pair. Violation iff intervals disjoint.
    lo1, hi1 = np.minimum(a, d), np.maximum(a, d)
    lo2, hi2 = np.minimum(b, c), np.maximum(b, c)
    return (hi1 < lo2) | (hi2 < lo1)

def face_violations(v):
    """Boolean arrays per orientation: True where the 2x2 face-block anchored at that cell violates DWC."""
    if v.ndim != 3: raise ValueError("expect 3-D")
    out = {}
    out['xy'] = _viol(v[:-1,:-1,:], v[1:,:-1,:], v[:-1,1:,:], v[1:,1:,:])
    out['xz'] = _viol(v[:-1,:,:-1], v[1:,:,:-1], v[:-1,:,1:], v[1:,:,1:])
    out['yz'] = _viol(v[:,:-1,:-1], v[:,1:,:-1], v[:,:-1,1:], v[:,1:,1:])
    return out

def count_violations(v):
    return {k: int(m.sum()) for k, m in face_violations(v).items()}

def cube_violations(v):
    """k=3 blocks (Boutry-Najman-Geraud JMIV 2020, Prop.3): for each 2x2x2 block and each of the 4 antagonist
    vertex pairs, intvl(u(p),u(p')) must meet Span of the other 6 vertices. Returns per-block violation mask."""
    if v.ndim != 3: raise ValueError("expect 3-D")
    c = {(i,j,k): v[i:v.shape[0]-1+i, j:v.shape[1]-1+j, k:v.shape[2]-1+k] for i in (0,1) for j in (0,1) for k in (0,1)}
    pairs = [((0,0,0),(1,1,1)),((1,0,0),(0,1,1)),((0,1,0),(1,0,1)),((0,0,1),(1,1,0))]
    bad = np.zeros(c[(0,0,0)].shape, bool)
    for p, q in pairs:
        rest = [c[t] for t in c if t not in (p, q)]
        rlo = np.minimum.reduce(rest); rhi = np.maximum.reduce(rest)
        lo, hi = np.minimum(c[p], c[q]), np.maximum(c[p], c[q])
        bad |= (hi < rlo) | (rhi < lo)
    return bad

def nd_violations(v):
    """(k=2 faces total, k=3 cubes total)"""
    return sum(count_violations(v).values()), int(cube_violations(v).sum())
