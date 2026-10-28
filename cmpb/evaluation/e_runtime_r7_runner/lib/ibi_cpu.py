"""CPU port of scripts/ddfp/experiment_DDFP_all.py::run_ibi_v10 (corrected tree, sha b6a9b109...) for tiny volumes.
Backends: build_ispan_gpu -> cpu_fp.build_ispan_cpu ; front_propagation_gpu(seeds) -> seedref.seed_aware_cpu_reference.
Semantics copied line-by-line (pin ranges, assembly order, global max_change<0.5 stop)."""
import numpy as np
from cpu_fp import build_ispan_cpu
from seedref import seed_aware_cpu_reference

def split_domain_1d(total, K):
    base, rem = divmod(total, K); out, cur = [], 0
    for i in range(K):
        sz = base + (1 if i < rem else 0); out.append((cur, cur + sz)); cur += sz
    return out

def _pin(lo, hi, seeds, asm, sub_z_start, g0, g1, W2, H2):
    n = g1 - g0
    if n <= 0: return
    ps = (g0 - 2 * sub_z_start) + 1
    n_g = min(n, asm.shape[2] - g0); n_l = min(n, lo.shape[2] - ps); a = min(n_g, n_l)
    if a <= 0: return
    v = asm[:W2, :H2, g0:g0 + a]
    lo[1:W2+1, 1:H2+1, ps:ps+a] = v; hi[1:W2+1, 1:H2+1, ps:ps+a] = v; seeds[1:W2+1, 1:H2+1, ps:ps+a] = v

def _asm_bdry(tgt, sub, zb, z0e, D2):
    l = zb - z0e; e0, g0 = 2*l, 2*zb; e1, g1 = e0-1, g0-1
    if 0 <= e0 < sub.shape[2] and g0 < D2: tgt[:, :, g0] = sub[:, :, e0]
    if 0 <= e1 < sub.shape[2] and g1 >= 0: tgt[:, :, g1] = sub[:, :, e1]

def _place(tgt, sub, k, K, z0, z1, z0e):
    il0 = z0 - z0e; il1 = il0 + (z1 - z0)
    es = 2*il0 if k > 0 else 0
    ee = 2*il1 - 1 if k < K-1 else sub.shape[2]
    gs = 2*z0 if k > 0 else 0
    ge = min(gs + (ee - es), tgt.shape[2]); a = ge - gs
    if a > 0 and es + a <= sub.shape[2]: tgt[:, :, gs:ge] = sub[:, :, es:es+a]

def ibi(vol, K, delta, max_rounds=8, tie='lower', keep_tiles=False):
    W, H, D = vol.shape; W2, H2, D2 = 2*W-1, 2*H-1, 2*D-1
    sl = split_domain_1d(D, K); bz = [z1 for _, z1 in sl[:-1]]
    asm = np.zeros((W2, H2, D2), np.float32); meta = []; hist = []; tiles = {}
    for k, (z0, z1) in enumerate(sl):
        z0e = max(0, z0 - delta); z1e = min(D, z1 + delta)
        lo, hi, l = build_ispan_cpu(vol[:, :, z0e:z1e])
        sub = seed_aware_cpu_reference(lo, hi, l, None, tie)[0][1:-1, 1:-1, 1:-1]
        tiles[(0, k)] = sub.copy()
        _place(asm, sub, k, K, z0, z1, z0e)
        if k < K-1: _asm_bdry(asm, sub, z1, z0e, D2)
        meta.append((k, z0, z1, z0e, z1e))
    hist.append(asm.copy()); R = 0; mc = None
    for r in range(1, max_rounds+1):
        prev = asm.copy(); new = asm.copy()
        for (k, z0, z1, z0e, z1e) in meta:
            lo, hi, l = build_ispan_cpu(vol[:, :, z0e:z1e]); seeds = np.full(lo.shape, np.nan, np.float32)
            if k > 0:   _pin(lo, hi, seeds, asm, z0e, 2*z0e, 2*z0+1, W2, H2)
            if k < K-1: _pin(lo, hi, seeds, asm, z0e, 2*z1, 2*z1e, W2, H2)
            sub = seed_aware_cpu_reference(lo, hi, l, seeds, tie)[0][1:-1, 1:-1, 1:-1]
            tiles[(r, k)] = sub.copy()
            _place(new, sub, k, K, z0, z1, z0e)
            if k < K-1: _asm_bdry(new, sub, z1, z0e, D2)
        mc = float(np.abs(new - prev).max()); asm = new; R = r; hist.append(asm.copy())
        if mc < 0.5: break
    return dict(u=asm, R=R, bz=bz, hist=hist, tiles=tiles if keep_tiles else None, meta=meta)

def seam_viol(u, bz):
    """repo count_boundary_violations (xz,yz faces in layers [2zb-1,2zb] and [2zb,2zb+1])."""
    D2 = u.shape[2]; tot = 0
    for zb in bz:
        for ez in (2*zb-1, 2*zb):
            if ez < 0 or ez+1 >= D2: continue
            for ax in (0, 1):
                if ax == 0: a, d, b, c = u[:-1,:,ez], u[1:,:,ez+1], u[1:,:,ez], u[:-1,:,ez+1]
                else:       a, d, b, c = u[:,:-1,ez], u[:,1:,ez+1], u[:,1:,ez], u[:,:-1,ez+1]
                tot += int(((np.maximum(a,d) < np.minimum(b,c)) | (np.maximum(b,c) < np.minimum(a,d))).sum())
    return tot
