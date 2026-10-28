"""Output-fidelity comparison (contract items 4-7) for A/B/C fields of ONE input.  CPU/NumPy/SciPy only, bounded memory, deterministic.
A different output is a valid representation difference, NOT a defect (defects = items 1-3, checked per record).
usage (library): compare_group(vol, fields={'A':u,'B16':u,'C16':u}, Ks={'B16':16,'C16':16}, thresholds=True) -> dict;  CLI: --selftest"""
import json
import sys
from pathlib import Path

import numpy as np

import r8f1l_lib as L
from r8f1l_lib import O

DBINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 8), (9, 10 ** 9)]            # expanded-plane distance d to the nearest seam (d = min_b |g - 2 z_b|), F1K sec.5 item 6
THRESH = list(range(32, 256, 32))


def seam_distance(D, K):
    g = np.arange(2 * D - 1)
    zb = np.array([z1 for _, z1 in O.split_domain_1d(D, K)[:-1]], dtype=np.int64)
    return np.abs(g[:, None] - 2 * zb[None, :]).min(axis=1) if len(zb) else np.full(2 * D - 1, 10 ** 9)


def bits(a):
    return np.ascontiguousarray(a, np.float32).view(np.uint32)


def pair_stats(X, Y, dplane, keep=10 ** 6):
    """Bitwise difference of two fields, concentration near seams, magnitude and sign. dplane: per expanded z-plane distance to the nearest seam."""
    assert X.shape == Y.shape and X.dtype == Y.dtype == np.float32
    diff = bits(X) != bits(Y)
    n = int(diff.sum())
    out = {'cells': int(diff.size), 'n_diff': n, 'frac_diff': n / diff.size, 'bit_identical': n == 0}
    nz = np.where(dplane <= 4)[0]                                                  # H4(ii): deviation over ALL cells of the seam slab d<=4 (identical cells count as 0)
    if len(nz):
        sl = np.abs(X[:, :, nz].astype(np.float64) - Y[:, :, nz].astype(np.float64))
        out.update(near_d4_cells=int(sl.size), near_d4_mean_abs=float(sl.mean()), near_d4_max_abs=float(sl.max()))
    if n == 0:
        return out, np.zeros((0, 3), np.int32)
    pp = diff.sum(axis=(0, 1)).astype(np.int64)
    out['by_distance'] = {f'{lo}-{hi}' if hi < 10 ** 9 else f'>{lo - 1}': int(pp[(dplane >= lo) & (dplane <= hi)].sum()) for lo, hi in DBINS}
    near = (dplane <= 4)
    f_obs, f_uni = float(pp[near].sum() / n), float(near.mean())
    out.update(frac_diff_within_d4=f_obs, uniform_expectation_d4=f_uni, concentration_ratio_d4=(f_obs / f_uni if f_uni > 0 else None))
    dx = X[diff].astype(np.float64) - Y[diff].astype(np.float64)
    q = np.quantile(np.abs(dx), [0, .5, .9, .99, 1])
    out.update(abs_delta_quantiles_min_med_p90_p99_max=[float(v) for v in q], frac_X_lower=float((dx < 0).mean()), mean_signed_delta=float(dx.mean()))
    coords = np.argwhere(diff)
    out['coords_kept'] = int(min(len(coords), keep))
    return out, coords[:keep].astype(np.int32)


def interval_position_hist(u, vol, bins=10, chunk=16):
    """Item 5: where non-original cells sit inside their ISpan interval [lo,hi] (position (u-lo)/(hi-lo)); counts per decile + degenerate/endpoint cells.
    Eight parity classes cover each expanded cell once; lo/hi come from the original corner voxels (independent of the solver)."""
    hist = np.zeros(bins, np.int64); degenerate = at_lo = at_hi = total = 0
    dev_sum = dev_max = 0.0                                         # deviation from the mean of the cell's original corner voxels (reference choice: frozen, see OUTPUT_FIDELITY)
    for px in (0, 1):
        for py in (0, 1):
            for pz in (0, 1):
                if (px, py, pz) == (0, 0, 0):
                    continue
                nx, ny, nz = (vol.shape[i] - p for i, p in enumerate((px, py, pz)))
                for z in range(0, nz, chunk):
                    e = min(nz, z + chunk)
                    lo = vol[:nx, :ny, z:e].astype(np.float32).copy(); hi = lo.copy(); sm = np.zeros_like(lo, dtype=np.float64)
                    for dx in range(px + 1):
                        for dy in range(py + 1):
                            for dz in range(pz + 1):
                                c = vol[dx:dx + nx, dy:dy + ny, z + dz:e + dz].astype(np.float32)
                                np.minimum(lo, c, out=lo); np.maximum(hi, c, out=hi); sm += c
                    part = u[px:2 * nx:2, py:2 * ny:2, 2 * z + pz:2 * e:2]
                    w = hi - lo
                    total += part.size; degenerate += int((w == 0).sum())
                    dv = np.abs(part.astype(np.float64) - sm / ((px + 1) * (py + 1) * (pz + 1)))
                    dev_sum += float(dv.sum()); dev_max = max(dev_max, float(dv.max()))
                    m = w > 0
                    pos = (part[m] - lo[m]) / w[m]
                    at_lo += int((pos == 0).sum()); at_hi += int((pos == 1).sum())
                    hist += np.bincount(np.minimum((pos * bins).astype(np.int64), bins - 1), minlength=bins)
    return {'non_original_cells': int(total), 'degenerate_interval': int(degenerate), 'at_lo': int(at_lo), 'at_hi': int(at_hi), 'decile_counts': [int(h) for h in hist],
            'mean_abs_dev_from_corner_mean': dev_sum / max(total, 1), 'max_abs_dev_from_corner_mean': dev_max}


def component_stats(u, dplane, thresholds=THRESH):
    """Item 7: connected components of {u >= t} (6- and 26-connectivity), split by the seam slab d<=4: local (entirely inside), far (entirely outside), straddling."""
    from scipy import ndimage as ndi
    slab = dplane <= 4
    res = {}
    for t in thresholds:
        m = u >= t
        row = {'voxels': int(m.sum())}
        for name, st in (('conn6', ndi.generate_binary_structure(3, 1)), ('conn26', np.ones((3, 3, 3), bool))):
            lab, n = ndi.label(m, structure=st)
            tot = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            ins = np.bincount(lab[:, :, slab].ravel(), minlength=n + 1)[1:] if slab.any() else np.zeros(n, np.int64)
            row[name] = {'n': int(n), 'local_in_slab': int(((tot == ins) & (tot > 0)).sum()), 'far_from_slab': int((ins == 0).sum()), 'straddling': int(((ins > 0) & (ins < tot)).sum())}
            del lab
        res[str(t)] = row
    return res


def compare_group(vol, fields, Ks, thresholds=True):
    """fields: label -> host float32 field of ONE input (labels 'A', 'B<K>', 'C<K>'); Ks: label -> K for B/C."""
    D = vol.shape[2]
    out = {'pairs': {}, 'position': {}, 'components': {}}
    coords = {}
    labels = sorted(fields)
    for lab in labels:
        out['position'][lab] = interval_position_hist(fields[lab], vol)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            Kp = Ks.get(a) or Ks.get(b)
            if Ks.get(a) and Ks.get(b) and Ks[a] != Ks[b] and not {a, b} <= set(Ks):
                continue
            if Ks.get(a) and Ks.get(b) and Ks[a] != Ks[b]:
                continue                                           # different-K B/C pairs are not part of the registered comparison
            s, c = pair_stats(fields[a], fields[b], seam_distance(D, Kp))
            out['pairs'][f'{a}__{b}'] = s
            coords[f'{a}__{b}'] = c
    if thresholds:
        for lab in labels:
            out['components'][lab] = component_stats(fields[lab], seam_distance(D, Ks.get(lab) or next(iter(Ks.values()), 2)))
    return out, coords


def selftest():
    """Planted-difference test: differences injected only at planes d<=4 must be reported as fully concentrated; random ones must not."""
    rng = np.random.default_rng(1)
    vol = rng.integers(0, 256, (6, 6, 24), dtype=np.uint8)
    X = np.zeros((11, 11, 119), np.float32); Y = X.copy()
    dpl = seam_distance(60, 4)                                    # band d<=4 covers 27/119 planes: a ratio >= 2 is attainable (it is NOT for thin slabs / large K, see freeze P-6)
    near = np.where(dpl <= 4)[0]
    for z in near[::2]:
        Y[:, 3, z] += 1
    s, c = pair_stats(X, Y, dpl)
    ok1 = s['n_diff'] == 11 * len(near[::2]) and s['frac_diff_within_d4'] == 1.0 and s['concentration_ratio_d4'] > 2 and abs(s['concentration_ratio_d4'] - 119 / 27) < 1e-9
    Y2 = X.copy(); far = np.where(dpl > 8)[0]
    Y2[2, 2, far[0]] += 1; Y2[rng.integers(0, 11, 300), rng.integers(0, 11, 300), rng.integers(0, 119, 300)] += 1
    s2, _ = pair_stats(X, Y2, dpl)
    ok2 = s2['concentration_ratio_d4'] < 2 and not s2['bit_identical']
    s3, _ = pair_stats(X, X.copy(), dpl)
    u = np.repeat(np.repeat(np.repeat(vol, 2, 0)[:-1], 2, 1)[:, :-1], 2, 2)[:, :, :-1].astype(np.float32)          # NOT a valid field; only for shape/plumbing
    hp = interval_position_hist(np.ascontiguousarray(u), vol)
    comp = component_stats(u, seam_distance(24, 4)[:u.shape[2]] if u.shape[2] <= 47 else dpl, thresholds=[64, 128])
    return {'planted_near_detected': bool(ok1), 'random_not_concentrated': bool(ok2), 'identical_detected': bool(s3['bit_identical']), 'position_plumbing': hp['non_original_cells'] > 0, 'components_plumbing': comp['64']['conn6']['n'] >= 0,
            'pass': bool(ok1 and ok2 and s3['bit_identical'] and hp['non_original_cells'] > 0)}


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        r = selftest(); print(json.dumps(r, indent=1)); sys.exit(0 if r['pass'] else 1)
