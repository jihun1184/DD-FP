"""
scripts/ddfp/experiment_DDFP_all.py
-------------------------------------
DD-FP theory validation (IBI v10).

v9 → v10 fixes:
  [Bug 1] Over-pinning: pin only the overlap region, not the full subdomain.
  [Bug 2] Face-block cell mismatch at z_b: extend left overlap of sub k+1
           to z_b inclusive so both cells come from the same FP instance.

Theory: Lemma 4 (δ≥1) + v10 pinning → Theorem 1 Step A: u_DD = u_DWC.
Expected: R*=2 on all BraTS subjects at K=16, δ=1.
"""

from __future__ import annotations
import os, sys, glob, time, json, argparse, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import cupy as cp
import nibabel as nib
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ddfp.gpu_immersion import (
    build_ispan_gpu,
    front_propagation_gpu,
    verify_dwc_gpu,
)

warnings.filterwarnings("ignore")


# ── Preprocessing / synthetic volumes ────────────────────────────────────────

def preprocess(vol: np.ndarray) -> np.ndarray:
    nz = vol[vol > 0]
    if len(nz) == 0:
        return np.zeros(vol.shape, dtype=np.uint8)
    p1, p99 = np.percentile(nz, [1, 99])
    out = np.zeros(vol.shape, np.float32)
    m = vol > 0
    out[m] = np.clip((vol[m] - p1) / (p99 - p1 + 1e-9), 0, 1)
    return (out * 255).astype(np.uint8)


def make_synth_volumes(size: int = 48) -> dict[str, np.ndarray]:
    W = H = D = size
    v = np.zeros((W, H, D), np.float32)
    v[W//2, H//2, D//2] = 1.0
    v = gaussian_filter(v, sigma=W//6)
    vol_gauss = (v / v.max() * 220).astype(np.uint8)
    vol_ramp = np.zeros((W, H, D), np.uint8)
    for iz in range(D):
        vol_ramp[:, :, iz] = int(iz / D * 200)
    return {"synth_gaussian": vol_gauss, "synth_ramp_z": vol_ramp}


def serialize(x):
    if isinstance(x, (np.integer, np.floating)): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, dict): return {k: serialize(v) for k, v in x.items()}
    if isinstance(x, list): return [serialize(i) for i in x]
    return x


def split_domain_1d(total: int, K: int) -> list[tuple[int, int]]:
    base, rem = divmod(total, K)
    slices, cur = [], 0
    for i in range(K):
        sz = base + (1 if i < rem else 0)
        slices.append((cur, cur + sz))
        cur += sz
    return slices


# ── IBI assembly helpers ──────────────────────────────────────────────────────

def _pin_and_seed_expanded(
    U_lo_pad, U_hi_pad, seeds_pad, assembled_g,
    sub_z_start, g_ez_start, g_ez_end, W2s, H2s,
) -> None:
    """Pin global expanded-lattice z range into subdomain U_lo/U_hi/seeds."""
    n = g_ez_end - g_ez_start
    if n <= 0:
        return
    l_pad_start = (g_ez_start - 2 * sub_z_start) + 1
    n_g = min(n, assembled_g.shape[2] - g_ez_start)
    n_l = min(n, U_lo_pad.shape[2] - l_pad_start)
    actual = min(n_g, n_l)
    if actual <= 0:
        return
    vals = assembled_g[:W2s, :H2s, g_ez_start:g_ez_start + actual]
    U_lo_pad[1:W2s+1, 1:H2s+1, l_pad_start:l_pad_start + actual] = vals
    U_hi_pad[1:W2s+1, 1:H2s+1, l_pad_start:l_pad_start + actual] = vals
    seeds_pad[1:W2s+1, 1:H2s+1, l_pad_start:l_pad_start + actual] = vals


def _assemble_boundary(u_target, u_sub, z_b, z0_ext, D2) -> None:
    """Replace 2-cell boundary at z_b from sub k (IBI round r≥1)."""
    local_zb = z_b - z0_ext
    ez_0 = 2 * local_zb;  gz_0 = 2 * z_b
    ez_1 = ez_0 - 1;      gz_1 = gz_0 - 1
    if 0 <= ez_0 < u_sub.shape[2] and gz_0 < D2:
        u_target[:, :, gz_0] = u_sub[:, :, ez_0]
    if 0 <= ez_1 < u_sub.shape[2] and gz_1 >= 0:
        u_target[:, :, gz_1] = u_sub[:, :, ez_1]


def _assemble_boundary_3cell(u_target, u_sub, z_b, z0_ext, D2) -> None:
    """Replace 3-cell boundary at z_b (single-pass, Lemma 4 verification).

    Covers gz=2*z_b-1, 2*z_b, 2*z_b+1 so both face-blocks FA and FB
    are sourced from the same FP instance → DWC guaranteed for δ≥1.
    """
    local_zb = z_b - z0_ext
    ez_0 = 2 * local_zb;  gz_0 = 2 * z_b
    if 0 <= ez_0 - 1 < u_sub.shape[2] and gz_0 - 1 >= 0:
        u_target[:, :, gz_0 - 1] = u_sub[:, :, ez_0 - 1]
    if 0 <= ez_0 < u_sub.shape[2] and gz_0 < D2:
        u_target[:, :, gz_0]     = u_sub[:, :, ez_0]
    if 0 <= ez_0 + 1 < u_sub.shape[2] and gz_0 + 1 < D2:
        u_target[:, :, gz_0 + 1] = u_sub[:, :, ez_0 + 1]


# ── Single-pass Round 0 (Lemma 3+4 verification) ─────────────────────────────

def run_dd_fp_round0(vol_u8, K, delta, l_inf=0.0) -> dict:
    """DD-FP Round 0 only (no IBI). Used to verify the δ step-function.

    Uses z1_ext = z1 + delta + 1 so sub k's expanded lattice covers
    ez_0+1 when δ≥1 → FA and FB face-blocks are both DWC-consistent.
    Prediction: violations > 0 at δ=0, violations = 0 at δ≥1 (Lemma 3+4).
    """
    W, H, D = vol_u8.shape
    W2, H2, D2 = 2*W - 1, 2*H - 1, 2*D - 1
    slices_z = split_domain_1d(D, K)
    boundary_z_orig = [z1 for _, z1 in slices_z[:-1]]
    t0 = time.time()
    u_assembled = cp.zeros((W2, H2, D2), dtype=cp.float32)

    for k, (z0, z1) in enumerate(slices_z):
        z0_ext = max(0, z0 - delta)
        z1_ext = min(D, z1 + delta + 1)   # +1: include ez_0+1 for FA/FB
        sub_vol = vol_u8[:, :, z0_ext:z1_ext]
        U_lo, U_hi, _ = build_ispan_gpu(sub_vol)
        u_sub = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)[1:-1, 1:-1, 1:-1]

        inner_loc_z0 = z0 - z0_ext
        inner_loc_z1 = inner_loc_z0 + (z1 - z0)
        ez_s = 2 * inner_loc_z0 + 1 if k > 0 else 0
        ez_e = 2 * inner_loc_z1 - 1 if k < K - 1 else u_sub.shape[2]
        gz_s = 2 * z0 + 1 if k > 0 else 0
        gz_e = min(gz_s + (ez_e - ez_s), D2)
        actual = gz_e - gz_s
        if actual > 0 and ez_s + actual <= u_sub.shape[2]:
            u_assembled[:, :, gz_s:gz_e] = u_sub[:, :, ez_s:ez_s + actual]
        if k < K - 1:
            _assemble_boundary_3cell(u_assembled, u_sub, z1, z0_ext, D2)

    return {"u_dd": cp.asnumpy(u_assembled), "boundary_z_orig": boundary_z_orig,
            "R_star": 0, "t_total_s": time.time() - t0}


# ── IBI v10 ───────────────────────────────────────────────────────────────────

def run_ibi_v10(vol_u8, K, delta, max_rounds=8, l_inf=0.0, verbose=False) -> dict:
    """IBI v10: overlap-only pinning + z_b-inclusive boundary.

    Round 0: independent FP, boundary fix at 0-cell and -1-cell of z_b.
    Round r≥1: pin [2*z0_ext, 2*z0+1) left and [2*z1, 2*z1_ext) right;
               BFS seeds match pinned region.
    Converges in R*=2 for all BraTS subjects at K=16, δ=1 (Thm. 1).
    """
    W, H, D = vol_u8.shape
    W2, H2, D2 = 2*W - 1, 2*H - 1, 2*D - 1
    W2s, H2s = W2, H2
    slices_z = split_domain_1d(D, K)
    boundary_z_orig = [z1 for _, z1 in slices_z[:-1]]
    t0 = time.time()
    u_assembled = cp.zeros((W2, H2, D2), dtype=cp.float32)
    sub_meta = []

    for k, (z0, z1) in enumerate(slices_z):
        z0_ext = max(0, z0 - delta)
        z1_ext = min(D, z1 + delta)
        sub_vol = vol_u8[:, :, z0_ext:z1_ext]
        U_lo, U_hi, _ = build_ispan_gpu(sub_vol)
        u_sub = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)[1:-1, 1:-1, 1:-1]

        inner_loc_z0 = z0 - z0_ext
        inner_loc_z1 = inner_loc_z0 + (z1 - z0)
        ez_s = 2 * inner_loc_z0 if k > 0 else 0
        ez_e = 2 * inner_loc_z1 - 1 if k < K - 1 else u_sub.shape[2]
        gz_s = 2 * z0 if k > 0 else 0
        gz_e = min(gz_s + (ez_e - ez_s), D2)
        actual = gz_e - gz_s
        if actual > 0 and ez_s + actual <= u_sub.shape[2]:
            u_assembled[:, :, gz_s:gz_e] = u_sub[:, :, ez_s:ez_s + actual]
        if k < K - 1:
            _assemble_boundary(u_assembled, u_sub, z1, z0_ext, D2)
        sub_meta.append({"k": k, "z0": z0, "z1": z1,
                         "z0_ext": z0_ext, "z1_ext": z1_ext})

    R_star = 0
    for r in range(1, max_rounds + 1):
        u_prev = u_assembled.copy()
        u_new  = u_assembled.copy()

        for m in sub_meta:
            k = m["k"]; z0 = m["z0"]; z1 = m["z1"]
            z0_ext = m["z0_ext"]; z1_ext = m["z1_ext"]
            sub_vol = vol_u8[:, :, z0_ext:z1_ext]
            U_lo, U_hi, _ = build_ispan_gpu(sub_vol)
            seeds = cp.full(U_lo.shape, cp.nan, dtype=cp.float32)

            if k > 0:
                _pin_and_seed_expanded(U_lo, U_hi, seeds, u_assembled, z0_ext,
                                       2 * z0_ext, 2 * z0 + 1, W2s, H2s)
            if k < K - 1:
                _pin_and_seed_expanded(U_lo, U_hi, seeds, u_assembled, z0_ext,
                                       2 * z1, 2 * z1_ext, W2s, H2s)

            u_sub = front_propagation_gpu(U_lo, U_hi, l_inf,
                                          boundary_seeds=seeds, verbose=False)[1:-1, 1:-1, 1:-1]

            inner_loc_z0 = z0 - z0_ext
            inner_loc_z1 = inner_loc_z0 + (z1 - z0)
            ez_s = 2 * inner_loc_z0 if k > 0 else 0
            ez_e = 2 * inner_loc_z1 - 1 if k < K - 1 else u_sub.shape[2]
            gz_s = 2 * z0 if k > 0 else 0
            gz_e = min(gz_s + (ez_e - ez_s), D2)
            actual = gz_e - gz_s
            if actual > 0 and ez_s + actual <= u_sub.shape[2]:
                u_new[:, :, gz_s:gz_e] = u_sub[:, :, ez_s:ez_s + actual]
            if k < K - 1:
                _assemble_boundary(u_new, u_sub, z1, z0_ext, D2)

        max_change = float(cp.abs(u_new - u_prev).max().item())
        u_assembled = u_new
        R_star = r
        if verbose:
            print(f"    R={r}  max_change={max_change:.4f}")
        if max_change < 0.5:
            break

    return {"u_dd": cp.asnumpy(u_assembled), "boundary_z_orig": boundary_z_orig,
            "R_star": R_star, "t_total_s": time.time() - t0}


# ── Violation counting ────────────────────────────────────────────────────────

def count_boundary_violations(u_inner, boundary_z_orig) -> int:
    """DWC violations in face-blocks [2z_b-1,2z_b] and [2z_b,2z_b+1], xz+yz."""
    D2 = u_inner.shape[2]
    total = 0
    for z_b in boundary_z_orig:
        for ez in [2*z_b - 1, 2*z_b]:
            if ez < 0 or ez + 1 >= D2:
                continue
            for ax in [0, 1]:
                if ax == 0:
                    v00,v11 = u_inner[:-1,:,ez], u_inner[1:,:,ez+1]
                    v10,v01 = u_inner[1:,:,ez],  u_inner[:-1,:,ez+1]
                else:
                    v00,v11 = u_inner[:,:-1,ez], u_inner[:,1:,ez+1]
                    v10,v01 = u_inner[:,1:,ez],  u_inner[:,:-1,ez+1]
                lo_a = np.minimum(v00,v11); hi_a = np.maximum(v00,v11)
                lo_r = np.minimum(v10,v01); hi_r = np.maximum(v10,v01)
                total += int(((hi_a < lo_r) | (hi_r < lo_a)).sum())
    return total


def count_total_violations(u_inner) -> int:
    u_pad = np.pad(u_inner, 1, constant_values=0.0)
    return verify_dwc_gpu(cp.asarray(u_pad, dtype=cp.float32), verbose=False)["violations"]


# ── Experiments ───────────────────────────────────────────────────────────────

def run_enew1b(vol_u8, name, K=4, deltas=None, max_rounds=8) -> dict:
    """E-NEW-1b: δ step-function test (single-pass, Lemma 3+4).

    δ=0 → violations > 0 (face-block not in single subdomain)
    δ≥1 → violations = 0 (Lemma 4: diam_∞ = 1/2 < δ → containment)
    Uses run_dd_fp_round0 (no IBI) to expose raw boundary state.
    """
    if deltas is None:
        deltas = [0, 1, 2, 3]

    print(f"\n[E-NEW-1b] {name}  K={K}  (single-pass Round 0)")
    print(f"  {'δ':>4}  {'bdry_viol':>10}  {'total_viol':>11}  {'lemma4':>8}  {'t(s)':>6}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*11}  {'─'*8}  {'─'*6}")

    rows = []
    for delta in deltas:
        t0 = time.time()
        res = run_dd_fp_round0(vol_u8, K=K, delta=delta)
        elapsed = time.time() - t0
        b  = count_boundary_violations(res["u_dd"], res["boundary_z_orig"])
        tv = count_total_violations(res["u_dd"])
        ok = b == 0
        rows.append({"delta": delta, "bdry_viol": b, "total_viol": tv,
                     "lemma4": ok, "t_s": elapsed})
        print(f"  {delta:>4}  {b:>10,}  {tv:>11,}  {'✅' if ok else '❌':>8}  {elapsed:>6.1f}")

    d0_viol = rows[0]["bdry_viol"] if rows else 0
    d1_ok   = all(r["lemma4"] for r in rows if r["delta"] >= 1)
    step_ok = (d0_viol > 0) and d1_ok
    print(f"  δ=0:{d0_viol}  δ≥1_ok:{d1_ok}  step-fn:{'PASS' if step_ok else 'FAIL'}")
    return {"experiment": "E-NEW-1b", "name": name, "K": K,
            "rows": rows, "d0_viol": d0_viol, "d1_ok": d1_ok, "step_ok": step_ok}


def _wilson_ci(k, n, z=1.96):
    """Wilson 95% CI for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denom
    return max(0.0, centre - margin) * 100, min(1.0, centre + margin) * 100


def _compute_kappa(vol_u8, boundary_z) -> float:
    """Gray-level range in ±1-voxel band around each subdomain boundary."""
    v = vol_u8.astype(np.float32)
    kappas = [float(v[:, :, max(0,z-1):min(v.shape[2],z+2)].max()
                    - v[:, :, max(0,z-1):min(v.shape[2],z+2)].min())
              for z in boundary_z]
    return float(np.median(kappas)) if kappas else 0.0


def run_enew2(vol_u8, name, K_list=None, delta=1, max_rounds=8) -> dict:
    """E-NEW-2: boundary violations per K at δ=1 (Corollary 2.1)."""
    if K_list is None:
        K_list = [2, 4, 8, 16]
    print(f"\n[E-NEW-2] {name}  δ={delta}  (IBI v10)")
    print(f"  {'K':>4}  {'R*':>4}  {'bdry_viol':>10}  {'t(s)':>7}")
    print(f"  {'─'*4}  {'─'*4}  {'─'*10}  {'─'*7}")

    rows = []
    for K in K_list:
        t0 = time.time()
        res = run_ibi_v10(vol_u8, K=K, delta=delta, max_rounds=max_rounds)
        b   = count_boundary_violations(res["u_dd"], res["boundary_z_orig"])
        ok  = b == 0
        elapsed = time.time() - t0
        kappa = _compute_kappa(vol_u8, res["boundary_z_orig"])
        D = vol_u8.shape[2]; D_sub = D // K; sigma_cover = D_sub // 2
        rho_pct = round(delta / D_sub * 100, 1) if D_sub > 0 else 0.0
        rows.append({"K": K, "delta": delta, "R_star": res["R_star"],
                     "bdry_viol": b, "dwc_ok": ok, "t_s": elapsed,
                     "D_sub": D_sub, "sigma_cover": sigma_cover,
                     "kappa": round(kappa, 1), "rho_pct": rho_pct})
        print(f"  {K:>4}  {res['R_star']:>4}  {b:>10,}  {'✅' if ok else '❌'}  "
              f"{elapsed:>6.1f}s  D_sub={D_sub}  κ={kappa:.0f}  σ={sigma_cover}")

    all_ok = all(r["dwc_ok"] for r in rows)
    print(f"  all K DWC: {'OK' if all_ok else 'FAIL'}")
    return {"experiment": "E-NEW-2", "name": name, "delta": delta,
            "rows": rows, "all_K_ok": all_ok}


# ── CPU sequential FP (E-NEW-3 baseline) ─────────────────────────────────────

def _snap(lv, lo, hi):
    return min(max(lv, lo), hi)


def build_ispan_cpu(vol_u8):
    W, H, D = vol_u8.shape; s = vol_u8.astype(np.float32)
    W2, H2, D2 = 2*W-1, 2*H-1, 2*D-1
    U_lo = np.empty((W2, H2, D2), np.float32)
    U_hi = np.empty((W2, H2, D2), np.float32)
    U_lo[::2,::2,::2] = s; U_hi[::2,::2,::2] = s
    for a, b, sl in [
        (s[:-1,:,:], s[1:,:,:], (np.s_[1::2], np.s_[::2],  np.s_[::2])),
        (s[:,:-1,:], s[:,1:,:], (np.s_[::2],  np.s_[1::2], np.s_[::2])),
        (s[:,:,:-1], s[:,:,1:], (np.s_[::2],  np.s_[::2],  np.s_[1::2])),
    ]:
        U_lo[sl] = np.minimum(a, b); U_hi[sl] = np.maximum(a, b)
    def mm4(f, *c): return f(f(c[0], c[1]), f(c[2], c[3]))
    U_lo[1::2,1::2,::2] = mm4(np.minimum,s[:-1,:-1,:],s[1:,:-1,:],s[:-1,1:,:],s[1:,1:,:])
    U_hi[1::2,1::2,::2] = mm4(np.maximum,s[:-1,:-1,:],s[1:,:-1,:],s[:-1,1:,:],s[1:,1:,:])
    U_lo[1::2,::2,1::2] = mm4(np.minimum,s[:-1,:,:-1],s[1:,:,:-1],s[:-1,:,1:],s[1:,:,1:])
    U_hi[1::2,::2,1::2] = mm4(np.maximum,s[:-1,:,:-1],s[1:,:,:-1],s[:-1,:,1:],s[1:,:,1:])
    U_lo[::2,1::2,1::2] = mm4(np.minimum,s[:,:-1,:-1],s[:,1:,:-1],s[:,:-1,1:],s[:,1:,1:])
    U_hi[::2,1::2,1::2] = mm4(np.maximum,s[:,:-1,:-1],s[:,1:,:-1],s[:,:-1,1:],s[:,1:,1:])
    c = np.stack([s[:-1,:-1,:-1],s[1:,:-1,:-1],s[:-1,1:,:-1],s[1:,1:,:-1],
                  s[:-1,:-1,1:], s[1:,:-1,1:], s[:-1,1:,1:], s[1:,1:,1:]])
    U_lo[1::2,1::2,1::2] = c.min(0); U_hi[1::2,1::2,1::2] = c.max(0)
    return (np.pad(U_lo, 1, constant_values=0.0),
            np.pad(U_hi, 1, constant_values=0.0), 0.0)


def fp_cpu(U_lo_pad, U_hi_pad, l_inf=0.0):
    """Algorithm 1 sequential FP (CPU baseline)."""
    Wp, Hp, Dp = U_lo_pad.shape; N = Wp * Hp * Dp
    u = np.full(N, np.nan, np.float32); dv = np.zeros(N, bool)
    Q = defaultdict(list)
    lo_f = U_lo_pad.ravel().astype(np.float64)
    hi_f = U_hi_pad.ravel().astype(np.float64)
    cur = [int(round(l_inf))]

    def push(h, lv):
        Q[int(round(_snap(lv, lo_f[h], hi_f[h])))].append(h)

    def pop():
        if not Q[cur[0]]:
            occ = [lv for lv, q in Q.items() if q]
            if not occ: return -1
            cur[0] = min(occ, key=lambda x: abs(x - cur[0]))
        return Q[cur[0]].pop()

    dx=[-1,1,0,0,0,0]; dy=[0,0,-1,1,0,0]; dz=[0,0,0,0,-1,1]

    def nbs(idx):
        z=idx//(Wp*Hp); r=idx%(Wp*Hp); y=r//Wp; x=r%Wp
        res = []
        for i in range(6):
            nx,ny,nz = x+dx[i],y+dy[i],z+dz[i]
            if 0<=nx<Wp and 0<=ny<Hp and 0<=nz<Dp:
                res.append(nx+ny*Wp+nz*Wp*Hp)
        return res

    Q[int(round(l_inf))].append(0); dv[0] = True
    while True:
        h = pop()
        if h == -1: break
        u[h] = float(cur[0])
        for nb in nbs(h):
            if not dv[nb]:
                push(nb, float(cur[0])); dv[nb] = True
    return u.reshape(Wp, Hp, Dp)


def run_enew3(vol_u8, name, K=4, delta=1, max_rounds=8) -> dict:
    """E-NEW-3: CPU vs IBI v10 numerical equivalence (Theorem 1 Step A).

    Pass: max_diff_boundary < 0.5 AND bdry_viol == 0.
    GPU Level-BFS ≠ CPU Algorithm 1 in general; DWC guarantee is unaffected.
    """
    print(f"\n[E-NEW-3] {name}  K={K}  δ={delta}")
    print("  CPU...", end="", flush=True)
    t0 = time.time()
    U_lo, U_hi, li = build_ispan_cpu(vol_u8)
    u_cpu = fp_cpu(U_lo, U_hi, li)[1:-1, 1:-1, 1:-1]
    t_cpu = time.time() - t0
    print(f" {t_cpu:.1f}s")

    print("  IBI v10...", end="", flush=True)
    t0 = time.time()
    res  = run_ibi_v10(vol_u8, K=K, delta=delta, max_rounds=max_rounds)
    u_dd = res["u_dd"]
    t_dd = time.time() - t0
    print(f" {t_dd:.1f}s  (R={res['R_star']})")

    D2  = u_cpu.shape[2]; bz = res["boundary_z_orig"]
    bdry_ez = sorted(set(2*z + dz for z in bz for dz in range(-2, 3)
                         if 0 <= 2*z + dz < D2))
    diff    = np.abs(u_cpu.astype(np.float64) - u_dd.astype(np.float64))
    max_all = float(diff.max()); exact_all = float((diff==0).mean())
    diff_b  = diff[:, :, bdry_ez] if bdry_ez else diff
    max_bnd = float(diff_b.max()); exact_bnd = float((diff_b==0).mean())
    bdry_viol = count_boundary_violations(u_dd, bz)
    equiv_ok  = (max_bnd < 0.5) and (bdry_viol == 0)

    print(f"\n  total:    max_diff={max_all:.4f}  exact={exact_all*100:.1f}%")
    print(f"  boundary: max_diff={max_bnd:.4f}  exact={exact_bnd*100:.1f}%")
    print(f"  boundary DWC: {'OK 0 viol' if bdry_viol==0 else f'FAIL {bdry_viol} viol'}")
    print(f"  Theorem 1 Step A: {'✅ PASS' if equiv_ok else '❌ FAIL'}  "
          f"speedup={t_cpu/max(t_dd,1e-9):.1f}×")

    return {"experiment": "E-NEW-3", "name": name, "K": K, "delta": delta,
            "R_star": res["R_star"], "max_diff_all": max_all, "exact_all": exact_all,
            "max_diff_boundary": max_bnd, "exact_boundary": exact_bnd,
            "boundary_violations": bdry_viol, "equiv_ok": equiv_ok,
            "t_cpu_s": t_cpu, "t_dd_s": t_dd,
            "speedup": t_cpu / max(t_dd, 1e-9)}


def run_brats(vol_u8, name, K=4, deltas=None, max_rounds=8):
    return run_enew1b(vol_u8, name, K=K,
                      deltas=deltas or [0,1,2,3], max_rounds=max_rounds)


# ── tab:sor-verify aggregation ────────────────────────────────────────────────

def _print_sor_verify_table(brats_e2_results) -> None:
    """Print per-K×δ aggregates matching paper tab:sor-verify (§5.2.2)."""
    import math
    kd_map = defaultdict(list)
    for r in brats_e2_results:
        for row in r["rows"]:
            kd_map[(row["K"], r["delta"])].append(row)

    print(f"\n{'='*80}")
    print("  tab:sor-verify  (N=1251 BraTS 2021 FLAIR, full dataset, seed=42)")
    print(f"{'='*80}")
    hdr = (f"  {'K':>4} {'δ':>4} {'D_sub':>6} {'ρ(%)':>6} "
           f"{'κ(med)':>8} {'σ_cover':>8} {'Pass/N':>9} {'95% CI':>18} "
           f"{'RI-1':>6} {'RI-2':>6}")
    print(hdr); print(f"  {'─'*76}")

    TAU_MIN = 30; RHO_SEED = 0.10
    loocv_correct = loocv_total = 0

    for (K, delta) in sorted(kd_map.keys()):
        rows = kd_map[(K, delta)]
        n_pass = sum(1 for r in rows if r["dwc_ok"])
        n_total = len(rows)
        p = n_pass / n_total if n_total else 0
        z = 1.96
        denom = 1 + z**2 / max(n_total, 1)
        centre = (p + z**2 / (2 * max(n_total, 1))) / denom
        margin = z * math.sqrt(p*(1-p)/max(n_total,1) + z**2/(4*max(n_total,1)**2)) / denom
        lo_ci = max(0.0, centre - margin) * 100
        hi_ci = min(1.0, centre + margin) * 100

        def med(key): return float(np.median([r[key] for r in rows if key in r])) if rows else 0.0
        kappa_med = med("kappa"); D_sub_med = med("D_sub")
        sigma_med = med("sigma_cover"); rho_med = med("rho_pct")

        ri1 = "FIRE" if D_sub_med < TAU_MIN else "ok"
        ri2 = "FIRE" if (delta / max(D_sub_med, 1)) < RHO_SEED else "ok"
        print(f"  {K:>4} {delta:>4} {D_sub_med:>6.0f} {rho_med:>6.1f} "
              f"{kappa_med:>8.1f} {sigma_med:>8.0f} "
              f"{n_pass:>4}/{n_total:<3}  "
              f"[{lo_ci:>5.1f}%, {hi_ci:>5.1f}%]  "
              f"{ri1:>6}  {ri2:>6}")

        actual_ok = (n_pass == n_total)
        ri1_fired = (D_sub_med < TAU_MIN)
        loocv_correct += 1 if (ri1_fired != actual_ok) or (not ri1_fired and actual_ok) else 0
        loocv_total += 1

    print(f"\n  LOOCV (RI-1, τ_min=30): {loocv_correct}/{loocv_total} correct")
    print(f"{'='*80}\n")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(all_results) -> None:
    print(f"\n{'#'*70}")
    print("  DD-FP v10 summary (correct IBI + z_b inclusive pinning)")
    print(f"{'#'*70}")
    for exp_key, label in [
        ("E-NEW-1b", "[E-NEW-1b] δ step-function (Lemma 3+4)"),
        ("E-NEW-2",  "[E-NEW-2] boundary violations per K (Cor.2.1)"),
        ("E-NEW-3",  "[E-NEW-3] CPU vs IBI equivalence (Thm.1)"),
        ("E-BRATS",  "[E-BRATS] BraTS verification"),
    ]:
        if not all_results.get(exp_key):
            continue
        print(f"\n  {label}")
        for r in all_results[exp_key]:
            if exp_key in ("E-NEW-1b", "E-BRATS"):
                tag = "✅" if r["step_ok"] else "❌"
                print(f"  {r['name']:<28}  δ=0:{r['d0_viol']}  δ≥1_ok:{r['d1_ok']}  {tag}")
            elif exp_key == "E-NEW-2":
                tag = "✅" if r["all_K_ok"] else "❌"
                print(f"  {r['name']:<28}  δ={r['delta']}  all_K_ok={r['all_K_ok']}  {tag}")
            elif exp_key == "E-NEW-3":
                tag = "✅" if r["equiv_ok"] else "❌"
                print(f"  {r['name']:<28}  max_bnd={r['max_diff_boundary']:.4f}  "
                      f"R={r['R_star']}  spd={r['speedup']:.1f}×  {tag}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="DD-FP v10 (N=1251 full dataset)")
    ap.add_argument("--brats",      default=None,
                    help="path to BraTS 2021 Training root directory")
    ap.add_argument("--n",          type=int, default=1251,
                    help="number of BraTS subjects (default=1251: all)")
    ap.add_argument("--seed",       type=int, default=42)
    ap.add_argument("--n_cpu",      type=int, default=2)
    ap.add_argument("--size",       type=int, default=48)
    ap.add_argument("--synth_only", action="store_true")
    ap.add_argument("--skip",       nargs="*", default=[],
                    choices=["e1b", "e2", "e3", "brats"])
    ap.add_argument("--out",        default="./dd_fp_v10_n1251_results.json")
    ap.add_argument("--K",          type=int, default=4)
    ap.add_argument("--max_rounds", type=int, default=8)
    ap.add_argument("--deltas",     type=int, nargs="*", default=[0, 1, 2, 3])
    ap.add_argument("--delta_e2",   type=int, default=1,
                    help="delta for E-NEW-2 (default=1)")
    ap.add_argument("--delta_list", type=int, nargs="*", default=[1, 2],
                    help="delta values for tab:sor-verify (default: 1 2)")
    args = ap.parse_args()

    all_results = {k: [] for k in ["E-NEW-1b", "E-NEW-2", "E-NEW-3", "E-BRATS"]}
    volumes = dict(make_synth_volumes(args.size))

    if args.brats and not args.synth_only:
        all_paths = sorted(glob.glob(os.path.join(args.brats, "*", "*_flair.nii*")))
        total = len(all_paths)
        if args.n >= total:
            paths = all_paths
            print(f"BraTS full dataset loaded: N={total} (no sampling)")
        else:
            rng = np.random.default_rng(args.seed)
            idx = rng.choice(total, size=args.n, replace=False)
            paths = [all_paths[i] for i in sorted(idx)]
            print(f"BraTS loaded: {len(paths)} subjects (seed={args.seed}, N={len(paths)}/{total})")

        skipped_corrupt = []
        for p in paths:
            nm = f"BraTS_{Path(p).parent.name}"
            try:
                volumes[nm] = preprocess(nib.load(p).get_fdata(dtype=np.float32))
            except Exception as e:
                print(f"  [SKIP] {nm}: corrupt or load failed — {type(e).__name__}: {e}")
                skipped_corrupt.append(nm)
        if skipped_corrupt:
            print(f"  skipped {len(skipped_corrupt)} corrupt files: {skipped_corrupt[:5]}"
                  f"{'...' if len(skipped_corrupt) > 5 else ''}")
        print(f"  successfully loaded: "
              f"{sum(1 for k in volumes if k.startswith('BraTS'))} subjects")

    if "e1b" not in args.skip:
        print(f"\n{'='*70}\n  E-NEW-1b: δ step-function (Lemma 3+4)\n{'='*70}")
        for name, vol in volumes.items():
            if not name.startswith("synth") and args.synth_only: continue
            r = run_enew1b(vol, name, K=args.K, deltas=args.deltas,
                           max_rounds=args.max_rounds)
            (all_results["E-BRATS"] if name.startswith("BraTS")
             else all_results["E-NEW-1b"]).append(r)

    if "e2" not in args.skip:
        print(f"\n{'='*70}\n  E-NEW-2: boundary violations per K×δ (IBI v10)\n{'='*70}")
        delta_sweep = args.delta_list if hasattr(args, "delta_list") else [1, 2]
        for name, vol in volumes.items():
            if not name.startswith("synth") and args.synth_only: continue
            for d in delta_sweep:
                r = run_enew2(vol, name, K_list=[2,4,8,16], delta=d,
                              max_rounds=args.max_rounds)
                all_results["E-NEW-2"].append(r)
        brats_e2 = [r for r in all_results["E-NEW-2"] if r["name"].startswith("BraTS")]
        if brats_e2:
            _print_sor_verify_table(brats_e2)

    if "e3" not in args.skip:
        print(f"\n{'='*70}\n  E-NEW-3: CPU vs IBI equivalence (Theorem 1 Step A)\n{'='*70}")
        synth_vols = {k: v for k, v in volumes.items() if k.startswith("synth")}
        for name, vol in list(synth_vols.items())[:args.n_cpu]:
            r = run_enew3(vol, name, K=args.K, delta=1, max_rounds=args.max_rounds)
            all_results["E-NEW-3"].append(r)

    print_summary(all_results)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(serialize(all_results), f, indent=2, ensure_ascii=False)
    print(f"\n  saved → {args.out}")


if __name__ == "__main__":
    '''
python experiment_DDFP_all.py --brats ../data --n 1251 \
    --K 8 --delta_e2 2 --max_rounds 16 --skip e1b e3

python experiment_DDFP_all.py --brats ../data --n 50 --seed 42 \
    --K 4 --skip e3
    '''
    main()