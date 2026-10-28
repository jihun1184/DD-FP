"""Step-by-step re-verification harness for Theorem R0 (an independent re-check of the existing proof;
no part of the frozen project code is used).  Usage:  python3 r0_verify.py <mode> [N] [seed]
modes: checker | local | random | exhaustive | mutants
"""
import sys, json, itertools, time
import numpy as np
from r0lib import (ispan, pad_frame, dwc_violations, critical_blocks, brute_critical, fp_run,
                   validate_admissible, neighbours)

# ----------------------------------------------------------------------------- helpers
def split_domain_1d(D, K):
    base, rem = divmod(D, K)
    out, z = [], 0
    for k in range(K):
        s = base + (1 if k < rem else 0)
        out.append((z, z + s)); z += s
    return out

def is_dwc(f):
    a, b = dwc_violations(f)
    return a == 0 and b == 0

def frame_adjacent_mask(shape):
    """Interior cells (of the padded lattice) that are 6-adjacent to a frame cell."""
    inner = np.zeros(shape, bool); inner[1:-1, 1:-1, 1:-1] = True
    P = np.pad(inner, 1, constant_values=False)
    m = np.zeros(shape, bool)
    for ax in range(3):
        for sh in (-1, 1):
            sl = [slice(1, -1)] * 3
            sl[ax] = slice(1 + sh, shape[ax] + 1 + sh)
            m |= ~P[tuple(sl)]
    return m & inner

def pick_root(shape, rng):
    fr = np.ones(shape, bool); fr[1:-1, 1:-1, 1:-1] = False
    cells = np.flatnonzero(fr.ravel())
    return int(cells[rng.integers(len(cells))])

# ----------------------------------------------------------------------------- per-window lemma checks
def check_window(u_win, rng, sched="rand", alpha=0, l_inf=0, collect=True):
    """Run FP on one window and test every in-window step of the proof. Returns (counts, uhat_interior, lo, hi)."""
    lo, hi = ispan(u_win)
    return check_window_lh(lo, hi, rng, sched, alpha, l_inf)

def check_window_lh(lo, hi, rng, sched="rand", alpha=0, l_inf=0, forced=None, within="rand"):
    lo_p, hi_p = pad_frame(lo, alpha), pad_frame(hi, alpha)
    shape = lo_p.shape
    root = pick_root(shape, rng)  # a frame cell (corner or random)
    run = fp_run(lo_p, hi_p, root, l_inf, rng, sched=sched, forced=forced, within=within)
    uh = run.uhat
    c = dict.fromkeys(["adm_bad", "A1lo", "A1hi", "A2lo", "A2hi", "range", "F1", "F2", "F3", "KL", "P", "Q", "TB_face", "TB_cube",
                       "nb_pairs"], 0)
    adm = validate_admissible(run, lo_p, hi_p, l_inf)
    c["adm_bad"] = len(adm); c["adm_items"] = adm
    # A1 / A2
    c["A1lo"] = sum(dwc_violations(lo)); c["A1hi"] = sum(dwc_violations(hi))
    if alpha == 0:
        c["A2lo"] = sum(dwc_violations(lo_p)); c["A2hi"] = sum(dwc_violations(hi_p))
    # Range lemma
    inner = np.zeros(shape, bool); inner[1:-1, 1:-1, 1:-1] = True
    c["range"] = int(((uh < lo_p) | (uh > hi_p)).sum())
    # F1, F2 (cells other than the root)
    ld = run.ldisc
    notroot = np.ones(shape, bool); notroot.ravel()[root] = False
    f1 = notroot & (uh < hi_p) & (uh != np.maximum(ld, lo_p))
    f2 = notroot & (uh > lo_p) & (uh != np.minimum(ld, hi_p))
    c["F1"], c["F2"] = int(f1.sum()), int(f2.sum())
    # F3 from the trace
    nb = neighbours(shape)
    lev = np.array(run.levels)
    didx = {root: -1}
    for t, cell in enumerate(run.order):
        for q in nb[cell]:
            if q not in didx: didx[q] = t
    tp = {cell: t for t, cell in enumerate(run.order)}
    L_, H_ = lo_p.ravel(), hi_p.ravel()
    f3 = 0
    for z, td in didx.items():
        b = int(uh.ravel()[z]); t1 = tp[z]
        seq = np.concatenate(([ld.ravel()[z]], lev[td + 1:t1])) if t1 > td + 1 else np.array([ld.ravel()[z]])
        ge = np.nonzero(seq >= b)[0]
        if ge.size and not (seq[ge[0]:] >= b).all(): f3 += 1
        le = np.nonzero(seq <= b)[0]
        if le.size and not (seq[le[0]:] <= b).all(): f3 += 1
    c["F3"] = f3
    # KL for every ordered 6-neighbour pair: violation iff max(u(p),lo(q)) < min(hi(p),u(q))
    kl = 0; npairs = 0
    for ax in range(3):
        a = [slice(None)] * 3; b = [slice(None)] * 3
        a[ax] = slice(0, shape[ax] - 1); b[ax] = slice(1, shape[ax])
        a, b = tuple(a), tuple(b)
        for (P, Q) in ((a, b), (b, a)):
            viol = np.maximum(uh[P], lo_p[Q]) < np.minimum(hi_p[P], uh[Q])
            kl += int(viol.sum()); npairs += viol.size
    c["KL"] = kl; c["nb_pairs"] = npairs
    # Lemma P: interior cells adjacent to the frame have uhat = lo
    adj = frame_adjacent_mask(shape)
    c["P"] = int((adj & (uh != lo_p)).sum())
    # Lemma Q: p interior neighbour of a frame-adjacent cell c0 -> uhat(p) <= max(lo(p), lo(c0))
    q_viol = 0
    for ax in range(3):
        for sh in (1, -1):
            sl_a = [slice(1, -1)] * 3; sl_b = [slice(1, -1)] * 3
            if sh == 1:
                sl_a[ax] = slice(1, shape[ax] - 2); sl_b[ax] = slice(2, shape[ax] - 1)
            else:
                sl_a[ax] = slice(2, shape[ax] - 1); sl_b[ax] = slice(1, shape[ax] - 2)
            c0 = tuple(sl_a); p = tuple(sl_b)
            m = adj[c0]
            q_viol += int((m & (uh[p] > np.maximum(lo_p[p], lo_p[c0]))).sum())
    c["Q"] = q_viol
    # T-B: uhat on the padded lattice is DWC
    c["TB_face"], c["TB_cube"] = dwc_violations(uh)
    c["_branches"] = run.branches
    return c, uh[1:-1, 1:-1, 1:-1], lo, hi

# ----------------------------------------------------------------------------- assembly
def assemble(u, K, rng, sched="rand", delta=1, own="asc", alpha=0, l_inf=0, collect_window=True):
    W, H, D = u.shape
    slabs = split_domain_1d(D, K)
    B = np.full((2 * W - 1, 2 * H - 1, 2 * D - 1), -1, np.int64)
    wins = []
    tot = None
    for k, (z0, z1) in enumerate(slabs):
        zmin, zmax = max(0, z0 - delta), min(D, z1 + delta)
        c, uh, lo, hi = check_window(u[:, :, zmin:zmax], rng, sched, alpha, l_inf)
        wins.append((zmin, zmax, uh))
        if tot is None: tot = {k_: v for k_, v in c.items() if isinstance(v, int)}; tot["adm_items"] = set(c["adm_items"])
        else:
            for k_, v in c.items():
                if isinstance(v, int): tot[k_] += v
            tot["adm_items"] |= set(c["adm_items"])
    own_slab = [None] * (2 * D - 1)
    for g in range(2 * D - 1):
        v = g // 2
        own_slab[g] = next(k for k, (z0, z1) in enumerate(slabs) if z0 <= v < z1)
    if own == "desc":
        for k in range(K - 1):
            zb = slabs[k][1]
            own_slab[2 * zb - 1], own_slab[2 * zb] = k + 1, k
    for g in range(2 * D - 1):
        k = own_slab[g]; zmin, zmax, uh = wins[k]
        loc = g - 2 * zmin
        if 0 <= loc < uh.shape[2]:
            B[:, :, g] = uh[:, :, loc]
    return B, slabs, tot

def check_assembled(u, B, slabs):
    W, H, D = u.shape
    res = {}
    res["unassigned"] = int((B < 0).sum())
    res["voxel_eq"] = int((B[::2, ::2, ::2] != u).sum())
    t1 = 0
    for k in range(len(slabs) - 1):
        zb = slabs[k][1]
        t1 += int((B[:, :, 2 * zb - 1] > B[:, :, 2 * zb]).sum())
    res["T1"] = t1
    res["B_face"], res["B_cube"] = dwc_violations(B)
    return res

# ----------------------------------------------------------------------------- volume generators
def gen_volume(rng):
    W, H = rng.integers(2, 6, size=2); D = int(rng.integers(2, 10))
    kind = int(rng.integers(0, 10))
    shp = (int(W), int(H), D)
    if kind >= 8:
        # shells: value depends on the distance to the volume boundary (alternating bright/dark rings), the adversarial case for
        # level up/down decisions; kind 9 adds small noise
        W, H = int(rng.integers(3, 7)), int(rng.integers(3, 7)); D = int(rng.integers(3, 10)); shp = (W, H, D)
        x, y, z = np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing="ij")
        d = np.minimum.reduce([x, y, z, W - 1 - x, H - 1 - y, D - 1 - z])
        prof = rng.integers(0, 256, d.max() + 1)
        u = prof[d] + (rng.integers(-6, 7, shp) if kind == 9 else 0)
        return np.asarray(np.clip(u, 0, 255), np.int64), kind
    if kind == 0: u = rng.integers(0, 3, shp)
    elif kind == 1: u = rng.integers(0, 4, shp)
    elif kind == 2: u = rng.choice([0, 128, 255], shp)
    elif kind == 3: u = rng.integers(0, 256, shp)
    elif kind == 4:
        base = rng.integers(0, 256, (1, 1, 1)); u = np.clip(base + np.cumsum(rng.integers(-20, 21, shp), axis=2), 0, 255)
    elif kind == 5: u = rng.choice([0, 255, 255, 0, 17], shp)
    elif kind == 6: u = np.where(rng.random(shp) < 0.15, rng.integers(0, 256, shp), 0)
    else: u = np.where(rng.random(shp) < 0.15, rng.integers(0, 256, shp), 255)
    return np.asarray(u, np.int64), kind

# ----------------------------------------------------------------------------- modes
def mode_checker(seed):
    rng = np.random.default_rng(seed)
    bad = 0; n = 0
    for _ in range(3000):
        shp = tuple(int(x) for x in rng.integers(1, 5, 3))
        X = rng.random(shp) < rng.random()
        if critical_blocks(X) != brute_critical(X): bad += 1
        n += 1
    # hand-built positives: face diagonal, antipodal cube pair and its complement
    X = np.zeros((2, 2, 1), bool); X[0, 0, 0] = X[1, 1, 0] = True
    pos_face = critical_blocks(X) == (1, 0)
    X = np.zeros((2, 2, 2), bool); X[0, 0, 0] = X[1, 1, 1] = True
    pos_cube = critical_blocks(X)[1] == 1
    X = ~X
    pos_comp = critical_blocks(X)[1] == 1
    return {"random_cases": n, "mismatch_vs_bruteforce": bad, "positive_controls": [bool(pos_face), bool(pos_cube), bool(pos_comp)]}

def mode_local():
    out = {}
    # A1/A2 exhaustive on all u in {0,1,2}^(2x2x2) and u in {0,1}^(3x2x2)
    cnt = {"vol": 0, "A1_viol": 0, "A2_viol": 0}
    for shp, vals in (((2, 2, 2), 3), ((3, 2, 2), 2), ((2, 2, 3), 2)):
        for flat in itertools.product(range(vals), repeat=int(np.prod(shp))):
            u = np.array(flat, np.int64).reshape(shp)
            lo, hi = ispan(u)
            cnt["vol"] += 1
            cnt["A1_viol"] += sum(dwc_violations(lo)) + sum(dwc_violations(hi))
            cnt["A2_viol"] += sum(dwc_violations(pad_frame(lo))) + sum(dwc_violations(pad_frame(hi)))
    out["A1_A2_exhaustive"] = cnt
    # Lemma N exhaustive: k=2 (4 cells) and k=3 (8 cells): all Y that are non-critical on S (and, for k=3, on its six faces)
    def faces_of_cube():
        fs = []
        for ax in range(3):
            for v in (0, 1):
                fs.append([(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1) if (i, j, k)[ax] == v])
        return fs
    cells3 = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    def diag_face(F, Y):  # F: 4 cells of a face; critical if Y∩F is a diagonal pair or its complement (= other diagonal)
        ins = [c for c in F if c in Y]
        if len(ins) != 2: return False
        a, b = ins
        return sum(x != y for x, y in zip(a, b)) == 2
    n3 = 0; viol3 = 0; cases3 = 0
    for mask in range(256):
        Y = {cells3[i] for i in range(8) if mask >> i & 1}
        if any(diag_face(F, Y) for F in faces_of_cube()): continue
        pair_crit = False
        for p in cells3:
            q = tuple(1 - x for x in p)
            if p < q and (Y == {p, q} or Y == set(cells3) - {p, q}): pair_crit = True
        if pair_crit: continue
        n3 += 1
        for p in cells3:
            q = tuple(1 - x for x in p)
            if p in Y and q in Y:
                cases3 += 1
                nbrs = [tuple(p[i] if i != ax else 1 - p[i] for i in range(3)) for ax in range(3)]
                if not any(n in Y for n in nbrs): viol3 += 1
    n2 = 0; viol2 = 0; cases2 = 0
    cells2 = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for mask in range(16):
        Y = {cells2[i] for i in range(4) if mask >> i & 1}
        if len(Y) == 2 and sum(x != y for x, y in zip(*sorted(Y))) == 2: continue
        n2 += 1
        for p in cells2:
            q = (1 - p[0], 1 - p[1])
            if p in Y and q in Y:
                cases2 += 1
                if not any(n in Y for n in ((1 - p[0], p[1]), (p[0], 1 - p[1]))): viol2 += 1
    out["N_exhaustive"] = {"k3_noncritical_Y": n3, "k3_antipodal_cases": cases3, "k3_violations": viol3,
                           "k2_noncritical_Y": n2, "k2_antipodal_cases": cases2, "k2_violations": viol2}
    # T-2 exhaustive: 2 adjacent planes, columns of the 2x2 footprint with non-decreasing values; all value triples
    t2 = {"configs": 0, "face_xz": 0, "face_yz": 0, "cube": 0}
    for vals in itertools.product(range(3), repeat=8):
        B = np.array(vals).reshape(2, 2, 2)  # axes: x, y, z(plane); z=0 lower, z=1 upper
        if not (B[:, :, 0] <= B[:, :, 1]).all(): continue
        t2["configs"] += 1
        a, b = dwc_violations(B)
        # all xz and yz faces and the cube span both planes; xy faces lie in one plane (excluded here)
        fx = 0
        for t in np.unique(B):
            X = B >= t
            for (ax1, ax2) in ((0, 2), (1, 2)):
                s = [slice(None)] * 3
                for fixed in (0, 1):
                    other = ({0, 1, 2} - {ax1, ax2}).pop()
                    s2 = list(s); s2[other] = fixed
                    sub = X[tuple(s2)]
                    fx += critical_blocks(sub[..., None] if sub.ndim == 2 else sub)[0]
            t2["cube"] += critical_blocks(X)[1]
        t2["face_xz"] += fx
    out["T2_exhaustive_monotone_columns"] = t2
    return out

def run_random(N, seed, scheds=("rand", "near")):
    rng = np.random.default_rng(seed)
    agg = {"volumes": 0, "windows_runs": 0, "pairs": 0}
    keys = ["adm_bad", "A1lo", "A1hi", "A2lo", "A2hi", "range", "F1", "F2", "F3", "KL", "P", "Q", "TB_face", "TB_cube"]
    agg.update({k: 0 for k in keys})
    asm_keys = ["unassigned", "voxel_eq", "T1", "B_face", "B_cube"]
    agg.update({k: 0 for k in asm_keys})
    agg["by_kind"] = {}
    ties = 0
    t0 = time.time()
    for i in range(N):
        u, kind = gen_volume(rng)
        D = u.shape[2]
        K = int(rng.integers(1, D + 1))
        sched = scheds[int(rng.integers(len(scheds)))]
        B, slabs, tot = assemble(u, K, rng, sched)
        res = check_assembled(u, B, slabs)
        agg["volumes"] += 1; agg["windows_runs"] += K
        agg["pairs"] += tot["nb_pairs"]
        for k in keys: agg[k] += tot[k]
        for k in asm_keys: agg[k] += res[k]
        bk = agg["by_kind"].setdefault(str(kind), [0, 0]); bk[0] += 1
        if any(tot[k] for k in keys) or any(res[k] for k in asm_keys): bk[1] += 1
        if (i + 1) % 500 == 0: print(f"  {i+1}/{N} {time.time()-t0:.0f}s viol_vols={sum(v[1] for v in agg['by_kind'].values())}", flush=True)
    return agg

def mode_exhaustive(seed, limit=None):
    """All u in {0,1,2}^(W x H x D) (tiny), K in {1,2,..}, and for every window ALL decision sequences (up/down at each
    level change, within-level order fixed) -> every distinct window output is combined with every other."""
    keys = ["range", "F1", "F2", "KL", "P", "Q", "TB_face", "TB_cube", "adm_bad"]
    rng = np.random.default_rng(seed)
    res = {"shape": [], "volumes": 0, "window_decision_sequences": 0, "distinct_window_outputs": 0, "assemblies": 0,
           "B_dwc_violations": 0, "T1_violations": 0, "voxel_eq": 0, "lemma_violations": 0, "max_branches": 0}
    shapes = [((2, 2, 2), 3, (1, 2)), ((2, 2, 3), 3, (1, 2, 3)), ((1, 2, 4), 3, (2, 3, 4)), ((3, 2, 2), 3, (2,))]
    for shp, vals, Ks in shapes:
        W, H, D = shp
        cnt_this = 0
        total = vals ** (W * H * D)
        full = (limit is None) or total <= limit
        it = itertools.product(range(vals), repeat=W * H * D) if full else (tuple(rng.integers(0, vals, W * H * D)) for _ in range(limit))
        res.setdefault("complete_enumeration", []).append(bool(full))
        for flat in it:
            u = np.array(flat, np.int64).reshape(shp)
            cnt_this += 1
            for K in Ks:
                slabs = split_domain_1d(D, K)
                outs = []
                for (z0, z1) in slabs:
                    zmin, zmax = max(0, z0 - 1), min(D, z1 + 1)
                    lo, hi = ispan(u[:, :, zmin:zmax]); lo_p, hi_p = pad_frame(lo), pad_frame(hi)
                    seen = {}
                    stack = [[]]
                    while stack:
                        fz = stack.pop()
                        r = fp_run(lo_p, hi_p, 0, 0, rng, sched="rand", within="fixed", forced=fz)
                        res["window_decision_sequences"] += 1
                        res["max_branches"] = max(res["max_branches"], r.branches)
                        if len(fz) < r.branches:
                            # extend decisions: explore both options at the first undecided branch
                            for opt in (0, 1): stack.append(fz + [opt])
                            continue
                        if validate_admissible(r, lo_p, hi_p, 0): res["lemma_violations"] += 1
                        key = r.uhat.tobytes()
                        if key not in seen:
                            seen[key] = r.uhat[1:-1, 1:-1, 1:-1].copy()
                            uh = r.uhat
                            bad = int(((uh < lo_p) | (uh > hi_p)).sum())
                            bad += sum(dwc_violations(uh))
                            res["lemma_violations"] += bad
                    outs.append((zmin, zmax, list(seen.values())))
                    res["distinct_window_outputs"] += len(seen)
                for combo in itertools.product(*[o[2] for o in outs]):
                    B = np.full((2 * W - 1, 2 * H - 1, 2 * D - 1), -1, np.int64)
                    for g in range(2 * D - 1):
                        v = g // 2
                        k = next(kk for kk, (z0, z1) in enumerate(slabs) if z0 <= v < z1)
                        zmin = outs[k][0]
                        B[:, :, g] = combo[k][:, :, g - 2 * zmin]
                    r_ = check_assembled(u, B, slabs)
                    res["assemblies"] += 1
                    res["B_dwc_violations"] += r_["B_face"] + r_["B_cube"]
                    res["T1_violations"] += r_["T1"]; res["voxel_eq"] += r_["voxel_eq"] + r_["unassigned"]
        res["shape"].append({"shape": list(shp), "values": vals, "volumes": cnt_this, "K": list(Ks)})
        res["volumes"] += cnt_this
    return res

def mode_mutants(N, seed):
    """Sensitivity of the harness: each mutant breaks one hypothesis; counts show which checks react."""
    out = {}
    cfgs = {
        "baseline": dict(),
        "M1_P3_violating_scheduler": dict(sched="bad"),
        "M2_frame_alpha1_start1": dict(alpha=1, l_inf=1),
        "M3_halo_delta2": dict(delta=2),
        "M4_descending_owner_at_seam": dict(own="desc"),
    }
    for name, kw in cfgs.items():
        rng = np.random.default_rng(seed)
        agg = {"volumes": 0, "adm_bad_runs": 0, "KL": 0, "F3": 0, "P": 0, "Q": 0, "TB": 0, "T1": 0, "B_dwc": 0, "vols_with_B_dwc_violation": 0,
               "vols_with_T1_violation": 0}
        for i in range(N):
            u, kind = gen_volume(rng)
            K = int(rng.integers(2, max(3, u.shape[2] + 1))) if u.shape[2] >= 2 else 1
            K = min(K, u.shape[2])
            B, slabs, tot = assemble(u, K, rng, **{"sched": "rand", **kw})
            res = check_assembled(u, B, slabs)
            agg["volumes"] += 1
            agg["adm_bad_runs"] += int(tot["adm_bad"] > 0)
            agg["KL"] += tot["KL"]; agg["F3"] += tot["F3"]; agg["P"] += tot["P"]; agg["Q"] += tot["Q"]
            agg["TB"] += tot["TB_face"] + tot["TB_cube"]; agg["T1"] += res["T1"]
            agg["B_dwc"] += res["B_face"] + res["B_cube"]
            agg["vols_with_B_dwc_violation"] += int(res["B_face"] + res["B_cube"] > 0)
            agg["vols_with_T1_violation"] += int(res["T1"] > 0)
        out[name] = agg
    return out

def enum_outputs(lo, hi, rng, cap=12):
    """All distinct outputs over all up/down decision sequences (within-level order random per run). None if > 2**cap leaves."""
    lo_p, hi_p = pad_frame(lo), pad_frame(hi)
    outs = {}; stack = [[]]; leaves = 0
    while stack:
        fz = stack.pop()
        r = fp_run(lo_p, hi_p, 0, 0, rng, sched="rand", within="rand", forced=fz)
        if len(fz) < r.branches:
            if len(fz) >= cap:
                # too many decision points to enumerate: fall back to 150 random decision sequences (flag via leaves = -1)
                outs = {}
                for _ in range(150):
                    r2 = fp_run(lo_p, hi_p, 0, 0, rng, sched="rand", within="rand")
                    outs.setdefault(r2.uhat.tobytes(), r2.uhat[1:-1, 1:-1, 1:-1].copy())
                return list(outs.values()), -1
            stack.append(fz + [0]); stack.append(fz + [1]); continue
        leaves += 1
        outs.setdefault(r.uhat.tobytes(), r.uhat[1:-1, 1:-1, 1:-1].copy())
    return list(outs.values()), leaves

def mode_wide(N, seed):
    """Interval maps (lo,hi)=(lo of u1, hi of u2) with u1<=u2: lo,hi are DWC (A1) and lo<=hi, but the pair is NOT the span of one image.
    Window-level steps only (Range, F1-F3, KL, P, Q, T-B), many up/down decision points; all decision sequences enumerated when <=2^10."""
    rng = np.random.default_rng(seed)
    keys = ["adm_bad", "A1lo", "A1hi", "A2lo", "A2hi", "range", "F1", "F2", "F3", "KL", "P", "Q", "TB_face", "TB_cube"]
    agg = {"maps": 0, "runs": 0, "maps_with_branch": 0, "maps_with_distinct_outputs": 0, "max_distinct_outputs": 0, "enumerated_leaves": 0,
           "enumeration_capped": 0}
    agg.update({k: 0 for k in keys})
    for i in range(N):
        W, H, D = (int(x) for x in rng.integers(2, 5, 3))
        shp = (W, H, D)
        kind = int(rng.integers(0, 4))
        if kind == 0: a, b = rng.integers(0, 256, shp), rng.integers(0, 256, shp)
        elif kind == 1: a, b = rng.integers(0, 6, shp), rng.integers(0, 6, shp)
        elif kind == 2: a, b = rng.choice([0, 40, 90, 200, 255], shp), rng.choice([0, 40, 90, 200, 255], shp)
        else:
            x, y, z = np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing="ij")
            d = np.minimum.reduce([x, y, z, W - 1 - x, H - 1 - y, D - 1 - z]); prof = rng.integers(0, 256, d.max() + 1)
            a = np.clip(prof[d] + rng.integers(-8, 9, shp), 0, 255); b = np.clip(prof[d] + rng.integers(-8, 9, shp), 0, 255)
        u1, u2 = np.minimum(a, b), np.maximum(a, b)
        lo, _ = ispan(u1); _, hi = ispan(u2)
        assert (lo <= hi).all()
        agg["maps"] += 1
        e = enum_outputs(lo, hi, rng, cap=10)
        if e is not None:
            outs, leaves = e
            if leaves == -1: agg["enumeration_capped"] += 1
            else: agg["enumerated_leaves"] += leaves
            if leaves > 1 or leaves == -1: agg["maps_with_branch"] += 1
            if len(outs) > 1: agg["maps_with_distinct_outputs"] += 1
            agg["max_distinct_outputs"] = max(agg["max_distinct_outputs"], len(outs))
        # lemma checks on randomized runs (random decisions, random order) -- plus one per leaf class is covered by outs below
        for _ in range(3):
            c, uh, _, _ = check_window_lh(lo, hi, rng, "rand")
            agg["runs"] += 1
            for k in keys: agg[k] += c[k]
        for uh in outs:
            up = pad_frame(np.zeros_like(uh))  # DWC of each enumerated output on the padded lattice
            full = np.pad(uh, 1, constant_values=0)
            agg["TB_face"] += dwc_violations(full)[0]; agg["TB_cube"] += dwc_violations(full)[1]
            agg["range"] += int(((uh < lo) | (uh > hi)).sum())
    return agg

def mode_dfs_span(N, seed):
    """Span maps of random volumes: enumerate ALL up/down decision sequences per window and combine all distinct window outputs
    (capped) into assembled fields; count how often the admissible freedom changes the output."""
    rng = np.random.default_rng(seed)
    agg = {"volumes": 0, "windows": 0, "windows_with_branch": 0, "windows_with_distinct_outputs": 0, "capped": 0, "assemblies": 0,
           "B_dwc_violations": 0, "T1_violations": 0, "voxel_eq": 0}
    for i in range(N):
        u, kind = gen_volume(rng)
        W, H, D = u.shape; K = int(rng.integers(1, D + 1)); slabs = split_domain_1d(D, K)
        per = []; ok = True
        for (z0, z1) in slabs:
            zmin, zmax = max(0, z0 - 1), min(D, z1 + 1)
            lo, hi = ispan(u[:, :, zmin:zmax])
            e = enum_outputs(lo, hi, rng, cap=8)
            agg["windows"] += 1
            outs, leaves = e
            if leaves == -1: agg["capped"] += 1
            agg["windows_with_branch"] += int(leaves > 1 or leaves == -1); agg["windows_with_distinct_outputs"] += int(len(outs) > 1)
            per.append((zmin, outs))
        agg["volumes"] += 1
        if not ok: continue
        combos = list(itertools.product(*[range(len(p[1])) for p in per]))
        if len(combos) > 32: combos = [combos[j] for j in rng.choice(len(combos), 32, replace=False)]
        for cb in combos:
            B = np.full((2 * W - 1, 2 * H - 1, 2 * D - 1), -1, np.int64)
            for g in range(2 * D - 1):
                v = g // 2; k = next(kk for kk, (a, b) in enumerate(slabs) if a <= v < b)
                zmin, outs = per[k]; B[:, :, g] = outs[cb[k]][:, :, g - 2 * zmin]
            r_ = check_assembled(u, B, slabs)
            agg["assemblies"] += 1; agg["B_dwc_violations"] += r_["B_face"] + r_["B_cube"]
            agg["T1_violations"] += r_["T1"]; agg["voxel_eq"] += r_["voxel_eq"] + r_["unassigned"]
    return agg

def mode_arbitrary(N, seed):
    """Scheduling lemmas under genuine non-determinism: ARBITRARY interval maps lo<=hi (not DWC, not spans).  Range, F1-F3, KL, P, Q do not
    use DWC of lo/hi.  Many runs per map (random up/down decisions, random order, random frame root); counts how often outputs differ."""
    rng = np.random.default_rng(seed)
    keys = ["adm_bad", "range", "F1", "F2", "F3", "KL", "P", "Q"]
    agg = {"maps": 0, "runs": 0, "maps_with_distinct_outputs": 0, "max_distinct_outputs": 0, "pairs": 0}
    agg.update({k: 0 for k in keys})
    for i in range(N):
        shp = tuple(int(x) for x in rng.integers(1, 5, 3)); mx = int(rng.choice([3, 6, 12, 255]))
        lo = rng.integers(0, mx + 1, shp); hi = lo + rng.integers(0, max(1, mx // int(rng.integers(1, 4))) + 1, shp); hi = np.minimum(hi, 255)
        outs = set()
        lo_c, hi_c = lo, hi
        for _ in range(30):
            c, uh, _, _ = check_window_lh(lo_c, hi_c, rng, "rand")
            agg["runs"] += 1; agg["pairs"] += c["nb_pairs"]
            for k in keys: agg[k] += c[k]
            outs.add(uh.tobytes())
        agg["maps"] += 1
        if len(outs) > 1: agg["maps_with_distinct_outputs"] += 1
        agg["max_distinct_outputs"] = max(agg["max_distinct_outputs"], len(outs))
    return agg

if __name__ == "__main__":
    mode = sys.argv[1]
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261027
    t0 = time.time()
    if mode == "checker": r = mode_checker(seed)
    elif mode == "local": r = mode_local()
    elif mode == "random": r = run_random(N, seed)
    elif mode == "exhaustive": r = mode_exhaustive(seed, limit=N if N > 0 else None)
    elif mode == "mutants": r = mode_mutants(N, seed)
    elif mode == "wide": r = mode_wide(N, seed)
    elif mode == "arbitrary": r = mode_arbitrary(N, seed)
    elif mode == "dfs_span": r = mode_dfs_span(N, seed)
    r["_meta"] = {"mode": mode, "N": N, "seed": seed, "seconds": round(time.time() - t0, 1)}
    print(json.dumps(r, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))
    json.dump(r, open(f"results_{mode}_{seed}.json", "w"), indent=1, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
