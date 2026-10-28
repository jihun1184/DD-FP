#!/usr/bin/env python3
"""Independent global DWC re-verification of the 53 retained outputs (3 BraTS + 30 CREMI3D + 20 DRIVE).
Written from the DWC definition (Boutry et al.): every 2^k block, k in [2,n], every antagonist pair:
    intvl(pair) ∩ span(other corners) != ∅     (gray-level form; == all threshold sets DWC).
For arrays with <=3 distinct values, additionally cross-checked with the threshold-set lookup oracle.
Usage: python3 -I r1_check53.py <brats_exports.zip> <drive_cremi.zip> <out.json>   (numpy only; zips are streamed, never extracted)"""
import sys, io, json, zipfile, itertools, re, hashlib, time
import numpy as np

def _lut(k):
    nc = 1 << k; allm = (1 << nc) - 1; lut = np.zeros(1 << nc, bool)
    for c in range(nc // 2):
        pair = (1 << c) | (1 << ((nc - 1) ^ c)); lut[pair] = True; lut[allm ^ pair] = True
    return lut
LUT = {2: _lut(2), 3: _lut(3)}

def _corners(a, axes):
    sh = [s - (1 if i in axes else 0) for i, s in enumerate(a.shape)]
    return {o: a[tuple(slice(o[axes.index(i)], o[axes.index(i)] + sh[i]) if i in axes else slice(None) for i in range(a.ndim))]
            for o in itertools.product((0, 1), repeat=len(axes))}

def interval_bad(a, axes):
    cs = _corners(a, axes); k = len(axes); full = (1 << k) - 1; offs = list(cs)
    idx = {o: sum(b << i for i, b in enumerate(o)) for o in offs}; inv = {v: o for o, v in idx.items()}
    bad = np.zeros(cs[offs[0]].shape, bool)
    for c in range(len(offs) // 2):
        p, q = inv[c], inv[full ^ c]; rest = [cs[o] for o in offs if o not in (p, q)]
        rlo, rhi = np.minimum.reduce(rest), np.maximum.reduce(rest)
        lo, hi = np.minimum(cs[p], cs[q]), np.maximum(cs[p], cs[q]); bad |= (hi < rlo) | (rhi < lo)
    return bad

def oracle_bad(a, axes, levels):
    cs_axes = len(axes); acc = None
    for t in levels:
        cs = _corners(a >= t, axes); code = sum(cs[o].astype(np.int16) << sum(b << i for i, b in enumerate(o)) for o in cs)
        m = LUT[cs_axes][code]; acc = m if acc is None else (acc | m)
    return acc

def check(a, slab=16, levels=None):
    """returns counts per block family. 2-D: 'xy' only. 3-D: xy,xz,yz faces + cube. Slab along axis 0 with overlap 1 (no double counting)."""
    n = a.ndim
    fam = {"xy": (0, 1)} if n == 2 else {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2), "cube": (0, 1, 2)}
    cnt = {k: 0 for k in fam}; orc = {k: 0 for k in fam} if levels is not None else None
    Z = a.shape[0]
    for z0 in range(0, Z - 1, slab):
        sub = a[z0:min(z0 + slab + 1, Z)]; last = z0 + slab >= Z - 1
        for key, axes in fam.items():
            m = interval_bad(sub, axes)
            keep = m if (last or 0 in axes) else m[:slab]       # blocks spanning axis 0 are anchored in [0,slab): disjoint between slabs; others: drop the shared top layer
            cnt[key] += int(keep.sum())
            if orc is not None:
                o = oracle_bad(sub, axes, levels); orc[key] += int((o if (last or 0 in axes) else o[:slab]).sum())
    return cnt, orc

def load(z, name): return np.load(io.BytesIO(z.read(name)), allow_pickle=False)

def run_one(label, arrs):
    """arrs: dict role->array. Returns record."""
    rec = {"case": label, "shape": list(next(iter(arrs.values())).shape)}
    for role, a in arrs.items():
        a = np.ascontiguousarray(a, dtype=np.float32); u = np.unique(a)
        cnt, orc = check(a, levels=u if len(u) <= 3 else None)
        rec[role] = {"n_unique": int(len(u)), "viol": cnt, "oracle_viol": orc, "nonfinite": int((~np.isfinite(a)).sum())}
        if orc is not None: assert orc == cnt, (label, role, cnt, orc)
    if "raw" in arrs and rec["raw"]["n_unique"] > 2:          # R1 evidence convention: raw field thresholded at 0.5 (binary) -> directly comparable to the recorded counts
        b = (np.asarray(arrs["raw"], np.float32) > 0.5).astype(np.float32); cnt, orc = check(b, levels=[1.0]); rec["raw_thr0.5"] = {"n_unique": int(len(np.unique(b))), "viol": cnt, "oracle_viol": orc}
        assert orc == cnt, (label, "raw_thr0.5", cnt, orc)
    if "ddfp" in arrs and "seq" in arrs: rec["ddfp_vs_seq_differing_cells"] = int((arrs["ddfp"] != arrs["seq"]).sum())
    return rec

def main(bz, dz, out, sel="BraTS,CREMI,DRIVE"):
    """sel: comma list of groups, each optionally sliced GROUP:start:stop over the sorted case list (keeps a run within a time limit)."""
    recs = []; t0 = time.time(); want = {}
    for item in sel.split(","):
        g, *r = item.split(":"); want[g] = slice(*(int(x) if x else None for x in r)) if r else slice(None)
    sha = hashlib.sha256(open(dz, "rb").read()).hexdigest() if want.keys() & {"CREMI", "DRIVE"} else None
    if "BraTS" in want:
        with zipfile.ZipFile(bz) as z:
            for c in sorted({n.split("/")[1] for n in z.namelist() if n.startswith("exports/") and n.endswith(".npy")})[want["BraTS"]]:
                g = lambda r: load(z, f"exports/{c}/{r}.npy")
                recs.append(dict(group="BraTS", **run_one(c, {"ddfp": g("corrected_production_ddfp"), "seq": g("sequential_fp"), "raw": g("raw_or_predwc")})))
                print(c, round(time.time() - t0), "s", flush=True)
    if sha:
        root = "QUALITATIVE_DRIVE_CREMI_RUN_R2_20260908/inputs/"
        with zipfile.ZipFile(dz) as z:
            for grp in [g for g in ("CREMI", "DRIVE") if g in want]:
                cs = sorted({n[len(root):].split("/")[2] for n in z.namelist() if n.startswith(root + grp + "/cases/") and n.endswith(".npy")})[want[grp]]
                for c in cs:
                    p = f"{root}{grp}/cases/{c}/"
                    recs.append(dict(group=grp, **run_one(c, {"ddfp": load(z, p + "corrected_ddfp.npy"), "seq": load(z, p + "sequential_fp.npy"), "raw": load(z, p + "raw_no_interp_expanded.npy")})))
                print(grp, len(cs), round(time.time() - t0), "s", flush=True)
    summ = {}
    for r in recs:
        s = summ.setdefault(r["group"], {"n": 0, **{f"{role}_{k}": 0 for role in ("ddfp", "seq", "raw") for k in ("face", "cube")}, "ddfp_ne_seq_cases": 0, "ddfp_vs_seq_cells": []})
        s["n"] += 1
        for role in [x for x in ("ddfp", "seq", "raw", "raw_thr0.5") if x in r]:
            s.setdefault(f"{role}_face", 0); s.setdefault(f"{role}_cube", 0); v = r[role]["viol"]; s[f"{role}_face"] += v["xy"] + v.get("xz", 0) + v.get("yz", 0); s[f"{role}_cube"] += v.get("cube", 0)
        s["ddfp_ne_seq_cases"] += r["ddfp_vs_seq_differing_cells"] > 0; s["ddfp_vs_seq_cells"].append(r["ddfp_vs_seq_differing_cells"])
    json.dump({"zip_sha256_drive_cremi": sha, "n_outputs": len(recs), "summary": summ, "records": recs}, open(out, "w"), indent=1)
    print(json.dumps(summ, indent=1))

if __name__ == "__main__":
    if len(sys.argv) not in (4, 5): sys.exit(__doc__)
    main(*sys.argv[1:])
