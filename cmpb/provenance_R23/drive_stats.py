"""DRIVE aggregates recomputed from the per-image record evidence_drive/exp_b1_corrected.csv (no algorithm is run)."""
import csv, statistics as st

METHODS = ["no_interp", "naive_interp", "seq_fp", "ddfp"]

def load(csv_path):
    return [r for r in csv.DictReader(open(csv_path, encoding="utf-8")) if r["dataset"] == "drive"]

def col(rows, m, k):
    return [float(r[k]) for r in rows if r["preprocessing"] == m]

def compute(csv_path):
    rows = load(csv_path)
    out = {}
    for m in METHODS:
        R = [r for r in rows if r["preprocessing"] == m]
        d = {"n": len(R)}
        for key, k in (("b4", "beta0_4conn"), ("b8", "beta0_8conn"), ("cc", "cc"), ("viol", "dwc_violations")):
            v = col(rows, m, k)
            d[key] = {"mean": st.mean(v), "min": min(v), "med": st.median(v), "max": max(v), "sum": sum(v)}
        d["tsi_mean"] = st.mean(col(rows, m, "tsi"))
        d["tsi_sd"] = st.pstdev(col(rows, m, "tsi")); d["tsi_max"] = max(col(rows, m, "tsi"))
        d["b0c_mean"] = st.mean(col(rows, m, "b0_consistency"))
        d["chi_flip_pct"] = 100 * st.mean(col(rows, m, "chi_sign_flip"))
        d["viol_rate_mean"] = st.mean(col(rows, m, "dwc_viol_rate"))
        d["imgs_with_viol"] = sum(1 for r in R if int(r["dwc_violations"]) > 0)
        d["binary_imgs"] = sum(int(r["is_binary"]) for r in R)
        out[m] = d
    S = {r["sample"]: r for r in rows if r["preprocessing"] == "seq_fp"}
    D = {r["sample"]: r for r in rows if r["preprocessing"] == "ddfp"}
    out["seq_ddfp_identical_20"] = sum(all(S[k][c] == D[k][c] for c in ("beta0_4conn", "beta0_8conn", "dwc_violations", "chi", "cc", "tsi")) for k in S)
    nv = sorted((int(r["dwc_violations"]), r["sample"]) for r in rows if r["preprocessing"] == "no_interp")
    out["no_interp_violation_rank"] = {s: (i + 1, c) for i, (c, s) in enumerate(nv)}
    return out
