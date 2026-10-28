"""Recompute the DRIVE and CREMI-2D aggregates quoted in the paper from the stored per-image record.

Input : data/exp_b1_corrected.csv  (per-image record of the earlier production evaluation; SHA-256 b64e4afafac84fe9...)
        data/FIG4_DWC_MECHANISM_SUMMARY_FROZEN.json  (frozen selection audit of the DRIVE case in Figure S1/S2)
Output: a table of checks on stdout; exit status 1 if any check fails.

No algorithm is run and no raw image array is read: every value is an aggregate of the stored rows.
The expected constants are the values printed in the manuscript (Section 5.7, Supplement S-C.1); they
are the same constants that the earlier script provenance_R23/verify_r23_numbers.py (62 checks, 0 failures)
compared with this file.  Usage:  python3 -I tools/recompute_drive_cremi2d.py [DATA_DIR]
"""
import csv, json, os, statistics as st, sys, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive_stats import compute

D = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
CSV = os.path.join(D, "exp_b1_corrected.csv")
rows = []
def chk(name, got, want):
    rows.append((name, got == want, "" if got == want else "got %r != want %r" % (got, want)))

sha = hashlib.sha256(open(CSV, "rb").read()).hexdigest()
chk("SHA-256 prefix of exp_b1_corrected.csv", sha[:16], "b64e4afafac84fe9")
a = compute(CSV); n, s_, d_, nv = a["naive_interp"], a["seq_fp"], a["ddfp"], a["no_interp"]
f2 = lambda x: "%.2f" % x
chk("DRIVE images per method", (nv["n"], n["n"], s_["n"], d_["n"]), (20, 20, 20, 20))
chk("DRIVE no_interp mean beta0^4 / beta0^8 / CC", (f2(nv["b4"]["mean"]), f2(nv["b8"]["mean"]), f2(nv["cc"]["mean"])), ("451.85", "3.60", "448.25"))
chk("DRIVE naive mean beta0^4 / beta0^8 / CC", (f2(n["b4"]["mean"]), f2(n["b8"]["mean"]), f2(n["cc"]["mean"])), ("3.60", "3.60", "0.00"))
chk("DRIVE seq_fp and ddfp mean beta0^4 / beta0^8", (f2(s_["b4"]["mean"]), f2(s_["b8"]["mean"]), f2(d_["b4"]["mean"]), f2(d_["b8"]["mean"])), ("451.85",) * 4)
chk("DRIVE no_interp DWC-critical blocks, total / images affected", (int(nv["viol"]["sum"]), nv["imgs_with_viol"]), (9544, 20))
chk("DRIVE seq_fp / ddfp critical blocks, total", (int(s_["viol"]["sum"]), int(d_["viol"]["sum"])), (0, 0))
chk("DRIVE seq_fp == ddfp per image (counts, CC, chi, TSI, violations)", a["seq_ddfp_identical_20"], 20)
chk("DRIVE naive: TSI mean / chi_flip %", (f2(n["tsi_mean"]), round(n["chi_flip_pct"])), ("222.74", 95))
chk("DRIVE ddfp b0-consistency / chi_flip", ("%.6f" % d_["b0c_mean"], d_["chi_flip_pct"]), ("1.000000", 0.0))
chk("DRIVE ratio of mean beta0^8 (ddfp / no_interp), rounded", round(d_["b8"]["mean"] / nv["b8"]["mean"]), 126)
chk("DRIVE binary fields: no_interp, seq_fp, ddfp, naive", (nv["binary_imgs"], s_["binary_imgs"], d_["binary_imgs"], n["binary_imgs"]), (20, 20, 20, 0))
J = json.load(open(os.path.join(D, "FIG4_DWC_MECHANISM_SUMMARY_FROZEN.json"))); fa = J["frozen_audit"]
chk("frozen audit: total no_interp violations == record sum", fa["no_interp_total_violations"], int(nv["viol"]["sum"]))
chk("DRIVE blocks (1167-1)(1129-1) x 20 == frozen audit total", 20 * (1167 - 1) * (1129 - 1), fa["total_2x2_blocks"])
cr = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["dataset"] == "cremi"]
by = lambda m: [r for r in cr if r["preprocessing"] == m]
chk("CREMI-2D patches per method", tuple(len(by(m)) for m in ("no_interp", "naive_interp", "seq_fp", "ddfp")), (50, 50, 50, 50))
v66 = sum(int(r["dwc_violations"]) for r in by("no_interp"))
chk("CREMI-2D no_interp violating face blocks / patches affected", (v66, sum(1 for r in by("no_interp") if int(r["dwc_violations"]) > 0)), (66, 35))
chk("CREMI-2D ddfp and seq_fp violations; ddfp max TSI", (sum(int(r["dwc_violations"]) for r in by("ddfp")), sum(int(r["dwc_violations"]) for r in by("seq_fp")), max(float(r["tsi"]) for r in by("ddfp"))), (0, 0, 0.0))
chk("CREMI-2D mean of stored per-patch rates (x1e-6)", "%.1f" % (st.mean(float(r["dwc_viol_rate"]) for r in by("no_interp")) / 1e-6), "5.2")
chk("CREMI-2D pooled rate, violations / face blocks (x1e-6)", "%.1f" % (v66 / (50 * 260100) / 1e-6), "5.1")

w = max(len(r[0]) for r in rows)
for name, ok, det in rows:
    print("%-*s | %-4s | %s" % (w, name, "OK" if ok else "FAIL", det))
bad = sum(1 for r in rows if not r[1])
print("TOTAL %d  FAIL %d" % (len(rows), bad))
sys.exit(1 if bad else 0)
