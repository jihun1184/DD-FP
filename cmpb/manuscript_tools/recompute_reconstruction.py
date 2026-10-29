"""Recompute the held-out slice reconstruction aggregates (Main Table 4, Supplement S-M) from the stored per-unit records.

Input : data/reconstruction/brats_heldout_slice_per_subject.jsonl  (1,251 BraTS 2021 subjects; one JSON object each)
        data/reconstruction/cremi_heldout_slice_per_roi.jsonl      (9 CREMI ROIs: samples A, B, C x R1..R3)
Output: a table of checks on stdout; exit status 1 if any check fails.

No algorithm is run and no image array is read: every value is an aggregate of stored per-subject / per-ROI metrics
(MAE, RMSE, signed bias, interior fraction of GPU-FP, Min, Max, Linear). The expected constants are the values printed in
the manuscript. This recomputes summaries; it does not re-run GPU-FP on the images (needs BraTS/CREMI and a CUDA GPU:
cmpb/evaluation/reconstruction_minmax/).  Usage:  python3 -I cmpb/manuscript_tools/recompute_reconstruction.py [DATA_DIR]
"""
import hashlib, json, os, statistics as st, sys

D = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
B = os.path.join(D, "reconstruction", "brats_heldout_slice_per_subject.jsonl")
C = os.path.join(D, "reconstruction", "cremi_heldout_slice_per_roi.jsonl")
M = ("fp", "min", "max", "linear")
rows = []
def chk(name, got, want):
    rows.append((name, got == want, "" if got == want else "got %r != want %r" % (got, want)))
def mean(rs, m, k): return st.fmean(r[m][k] for r in rs)
def f4(x): return "%.4f" % x

sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
chk("SHA-256 prefix, BraTS record", sha(B), "d54a9c0d39058c09")
chk("SHA-256 prefix, CREMI record", sha(C), "5ebd0d7f11c93e6b")
b = [json.loads(l) for l in open(B, encoding="utf-8") if l.strip()]
c = [json.loads(l) for l in open(C, encoding="utf-8") if l.strip()]
chk("BraTS subjects, unique", (len(b), len({r["subject"] for r in b})), (1251, 1251))
chk("BraTS mean MAE FP/Min/Max/Linear", tuple(f4(mean(b, m, "mae")) for m in M), ("10.4209", "10.5801", "10.7772", "5.3263"))
chk("BraTS mean RMSE FP/Min/Max/Linear", tuple(f4(mean(b, m, "rmse")) for m in M), ("19.3837", "19.5637", "19.6369", "11.7105"))
chk("BraTS mean signed bias FP", f4(mean(b, "fp", "bias")), "-8.9233")
chk("BraTS FP interior fraction (%)", "%.3f" % (100 * st.fmean(r["fp_interior_fraction"] for r in b)), "5.534")
mae = lambda m: [r[m]["mae"] for r in b]
chk("BraTS subjects with FP MAE below Min / Max / Linear", tuple(sum(x < y for x, y in zip(mae("fp"), mae(m))) for m in ("min", "max", "linear")), (1251, 1175, 0))
chk("BraTS Linear below FP", sum(x < y for x, y in zip(mae("linear"), mae("fp"))), 1251)
chk("BraTS subjects not below Max (Max-adverse)", sum(x >= y for x, y in zip(mae("fp"), mae("max"))), 76)
chk("BraTS reduction of mean MAE vs Min / Max (%)", tuple("%.2f" % (100 * (1 - st.fmean(mae("fp")) / st.fmean(mae(m)))) for m in ("min", "max")), ("1.50", "3.31"))
chk("BraTS first 20 records (original 20) mean FP / Min / Max / Linear MAE", tuple(f4(mean(b[:20], m, "mae")) for m in M), ("10.6352", "10.7992", "11.0042", "5.4475"))
chk("CREMI ROIs / samples", (len(c), sorted({r["sample"] for r in c})), (9, ["A", "B", "C"]))
chk("CREMI mean MAE FP/Min/Max/Linear", tuple(f4(mean(c, m, "mae")) for m in M), ("27.1087", "27.7246", "22.6912", "20.8116"))
chk("CREMI mean RMSE FP/Min/Max/Linear", tuple(f4(mean(c, m, "rmse")) for m in M), ("35.8992", "36.6408", "30.0813", "26.9859"))
cm = lambda m: [r[m]["mae"] for r in c]
chk("CREMI ROIs with FP MAE below Min / Max / Linear", tuple(sum(x < y for x, y in zip(cm("fp"), cm(m))) for m in ("min", "max", "linear")), (9, 0, 0))
chk("CREMI FP mean MAE above Max (%)", "%.2f" % (100 * (st.fmean(cm("fp")) / st.fmean(cm("max")) - 1)), "19.47")
chk("CREMI MAE by source A/B/C (FP)", tuple("%.3f" % st.fmean(r["fp"]["mae"] for r in c if r["sample"] == s) for s in "ABC"), ("20.148", "24.519", "36.659"))
chk("CREMI MAE by source A/B/C (Max)", tuple("%.3f" % st.fmean(r["max"]["mae"] for r in c if r["sample"] == s) for s in "ABC"), ("18.757", "23.034", "26.282"))
chk("CREMI mean signed bias FP, source C", (f4(mean(c, "fp", "bias")), "%.4f" % st.fmean(r["fp"]["bias"] for r in c if r["sample"] == "C")), ("-16.1984", "-24.1349"))
chk("CREMI FP interior fraction (%)", "%.2f" % (100 * st.fmean(r["fp_interior_fraction"] for r in c)), "11.29")

bad = 0
for name, ok, msg in rows:
    print("%-4s %s %s" % ("PASS" if ok else "FAIL", name, msg)); bad += (not ok)
print("%d checks, %d failures" % (len(rows), bad)); sys.exit(1 if bad else 0)
