#!/usr/bin/env python3
"""Generate the memory tables of the CMPB final manuscript from the stored recomputation (p2_memory_recompute.json).
Output: figsrc/mem_main_table.tex (main text), figsrc/mem_full_table.tex (supplement), tables_numbers.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data"); O = os.path.join(HERE, "..", "figsrc")
agg = json.load(open(os.path.join(D, "p2_memory_recompute.json")))["aggregate"]
S3 = {(r["m"], r["method"], r["K"]): r for r in agg if r["src"] == "S3"}
R8C = {(r["m"], r["method"], r["K"]): r for r in agg if r["src"] == "R8C"}
MIB = 1024.0
def cell(v, a): return "%.0f (%.3f)" % (v, v / a)
out = {"B_alloc_ratio": [], "C_alloc_ratio": [], "B_res_ratio": [], "C_res_ratio": []}
# ranges over all S3 B / C rows (ratios to the A row of the same depth)
for (m, meth, K), r in S3.items():
    if meth == "A": continue
    A = S3[(m, "A", 1)]
    out[meth + "_alloc_ratio"].append(r["live_MiB_med"] / A["live_MiB_med"])
    out[meth + "_res_ratio"].append(r["res_MiB_med"] / A["res_MiB_med"])
summary = {k: [round(min(v), 3), round(max(v), 3), len(v)] for k, v in out.items()}
json.dump(summary, open(os.path.join(HERE, "..", "tables_numbers.json"), "w"), indent=1)
print(summary)
Ne = {1: 70897269, 2: 142023979, 4: 284277399}
sel = {1: 16, 2: 32, 4: 64}
L = [r"\begin{tabularx}{\linewidth}{@{}l l r *{3}{>{\centering\arraybackslash}X}@{}}", r"\toprule",
     r"Depth & Method & $K$ & Allocated & Reserved & Device\\", r"\midrule"]
for m in (1, 2, 4):
    A = S3[(m, "A", 1)]
    for i, cfg in enumerate(("A", "B", "C")):
        K = 1 if cfg == "A" else sel[m]
        r = S3[(m, cfg, K)]
        lab = r"$\times%d$" % m if i == 0 else ""
        L.append("%s & %s & %d & %s & %s & %s" % (lab, cfg, K,
            cell(r["live_MiB_med"], A["live_MiB_med"]),
            cell(r["res_MiB_med"], A["res_MiB_med"]),
            cell(r["dev_abs_MiB_med"], A["dev_abs_MiB_med"])) + r"\\")
    if m != 4: L.append(r"\addlinespace[3pt]")
L += [r"\bottomrule", r"\end{tabularx}"]
open(os.path.join(O, "mem_main_table.tex"), "w").write("\n".join(L) + "\n")

# full table: all S3 rows plus the R8C rows that S3 does not contain. Rows of R8C (3 subjects) and S3b (2 subjects) that
# repeat an S3 key have identical allocated and reserved peaks (asserted); their device/host peaks differ and are summarised
# in the note of the supplement (maximum absolute differences are written to tables_numbers.json).
F = []
F.append(r"\begin{tabular}{@{}lllrrrr@{}}")
F.append(r"\toprule")
F.append(r"Depth & Config. & $K$ & allocated & reserved & device & host HWM\\")
F.append(r"\midrule")
S3k = {(r["m"], r["method"], r["K"]): r for r in agg if r["src"] == "S3"}
dd = [0.0, 0.0]
for r in agg:
    k = (r["m"], r["method"], r["K"])
    if r["src"] != "S3" and k in S3k:
        q = S3k[k]
        assert r["live_MiB_med"] == q["live_MiB_med"] and r["res_MiB_med"] == q["res_MiB_med"], k
        dd[0] = max(dd[0], abs(r["dev_abs_MiB_med"] - q["dev_abs_MiB_med"])); dd[1] = max(dd[1], abs(r["host_hwm_MiB_med"] - q["host_hwm_MiB_med"]))
rows = sorted([r for r in agg if r["src"] == "S3" or (r["src"] == "R8C" and (r["m"], r["method"], r["K"]) not in S3k)],
              key=lambda r: (r["m"], r["method"], r["K"]))
for r in rows:
    dag = "$^{\\dagger}$" if r["src"] != "S3" else ""
    F.append("$\\times%d$ & %s & %d%s & %.1f & %.1f & %.1f & %.1f\\\\" % (r["m"], r["method"], r["K"], dag,
             r["live_MiB_med"], r["res_MiB_med"], r["dev_abs_MiB_med"], r["host_hwm_MiB_med"]))
F.append(r"\bottomrule")
F.append(r"\end{tabular}")
open(os.path.join(O, "mem_full_table.tex"), "w").write("\n".join(F) + "\n")
tn = json.load(open(os.path.join(HERE, "..", "tables_numbers.json"))); tn["full_table_repeat_max_abs_diff_MiB_device_host"] = [round(dd[0], 1), round(dd[1], 1)]
json.dump(tn, open(os.path.join(HERE, "..", "tables_numbers.json"), "w"), indent=1)
print(len(rows), "rows in full table", dd)

# ---- G5 per-volume table (supplement S-I) from the stored subject table
import csv
sub = list(csv.DictReader(open(os.path.join(D, "subject_table.csv"))))
G = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
     r"Volume & frozen (s) & AF (s) & pile (s) & frozen/AF & frozen/pile & AF/pile\\", r"\midrule"]
for r in sub:
    G.append("%s & %.2f & %.2f & %.2f & %.2f & %.2f & %.2f\\\\" % (r["subject"].replace("BraTS2021_", ""), float(r["t_frozen_s"]), float(r["t_af_s"]),
             float(r["t_pile_s"]), float(r["speedup_af"]), float(r["speedup_pile"]), float(r["r_af_over_pile"])))
G += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(O, "g5_table.tex"), "w").write("\n".join(G) + "\n")
print(len(sub), "G5 rows")
