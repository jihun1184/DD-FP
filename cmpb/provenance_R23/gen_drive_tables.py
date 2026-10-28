#!/usr/bin/env python3
"""gen_drive_tables.py CSV OUTDIR -- writes drive_core_table.tex (main text) and drive_perimage_table.tex (Supplement S-G.4)
from the per-image record; every number in the tables is computed here, none is typed."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive_stats import compute
csv_path, outdir = sys.argv[1], sys.argv[2]
a = compute(csv_path)
os.makedirs(outdir, exist_ok=True)

def f2(x): return "%.2f" % x
def g(x):
    s = ("%g" % x) if abs(x - round(x)) > 1e-9 else str(int(round(x)))
    return s
def th(n):
    n = int(round(n)); s = "{:,}".format(n); return s.replace(",", "{,}")
NAME = {"no_interp": r"\texttt{no\_interp}", "naive_interp": r"\texttt{naive\_interp}", "seq_fp": r"\texttt{seq\_fp}", "ddfp": r"\texttt{ddfp}"}
LAB = {"no_interp": "No interpolation", "naive_interp": r"Naive interpolation", "seq_fp": "Sequential FP", "ddfp": "DD-FP"}
rows = []
for m in ("no_interp", "naive_interp", "seq_fp", "ddfp"):
    d = a[m]
    if m == "naive_interp":
        crit = r"n/a$^{\dagger}$"
    else:
        crit = "%s (%d/%d)" % (th(d["viol"]["sum"]), d["imgs_with_viol"], d["n"])
    tsi = "0" if d["tsi_mean"] == 0 else f2(d["tsi_mean"])
    chi = "%d\\%%" % round(d["chi_flip_pct"])
    rows.append("%s & %s & %s & %s & %s & %s & %s \\\\" % (LAB[m], f2(d["b4"]["mean"]), f2(d["b8"]["mean"]), g(round(d["cc"]["mean"], 2)) if d["cc"]["mean"] else "0", crit, tsi, chi))
core = r"""\begin{table}[tbp]
\centering\footnotesize
\caption{\textbf{DRIVE, 20 images: component counts and DWC-critical configurations of the field used by each method} (earlier record, tier~C; aggregates recomputed in R2.3 from the per-image record, the Part-B pipeline was not re-run). $\beta_0^{4},\beta_0^{8}$: mean numbers of $4$- and $8$-connected components of the (thresholded) field; CC $=|\beta_0^{4}-\beta_0^{8}|$; critical blocks: DWC-critical $2\times2$ blocks summed over the images (images with at least one in brackets); TSI: mean threshold sensitivity index; $\chi_{\mathrm{flip}}$: share of images with a sign change of the Euler characteristic. Sequential FP and DD-FP agree in $\beta_0^{4}$, $\beta_0^{8}$, CC, $\chi$ and the critical-block count in all 20 images. $^{\dagger}$The naive-interpolation field is real-valued; the interval criterion applied to it as given returns no violation, which is not comparable with the binary rows, so no count is shown. Component counts do not show anatomical connectivity (\cref{sec:partb-2d}).}
\label{tab:drive-core}
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
Method & $\beta_0^{4}$ & $\beta_0^{8}$ & CC & Critical blocks & TSI & $\chi_{\mathrm{flip}}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(outdir, "drive_core_table.tex"), "w").write(core)

def cell(d, key, nd=2, mean=True):
    x = d[key]
    mean_s = ("%.2f" % x["mean"]) if mean else ""
    return "%s (%s / %s / %s)" % (mean_s, g(x["min"]), g(x["med"]), g(x["max"]))
lines = []
for m in ("no_interp", "naive_interp", "seq_fp", "ddfp"):
    d = a[m]
    crit = "n/a$^{\\dagger}$" if m == "naive_interp" else "%.2f (%s / %s / %s)" % (d["viol"]["mean"], g(d["viol"]["min"]), g(d["viol"]["med"]), g(d["viol"]["max"]))
    lines.append("%s & %s & %s & %s \\\\" % (NAME[m], cell(d, "b4"), cell(d, "b8"), crit))
per = r"""\begin{table}[H]
\centering
\footnotesize
\caption{DRIVE (20 images): mean (minimum / median / maximum over the images) of the $4$- and $8$-connected component counts and of the number of DWC-critical $2\times2$ blocks of the field used by each method. Computed in R2.3 from the per-image record \texttt{evidence\_drive/exp\_b1\_corrected.csv} (tier~C record; the Part-B pipeline was not re-run). This table replaces the table of means of R2.2, whose means are the first numbers of each cell. $^{\dagger}$The naive-interpolation field is real-valued and the interval criterion applied to it as given returns no violation; this is not comparable with the binary rows (see the note in the main-text table).}
\label{tab:S-drive-b0}
\begin{tabular}{@{}llll@{}}
\toprule
Method & $\beta_0^{4}$ & $\beta_0^{8}$ & Critical blocks per image \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(outdir, "drive_perimage_table.tex"), "w").write(per)
print("tables written to", outdir)
