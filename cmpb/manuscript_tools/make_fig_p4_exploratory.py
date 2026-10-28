#!/usr/bin/env python3
"""Supplement figures on the work structure of first-discovery runs on one real volume (BraTS2021_00158).

(layer/visit counts of the CPU control-flow reference of the exact worklist are read for the text; the figure is made by make_fig_p4_s2.py)
fig_round_time.pdf     : host-visible round times of the frozen kernel (project arms B5 = whole grid, B8c = four tiles);
                         single run, exploratory (P4-R1).
All plotted numbers are read from data/p4_exploratory/ (copies of the project files; SHA-256 in SHA256SUMS_SOURCES.txt).
Usage: python tools/make_fig_p4_exploratory.py
"""
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

H = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(H, "..", "data", "p4_exploratory")
O = os.path.join(H, "..", "paper", "figures")
os.makedirs(O, exist_ok=True)
BL, OR, GR, INK = "#0072B2", "#D55E00", "#8a8a8a", "#222222"   # Okabe-Ito: blue = frozen kernel, orange = EW
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 7, "axes.linewidth": 0.6,
                     "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
rd = lambda n: list(csv.DictReader(open(os.path.join(D, n))))
nums = {}

# ------------------------------------------------------------------ figure 1: work structure
lay, lah = rd("c1_00158_B5_lower_layers.csv"), rd("c1_00158_B5_higher_layers.csv")
vis, vih = rd("c1_00158_B5_lower_visits.csv"), rd("c1_00158_B5_higher_visits.csv")
A = np.array([int(r["A"]) for r in lay]); N = int(A.sum()); R = len(A); V = len(vis); Rh = len(lah); Vh = len(vih)
v0 = [r for r in vis if r["visit"] == "0"][0]; l0 = int(v0["layers"]); c0 = int(v0["cells"])
small = A < 1024
nums.update(N=N, R=R, V=V, R_higher=Rh, V_higher=Vh, visit0_layers=l0, visit0_cell_share=c0 / N,
            layers_below_1024=float(small.mean()), cells_in_small_layers=float(A[small].sum() / N), frozen_slots=R * N)

# (the layer/visit figure is replaced by tools/make_fig_p4_s2.py; the counts above are used in the supplement text)

# ------------------------------------------------------------------ figure 2: round time of the frozen kernel
audit = json.load(open(os.path.join(D, "p4r1_audit.json")))
fit = audit["N_fit"]; L = fit["L_measured_us"]
fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.35), gridspec_kw=dict(wspace=0.45, left=0.085, right=0.985, top=0.88, bottom=0.215))
for k, (op, col, name) in enumerate((("B5", BL, "whole grid"), ("B8c", "#b5543c", "four tiles"))):
    ax = axs[k]
    t = np.load(os.path.join(D, f"_t_{op}.npy")) * 1e6; p = np.load(os.path.join(D, f"_p_{op}.npy"))
    Ar = np.array([int(r["A"]) for r in rd(f"gpu_rounds_00158_{op}.csv")])
    m = Ar < 50000
    ax.scatter(p[m], t[m], s=2, alpha=0.25, color=col, lw=0, label="rounds, $|A_r|<5\\cdot10^4$", rasterized=True)
    ax.scatter(p[~m], t[~m], s=4, color="k", lw=0, label="rounds, $|A_r|\\geq5\\cdot10^4$", rasterized=True)
    ax.axhline(L, color="#cc0000", ls=":", lw=1.0, label=f"1-cell round, $L$ = {L:.1f} µs")
    ax.set_ylim(0, t.max() * 1.05); ax.set_xlabel("processed fraction of tile"); ax.set_ylabel("round time (µs)")
    ax.set_title(f"({'ab'[k]}) {name}", loc="left", fontsize=7.5, fontweight="bold")
    nums[f"{op}_rounds"] = int(len(t)); nums[f"{op}_median_round_us"] = float(np.median(t))
axs[0].legend(fontsize=5.4, loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
ax = axs[2]; pts = np.array(fit["points"], float)
ax.scatter(pts[:, 0] / 1e6, pts[:, 1], color="k", zorder=3, s=14, label="median round, one tile")
xx = np.linspace(0, 75, 50)
ax.plot(xx, fit["slope_ps_per_cell"] * xx + fit["intercept_us"], color="#555", lw=0.9,
        label=f"fit: {fit['slope_ps_per_cell']:.1f} ps/cell, {fit['intercept_us']:.0f} µs")
ax.axhline(L, color="#cc0000", ls=":", lw=1.0, label="$L$")
ax.set_xlabel("grid cells per tile (M)"); ax.set_ylabel("median round time (µs)"); ax.set_ylim(-60, 1000)
ax.legend(fontsize=5.4, loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
ax.set_title("(c) round time vs. grid", loc="left", fontsize=7.5, fontweight="bold")
nums.update(fit_slope_ps_per_cell=fit["slope_ps_per_cell"], fit_intercept_us=fit["intercept_us"], fit_r2=fit["r2"], L_us=L)
fig.savefig(os.path.join(O, "fig_round_time.pdf"), dpi=300)
json.dump(nums, open(os.path.join(O, "fig_p4_exploratory_numbers.json"), "w"), indent=1)
print(json.dumps(nums, indent=1))
