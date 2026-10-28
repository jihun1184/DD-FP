#!/usr/bin/env python3
"""Main-text figure 'Frontier activity and launch structure' (promoted from Supplement Figure S6 a and d, V7).
Same stored data as tools/make_fig_p4_s2.py (data/p4_exploratory/s2/, SHA-256 in SHA256SUMS_S2.txt); no new measurement and no derived quantity beyond
those plotted in the former S6(a) and S6(d, upper). Panel (a): one volume (00158), whole-grid kernel; panel (b): three volumes, exact worklist.
Usage: python tools/make_fig_frontier_main.py"""
import csv, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "..", "data", "p4_exploratory", "s2"); O = os.path.join(H, "..", "paper", "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 6.4, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#8d8c86"
subs = ["00158", "00228", "01280"]
rd = lambda n: list(csv.DictReader(open(os.path.join(D, n))))
S2 = {r["subject"]: r for r in rd("B2_STRUCTURAL_COUNTS.csv")}
fig = plt.figure(figsize=(4.9, 2.7), dpi=300)
gs = fig.add_gridspec(1, 2, left=0.115, right=0.965, top=0.86, bottom=0.30, wspace=0.40, width_ratios=[1.25, 0.95])
# (a) whole-grid kernel: active cells per round, one volume
ax = fig.add_subplot(gs[0, 0])
rr = rd("t_00158_B5_lower_rounds.csv"); A = np.array([int(x["A"]) for x in rr]); vis = np.array([int(x["visit"]) for x in rr]); big = vis == vis[0]; idx = np.arange(len(A))
ax.scatter(idx[~big], A[~big], s=1.2, c=GREY, lw=0, rasterized=True, label="other visits")
ax.scatter(idx[big], A[big], s=1.2, c=BLUE, lw=0, rasterized=True, label="level-0 visit")
N = int(S2["00158"]["N_padded"]); ax.axhline(N, color=ORANGE, lw=1.1); ax.set_yscale("log"); ax.set_ylim(0.7, 6e8)
ax.text(len(A) * 0.985, N * 0.55, "grid size $N$ (threads per round)", ha="right", va="top", fontsize=6.0, color=ORANGE)
ax.text(len(A) * 0.985, 1.5e6, f"layer size: median {int(float(S2['00158']['median_layer']))},\n90% $\\leq$ {int(float(S2['00158']['p90_layer'])):,} cells", ha="right", va="center", fontsize=6.3, linespacing=1.3)
ax.set_xlabel(f"closure round $r$ ($R$ = {len(A):,})"); ax.set_ylabel("active cells $A_r$")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), frameon=False, markerscale=4, fontsize=6.3, handletextpad=0.2)
ax.set_title("(a) Whole-grid kernel, volume 00158", loc="left", fontsize=6.9, fontweight="bold")
# (b) exact worklist: launches = layers + visits, three volumes
ax = fig.add_subplot(gs[0, 1])
for k, s in enumerate(subs):
    R, V = int(S2[s]["R_layers"]), int(S2[s]["V_visits"]); tot = R + V
    ax.barh(k, V, color=ORANGE, edgecolor="white", lw=0.5, label="$V$ visits" if k == 0 else None)
    ax.barh(k, tot - V, left=V, color=BLUE, edgecolor="white", lw=0.5, label="$R$ layers" if k == 0 else None)
    ax.text(tot + 350, k, f"{tot:,}", va="center", fontsize=6.3)
ax.set_yticks(range(3)); ax.set_yticklabels(subs); ax.set_xlim(0, 28000); ax.set_xticks([0, 10000, 20000]); ax.invert_yaxis(); 
ax.set_xlabel("launches per volume, $R+V$")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), frameon=False, fontsize=5.6, handlelength=1.0, ncol=2, columnspacing=0.8)
ax.set_title("(b) Exact worklist, 3 volumes", loc="left", fontsize=6.5, fontweight="bold", x=-0.05)
fig.savefig(os.path.join(O, "fig_frontier_main.pdf"), dpi=300, bbox_inches="tight", pad_inches=0.03)
nums = {"N_padded_00158": N, "R_rounds_00158": len(A), "median_layer": float(S2["00158"]["median_layer"]), "p90_layer": float(S2["00158"]["p90_layer"]),
        "launches_R_plus_V": {s: int(S2[s]["R_layers"]) + int(S2[s]["V_visits"]) for s in subs}}
json.dump(nums, open(os.path.join(O, "fig_frontier_main_numbers.json"), "w"), indent=1); print(json.dumps(nums))
