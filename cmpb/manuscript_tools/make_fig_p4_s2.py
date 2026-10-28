#!/usr/bin/env python3
"""Supplement figure S6 (V7): exact-worklist control flow, visits under the two tie rules and measured GPU time of one record (project P4-R2, three BraTS natives).
In V7 the former panels (a) (active cells per round) and (d, upper) (launches R + V) are in the main text (tools/make_fig_frontier_main.py).
Adapted from the project script make_fig_s2.py (paper/figure_sources/make_fig_s2_original_project.py); every plotted number is read
from data/p4_exploratory/s2/ (copies of the project files; SHA-256 in SHA256SUMS_S2.txt).
Changes against the project figure: (c) marks the 00228 CPU baseline that exceeds the pre-registered stability criterion; (d) shows V as
the visit count of the observed schedule, not as a bound enforced by the contract; text reduced, descriptions moved to the caption.
Usage: python tools/make_fig_p4_s2.py"""
import csv, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
H = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(H, "..", "data", "p4_exploratory", "s2"); O = os.path.join(H, "..", "paper", "figures")
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 6.5, "axes.linewidth": 0.6, "hatch.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY, INK, MUTE, RED = "#2a78d6", "#eb6834", "#8d8c86", "#1a1a19", "#52514e", "#b03030"
subs = ["00158", "00228", "01280"]; f = float
rd = lambda n: list(csv.DictReader(open(os.path.join(D, n))))
raw = rd("P4_R2_E_RAW.csv"); B1 = {(r["subject"], r["arm"]): r for r in rd("B1_TIME_DECOMPOSITION.csv")}; S2 = {r["subject"]: r for r in rd("B2_STRUCTURAL_COUNTS.csv")}
REG = json.load(open(os.path.join(D, "B2_LAUNCH_REGRESSION.json"))); DEC = json.load(open(os.path.join(D, "P4_R2_E_DECISION.json")))["subjects"]
vis_hi = {"00158": 597, "00228": 511, "01280": 528}   # tie = higher, GPU control-flow log (D_PROPERTY_TRANSFER_SUMMARY.json, T5)
D5 = json.load(open(os.path.join(D, "D_PROPERTY_TRANSFER_SUMMARY.json"))); assert D5["T5_all_equal"], "tie rules must give identical output"
nums = {"rho_B5": {s: DEC[s]["rho_B5"] for s in subs}, "rho_C": {s: DEC[s]["rho_C"] for s in subs}, "class": {s: DEC[s]["class"] for s in subs},
        "cpu_frozen_iqr_over_median": {s: DEC[s]["arms"]["cpu_frozen"]["wall"]["iqr_over_median"] for s in subs},
        "gpu_c1_wall_median_s": {s: DEC[s]["arms"]["gpu_c1"]["wall"]["median"] for s in subs},
        "loop_share_of_gpu_c1_wall": {s: f(B1[(s, "gpu_c1")]["t_loop_s_median"]) / f(B1[(s, "gpu_c1")]["wall_median_s"]) for s in subs},
        "R_plus_V": {s: int(S2[s]["C1_launches_R_plus_V"]) for s in subs}, "V_tie_lower": {s: int(S2[s]["V_visits"]) for s in subs}, "V_tie_higher": vis_hi,
        "launch_floor_share_of_loop": {s: REG["floor_comparison"][s]["floor_share_of_loop"] for s in subs}}
fig = plt.figure(figsize=(6.4, 5.2), dpi=300)
gs = fig.add_gridspec(2, 2, left=0.105, right=0.965, top=0.955, bottom=0.17, wspace=0.34, hspace=0.62)
# ---- (a) control flow of the exact worklist (schematic)
ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(-0.4, 10); ax.axis("off")
def box(x, y, w, h, t, ec=INK, fc="#eef2f8", fs=5.8):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=0.7)); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
def arr(x0, y0, x1, y1, ls="-", c=INK): ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=5.5, lw=0.7, color=c, ls=ls))
box(0.2, 8.0, 9.6, 1.5, "setup (one full-grid pass): histogram → cap[ℓ];\nbit-set, buckets B[0..510], seed = frame cell")
box(0.2, 5.0, 2.4, 1.8, "visit at ℓ:\nlayer 0 = B[ℓ]", fs=5.6); box(3.1, 5.0, 3.4, 1.8, "expand\n($A_r$ threads)\nout = clip(ℓ,lo,hi)", fs=5.6); box(7.1, 5.0, 2.7, 1.8, "read-back\nV empty?", fs=5.6)
arr(2.6, 5.9, 3.1, 5.9); arr(6.5, 5.9, 7.1, 5.9)
ax.add_patch(FancyArrowPatch((8.4, 5.0), (4.8, 5.0), connectionstyle="arc3,rad=-0.55", arrowstyle="-|>", mutation_scale=5.5, lw=0.7, color=BLUE)); ax.text(1.6, 3.9, "next layer (R in total)", fontsize=5.6, color=BLUE)
arr(8.4, 5.0, 8.4, 3.2); box(4.7, 1.7, 5.1, 1.4, "select (1 thread):\nnearest non-empty bucket → ℓ′", fc="#fbeee8", ec=ORANGE, fs=5.3)
arr(5.0, 2.4, 1.4, 2.4, ls="--", c=ORANGE); ax.text(1.4, 1.4, "next visit (V in total)", fontsize=5.6, color=ORANGE)
ax.text(0.2, -0.3, "orange: choice of the next level, after the queue is empty;\nblue: layering inside a visit (implementation choice)", fontsize=5.4, color=MUTE, va="bottom")
ax.set_title("(a) exact worklist (schematic)", loc="left", fontsize=7, fontweight="bold", x=-0.02)
# ---- (c) measured time
gc = gs[1, :].subgridspec(1, 2, wspace=0.50)
ax1 = fig.add_subplot(gc[0]); arms = [("cpu_frozen", "CPU\nwhole-grid"), ("gpu_b5", "GPU\nwhole-grid"), ("cpu_c1", "CPU\nEW"), ("gpu_c1", "GPU\nEW")]
for i, (a, lab) in enumerate(arms):
    for j, s in enumerate(subs):
        v = [f(r["wall_total_s"]) for r in raw if r["subject"] == s and r["arm"] == a and r["phase"] == "measure"]; y = i + (j - 1) * 0.2
        unstable = a == "cpu_frozen" and DEC[s]["arms"]["cpu_frozen"]["wall"]["iqr_over_median"] > 0.05
        if unstable: ax1.scatter(v, [y] * len(v), s=7, facecolors="none", edgecolors=RED, lw=0.5)
        else: ax1.scatter(v, [y] * len(v), s=3, c=[BLUE if a.startswith("gpu") else GREY], lw=0, alpha=0.6)
        ax1.plot([np.median(v)] * 2, [y - 0.09, y + 0.09], color=RED if unstable else INK, lw=1.0)
ax1.set_xscale("log"); ax1.set_yticks(range(4)); ax1.set_yticklabels([l for _, l in arms], fontsize=6); ax1.set_xlabel("wall time (s); bar: median of 7 reps, three volumes"); ax1.invert_yaxis(); ax1.set_ylim(4.3, -0.5)
ax1.text(0.99, 0.10, "CPU whole-grid / GPU EW: 8.2, 8.3 (00158, 01280)\n7.5 (00228, red: baseline unstable)", transform=ax1.transAxes, ha="right", va="bottom", fontsize=5.4, color=MUTE)
ax1.set_title("(c) measured time, three volumes", loc="left", fontsize=7, fontweight="bold")
ax2 = fig.add_subplot(gc[1]); ph = [("h2d_s", "upload", "#cfcfca"), ("t_ispan_s", "interval map", "#b2b2ac"), ("t_setup_s", "setup", "#94948e"), ("t_loop_s", "loop", BLUE), ("t_verify_s", "verify", "#6d6d68"), ("d2h_s", "download", "#4a4a46")]
for k, s in enumerate(subs):
    d = B1[(s, "gpu_c1")]; left = 0; wall = f(d["wall_median_s"])
    for key, lab, c in ph:
        w = f(d[key + "_median"]) / wall * 100; ax2.barh(k, w, left=left, color=c, edgecolor="white", lw=0.5, label=lab if k == 0 else None)
        if key == "t_loop_s":
            fl = REG["floor_comparison"][s]["floor_launch_sync_s"] / wall * 100
            ax2.barh(k, fl, left=left, fill=False, hatch="////", edgecolor="white", lw=0)
        left += w
    ax2.text(101, k, f"{wall:.3f} s", va="center", fontsize=5.8)
ax2.set_yticks(range(3)); ax2.set_yticklabels(subs, fontsize=6); ax2.set_xlim(0, 120); ax2.set_xlabel("share of GPU EW wall time (%)"); ax2.invert_yaxis()
from matplotlib.patches import Patch
h_, l_ = ax2.get_legend_handles_labels(); h_.insert(4, Patch(fc=BLUE, ec="white", hatch="////")); l_.insert(4, "launch floor (inferred)")
ax2.legend(h_, l_, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.34), frameon=False, fontsize=5.4, handlelength=1.0, columnspacing=0.7)
# ---- (b) visits under the two tie rules
ax4 = fig.add_subplot(gs[0, 1]); w_ = 0.36
for k, s in enumerate(subs):
    V = int(S2[s]["V_visits"]); ax4.barh(k - w_ / 2, V, height=w_, color=ORANGE, edgecolor="white", lw=0.5); ax4.barh(k + w_ / 2, vis_hi[s], height=w_, fill=False, edgecolor=ORANGE, ls="--", lw=0.8)
    ax4.text(V + 80, k - w_ / 2, f"{V:,}", va="center", fontsize=5.6); ax4.text(vis_hi[s] + 80, k + w_ / 2, f"{vis_hi[s]}", va="center", fontsize=5.6)
ax4.set_yticks(range(3)); ax4.set_yticklabels(subs, fontsize=6); ax4.invert_yaxis(); ax4.set_xlim(0, 6600); ax4.set_xlabel("visits $V$: solid tie lower (production), dashed tie higher")
ax4.set_title("(b) visits $V$ under the two tie rules", loc="left", fontsize=7, fontweight="bold")
ax4.text(0.99, 0.02, "same output in 3/3 volumes", transform=ax4.transAxes, ha="right", va="bottom", fontsize=5.4, color=MUTE)
fig.savefig(os.path.join(O, "fig_frontier_gpu.pdf"), dpi=300)
json.dump(nums, open(os.path.join(O, "fig_frontier_gpu_numbers.json"), "w"), indent=1); print(json.dumps(nums)[:600])
