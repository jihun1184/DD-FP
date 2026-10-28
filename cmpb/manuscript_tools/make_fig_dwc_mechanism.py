#!/usr/bin/env python3
"""Main-text DWC-mechanism figure (DRIVE case 33_manual1, anchor (572,512)).
Re-assembles the six raster panels (3 field crops, 3 local zooms) that were extracted losslessly from
paper/figures/fig_dwc_mechanism_r6.pdf (paper/figure_sources/dwc_mechanism_panels/); the exact 2x2 states are the frozen
audit values (raw 10/01; sequential FP and DD-FP 10/00). No experiment is rerun; only the layout is set for single-column width.
Usage: python tools/make_fig_dwc_mechanism.py"""
import os
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
H = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(H, "..", "paper", "figure_sources", "dwc_mechanism_panels")
OUT = os.path.join(H, "..", "paper", "figures", "fig_dwc_mechanism_main.pdf")
im = [Image.open(os.path.join(SRC, f"m-{i:03d}.png")).convert("RGB") for i in range(6)]
RAW, SEQ, DD = np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 0]]), np.array([[1, 0], [0, 0]])
C_NO, C_SEQ, C_DD, C_CRIT, C_REP = "#3F6AA1", "#4A4A4A", "#00A676", "#8B1E3F", "#3F5F9E"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "image.interpolation": "none"})
fig = plt.figure(figsize=(4.77, 3.05), facecolor="white")
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 0.8, 0.8], hspace=0.26, wspace=0.10, left=0.085, right=0.995, top=0.93, bottom=0.075)
ax = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]
for c, (t, col) in enumerate(zip(["No interpolation", "Sequential FP", "DD-FP"], [C_NO, C_SEQ, C_DD])):
    ax[0][c].imshow(im[c]); ax[1][c].imshow(im[3 + c])
    ax[0][c].set_title(t, fontsize=6.6, fontweight="bold", color=col, pad=2.5)
    for r in (0, 1): ax[r][c].axis("off")


def mat(a, m, accent, ls, label):
    a.set_xlim(0, 2); a.set_ylim(0, 2); a.set_aspect("equal"); a.axis("off")
    for r in range(2):
        for c in range(2):
            v = int(m[r, c])
            a.add_patch(Rectangle((c, 1 - r), 1, 1, fc="white" if v else "black", ec="0.35", lw=0.6))
            a.text(c + 0.5, 1.5 - r, str(v), ha="center", va="center", fontsize=8.5, fontweight="bold", color="black" if v else "white")
    a.add_patch(Rectangle((1, 0), 1, 1, fill=False, ec=accent, lw=1.3, ls=ls))
    a.text(1, -0.14, label, ha="center", va="top", fontsize=5.8, fontweight="bold", color=accent)


mat(ax[2][0], RAW, C_CRIT, (0, (3, 1.5)), "10/01 DWC-critical")
mat(ax[2][1], SEQ, C_REP, "-", "10/00 DWC-resolved")
mat(ax[2][2], DD, C_REP, "-", "10/00 DWC-resolved")
for k, (txt, y) in enumerate([("Field context", 0.76), ("Local zoom", 0.47), ("Exact 2×2 state", 0.19)]):
    fig.text(0.02, y, txt, rotation=90, ha="center", va="center", fontsize=6.2, fontweight="bold")
for i, a in enumerate(sum(ax, [])):
    p = a.get_position()
    if i < 6: fig.text(p.x1 - 0.004, p.y1 - 0.002, f"({'abcdefghi'[i]})", fontsize=5.6, fontweight="bold", va="top", ha="right", color="white")
    else: fig.text(p.x0, p.y1 + 0.004, f"({'abcdefghi'[i]})", fontsize=5.6, fontweight="bold", va="bottom")
p0, p1, p2 = [a.get_position() for a in ax[2]]; y = (p0.y0 + p0.y1) / 2 + 0.01
fig.text((p0.x1 + p1.x0) / 2, y, "→", ha="center", va="center", fontsize=11, color="0.3")
fig.text((p1.x1 + p2.x0) / 2, y, "=", ha="center", va="center", fontsize=11, color=C_DD, fontweight="bold")
fig.savefig(OUT, dpi=300); print(OUT, os.path.getsize(OUT))
