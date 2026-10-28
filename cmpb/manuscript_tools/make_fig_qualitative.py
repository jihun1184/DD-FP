#!/usr/bin/env python3
"""Main-text qualitative figure (real biomedical data). Re-assembles the 15 panel rasters that were extracted
losslessly from fig_qualitative_comparison_audited_v5.pdf (paper/figure_sources/); no experiment is rerun and no
panel content is altered. Only column headers, panel letters and the contour key are set again for the
single-column width of the main text.  Usage: python tools/make_fig_qualitative.py"""
import os
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
H = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(H, "..", "paper", "figure_sources", "qualitative_panels")
OUT = os.path.join(H, "..", "paper", "figures", "fig_qualitative_main.pdf")
import numpy as np
# The rasters were extracted with pdfimages and are stored bottom-up in the source PDF: flip to the displayed orientation.
ims = [Image.open(os.path.join(SRC, f"im-{i:03d}.png")).convert("RGB").transpose(Image.FLIP_TOP_BOTTOM) for i in range(15)]


def swap_contour_colours(im, c_a=(208, 136, 0), c_b=(0, 138, 147)):
    """Exchange the two overlay line colours (alpha-preserving unmixing against a grey background).
    Used for the CREMI ROI overlay (panel o): there the dashed line (no interpolation) was drawn teal and the solid line
    (DD-FP) orange, opposite to the BraTS panel and the key. Geometry and line styles are unchanged; only hue is exchanged."""
    a = np.asarray(im).astype(float); r, g, b = a[..., 0], a[..., 1], a[..., 2]
    al_a = np.clip((r - b) / (c_a[0] - c_a[2]), 0, 1)            # orange: r-b = 208*alpha (grey background cancels)
    al_b = np.clip((b - r) / (c_b[2] - c_b[0]), 0, 1)            # teal:   b-r = 147*alpha
    al = np.where(al_a > al_b, al_a, al_b); is_a = al_a > al_b
    bg = np.where(is_a, b, r) / np.maximum(1 - al, 1e-3)
    bg = np.where(al > 0.97, 128.0, np.clip(bg, 0, 255))
    ca = np.array(c_a, float); cb = np.array(c_b, float)
    new_c = np.where(is_a[..., None], cb, ca)
    out = al[..., None] * new_c + (1 - al[..., None]) * bg[..., None]
    out = np.where((al > 0.02)[..., None], out, a)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


ims[14] = swap_contour_colours(ims[14])
titles = ["Source image", "No interpolation", "Sequential FP", "DD-FP", "ROI overlay"]
rows = ["BraTS 2021", "DRIVE", "CREMI 3D"]
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "image.interpolation": "none"})
fig = plt.figure(figsize=(4.77, 3.30), facecolor="white")
gs = fig.add_gridspec(3, 5, left=0.062, right=0.998, top=0.925, bottom=0.075, wspace=0.03, hspace=0.05)
ax = [[fig.add_subplot(gs[r, c]) for c in range(5)] for r in range(3)]
for r in range(3):
    for c in range(5):
        a = ax[r][c]; a.imshow(ims[5 * r + c]); a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values(): s.set_visible(False)
        a.text(0.04, 0.96, f"({'abcdefghijklmno'[5 * r + c]})", transform=a.transAxes, ha="left", va="top", fontsize=5.2,
               fontweight="bold", color="white", bbox=dict(boxstyle="round,pad=0.12", fc="black", ec="none", alpha=0.55))
for c, t in enumerate(titles):
    p = ax[0][c].get_position()
    fig.text((p.x0 + p.x1) / 2, p.y1 + 0.008, t, ha="center", va="bottom", fontsize=6.2, fontweight="bold" if c == 3 else "normal")
for r, lab in enumerate(rows):
    p = ax[r][0].get_position()
    fig.text(0.022, (p.y0 + p.y1) / 2, lab, ha="center", va="center", rotation=90, fontsize=6.4, fontweight="bold")
h = [Line2D([0], [0], color="#d98a00", lw=1.2, ls=(0, (3, 2)), label="no interpolation (dashed)"),
     Line2D([0], [0], color="#0e8f9a", lw=1.2, label="DD-FP (solid)"),
     Patch(fc="none", ec="#17b3bf", lw=0.9, ls="--", label="ROI box")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=5.6, bbox_to_anchor=(0.55, -0.005), handlelength=1.8, columnspacing=1.4)
fig.savefig(OUT, dpi=300); print(OUT, os.path.getsize(OUT))
