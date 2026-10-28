#!/usr/bin/env python3
"""Figures R-time and R-mem of the CMPB final manuscript (all numbers read from stored records).

Inputs (read-only, in ../data):
  timing_E6_vs_P6.json                tier E (frozen R7 run, stage E6) summary
  tierS_time_ratios_from_frozen_cli.csv   tier S cell-level times (A=single pass, B=Slabs, C=IBI final)
  rows.jsonl                          G5 v2 stage 2 (server_20261023_120411), 20 subjects x 3 methods x 6 reps
  p2_memory_recompute.json            memory records (R8C, S3, S3b)
Output: ../fig/fig_runtime.pdf, ../fig/fig_memory.pdf and ../fig/fig_numbers.json (every plotted number).
"""
import csv, json, os, statistics as st, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "data")
O = os.path.join(HERE, "..", "paper", "figures")
os.makedirs(O, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
# one encoding for the whole paper: A single pass (blue), B Slabs (orange), C IBI final (green), CPU (grey)
COL = {"seq": "#7f7f7f", "A": "#0072B2", "B": "#E69F00", "C": "#009E73", "AF": "#56B4E9", "Pile": "#CC79A7"}
nums = {}

# ---------------------------------------------------------------- runtime
e6 = json.load(open(os.path.join(D, "timing_E6_vs_P6.json")))["E6_R7_2026-10-15_16"]
means = e6["means_s"]
nums["tierE_means_s"] = means
nums["tierE_seq_over_k1_mean_of_ratios"] = e6["seq_over_k1"]["mean_of_ratios"]
nums["tierE_ddfp_over_k1_mean_of_ratios"] = e6["ddfp_over_k1"]["mean_of_ratios"]

cells = list(csv.DictReader(open(os.path.join(D, "tierS_time_ratios_from_frozen_cli.csv"))))
rows = [json.loads(l) for l in open(os.path.join(D, "rows.jsonl"), encoding="utf-8")]
timed = [r for r in rows if r.get("phase") == "timed" and r["status"] == "OK"]
by = {}
for r in timed:
    by.setdefault((r["subject"], r["method"]), []).append(r["t_total_s"])
subs = sorted({s for s, _ in by})
med = {k: st.median(v) for k, v in by.items()}
g5 = {m: [med[(s, m)] for s in subs] for m in ("frozen", "af", "pile")}
nums["g5_n_subjects"] = len(subs)
nums["g5_median_of_subject_medians_s"] = {m: st.median(v) for m, v in g5.items()}
nums["g5_ratio_frozen_over_pile_median"] = st.median([a / b for a, b in zip(g5["frozen"], g5["pile"])])
nums["g5_ratio_af_over_pile_median"] = st.median([a / b for a, b in zip(g5["af"], g5["pile"])])

with plt.rc_context({"font.size": 6.6, "axes.titlesize": 6.9, "axes.labelsize": 6.4, "xtick.labelsize": 6.0, "ytick.labelsize": 6.0, "legend.fontsize": 5.8}):
    fig, ax = plt.subplots(1, 3, figsize=(4.9, 2.75), gridspec_kw={"width_ratios": [1.25, 1.3, 1.1], "wspace": 0.80})
    a = ax[0]
    labels = ["Seq. FP\nCPU", "GPU-FP\n$K{=}1$", "DD-FP\nIBI final"]
    vals = [means["seq"], means["k1"], means["ddfp"]]
    cols = [COL["seq"], COL["A"], COL["C"]]
    a.bar(range(3), vals, color=cols, width=0.62)
    a.set_yscale("log"); a.set_ylim(5, 400)
    for i, v in enumerate(vals):
        a.text(i, v * 1.12, f"{v:.1f}", ha="center", va="bottom", fontsize=6.4)
    a.set_xticks(range(3)); a.set_xticklabels(labels, fontsize=5.5)
    a.set_ylabel("time per volume (s)")
    a.set_title("(a) CPU vs GPU\n20 volumes", loc="left", fontsize=6.9, linespacing=1.25)


    b = ax[1]
    mk = {"1": "o", "2": "s", "4": "^"}
    for c in cells:
        K = int(c["K"]); m = c["copies"]
        nums.setdefault("tierS_cells", []).append({k: c[k] for k in ("subject", "copies", "K", "B_over_A", "C_over_A")})
        b.plot(K, float(c["B_over_A"]), mk[m], color=COL["B"], ms=4.2, mfc=COL["B"] if m == "1" else "white", mew=0.9)
        b.plot(K, float(c["C_over_A"]), mk[m], color=COL["C"], ms=4.2, mfc=COL["C"] if m == "1" else "white", mew=0.9)
    b.axhline(1.0, color=COL["A"], lw=1.0, ls="--")
    b.set_xscale("log", base=2); b.set_xticks([8, 16, 32, 64, 128]); b.set_xticklabels(["8", "16", "32", "64", "128"])
    b.set_ylim(0.2, 1.7); b.set_xlabel("number of slabs $K$"); b.set_ylabel("time relative to single pass")
    b.set_title("(b) Slab decomposition\n3 volumes", loc="left", fontsize=6.9, linespacing=1.25)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=COL["B"], label="Direct assembly"),
         Line2D([], [], marker="o", ls="", color=COL["C"], label="IBI final"),
         Line2D([], [], marker="o", ls="", mfc="k", color="k", label="depth $\\times$1 (filled)"),
         Line2D([], [], marker="s", ls="", mfc="white", color="k", label="depth $\\times$2, $\\times$4 (open)")]
    b.legend(handles=h, loc="upper center", frameon=False, ncol=2, fontsize=5.8, bbox_to_anchor=(0.5, -0.30), columnspacing=0.8, handletextpad=0.2)

    c_ = ax[2]
    import random
    rnd = random.Random(3)
    for i, (m, name) in enumerate((("frozen", "whole-grid\nkernel"), ("af", "active\nfrontier"), ("pile", "active\npile"))):
        col = COL["A"] if m == "frozen" else COL["AF"] if m == "af" else COL["Pile"]
        xs = [i + rnd.uniform(-0.17, 0.17) for _ in g5[m]]
        c_.scatter(xs, g5[m], s=7, color=col, alpha=0.85, lw=0)
        mm = st.median(g5[m]); c_.plot([i - 0.28, i + 0.28], [mm, mm], color="k", lw=1.0)
        c_.text(i + (0.34 if m == "frozen" else 0), mm if m == "frozen" else mm * 0.62, f"{mm:.2f}", ha="left" if m == "frozen" else "center", va="center" if m == "frozen" else "top", fontsize=6.0)
    c_.set_yscale("log"); c_.set_ylim(1.0, 30)
    c_.set_xticks(range(3)); c_.set_xticklabels(["whole\ngrid", "active\nfrontier", "active\npile"], fontsize=5.5)
    c_.set_ylabel("time per volume (s)")
    c_.set_title("(c) GPU kernels\n20 volumes", loc="left", fontsize=6.9, linespacing=1.25)
    # comparison type: the references differ (CPU reference in (a); GPU kernels in (b), (c))
    fig.canvas.draw()
    pa, pb = ax[0].get_position(), ax[1].get_position()
    xsep = 0.5 * (pa.x1 + pb.x0) + 0.012
    from matplotlib.lines import Line2D as _L
    fig.add_artist(_L([xsep, xsep], [0.04, 0.97], transform=fig.transFigure, color="#9a9a9a", lw=0.7, ls=(0, (3, 2))))
    fig.text(0.5 * (pa.x0 + xsep), 1.04, "CPU vs GPU", ha="center", va="bottom", fontsize=6.6, style="italic", color="#444444")
    fig.text(0.5 * (xsep + ax[2].get_position().x1), 1.04, "GPU vs GPU (references differ)", ha="center", va="bottom", fontsize=6.6, style="italic", color="#444444")
    fig.savefig(os.path.join(O, "fig_runtime.pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- memory
mem = json.load(open(os.path.join(D, "p2_memory_recompute.json")))["aggregate"]
S3 = {(r["m"], r["method"], r["K"]): r for r in mem if r["src"] == "S3"}
MIB = 1.0
sel = {1: (16, 16), 2: (32, 32), 4: (64, 64)}   # slab thickness about 9.7 planes in all three
fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.7), gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.32})
a = ax[0]
metrics = [("live_MiB_med", "allocated\n(pool used peak)"), ("res_MiB_med", "reserved\n(pool total peak)"),
           ("dev_abs_MiB_med", "device\n(NVML sampled)")]
w = 0.12
x0 = 0
ticks = []
for mi, m in enumerate((1, 2, 4)):
    A = S3[(m, "A", 1)]
    Kb, Kc = sel[m]
    for ci, (cfg, K) in enumerate((("B", Kb), ("C", Kc))):
        R = S3[(m, cfg, K)]
        for ji, (key, lab) in enumerate(metrics):
            ratio = R[key] / A[key]
            x = mi * 1.0 + ci * 0.43 + ji * w
            a.bar(x, ratio, width=w * 0.92, color=COL[cfg], alpha=[1.0, 0.62, 0.32][ji], edgecolor="none")
            nums.setdefault("mem_ratios_vs_A", []).append(dict(m=m, cfg=cfg, K=K, metric=key, ratio=ratio, MiB=R[key], A_MiB=A[key]))
        a.text(mi * 1.0 + ci * 0.43 + w, 1.03, f"{cfg}\nK={K}", ha="center", va="bottom", fontsize=6.0)
    ticks.append(mi * 1.0 + 0.3)
a.axhline(1.0, color=COL["A"], ls="--", lw=1.0)
a.set_ylim(0, 1.3); a.set_xticks(ticks); a.set_xticklabels(["depth $\\times$1\n($N_e$=7.09e7)", "depth $\\times$2\n(1.42e8)", "depth $\\times$4\n(2.84e8)"])
a.set_ylabel("peak / peak of single pass (A)")
a.set_title("(a) Memory relative to A (one subject; slab $\\approx$9.7 planes)", loc="left")
hh = [plt.Rectangle((0, 0), 1, 1, color="k", alpha=al) for al in (1.0, 0.62, 0.32)]
a.legend(hh, ["allocated (live)", "reserved (pool)", "device (NVML)"], loc="upper center", frameon=False, fontsize=6.2,
         bbox_to_anchor=(0.5, -0.2), ncol=3)

b = ax[1]
Ne = {1: 70897269, 2: 142023979, 4: 284277399}
for cfg, K_of in (("A", {1: 1, 2: 1, 4: 1}), ("B", {m: sel[m][0] for m in sel}), ("C", {m: sel[m][1] for m in sel})):
    xs = [Ne[m] / 1e8 for m in (1, 2, 4)]
    ys = [S3[(m, cfg, K_of[m])]["live_MiB_med"] / 1024 for m in (1, 2, 4)]
    b.plot(xs, ys, "-o", color=COL[cfg], ms=3.5, lw=1.2, label={"A": "A single pass", "B": "B direct assembly (K=16,32,64)", "C": "C IBI final (K=16,32,64)"}[cfg])
    nums.setdefault("mem_live_GiB_vs_Ne", {})[cfg] = list(zip(xs, ys))
b.set_xlabel("expanded cells $N_e$ ($\\times10^8$)"); b.set_ylabel("allocated peak (GiB)")
b.set_title("(b) Peaks grow with $N_e$ (same slab thickness)", loc="left"); b.legend(frameon=False, loc="upper left", fontsize=6.2)
b.set_ylim(0, 8.6)
fig.savefig(os.path.join(O, "fig_memory.pdf"), bbox_inches="tight")
plt.close(fig)
json.dump(nums, open(os.path.join(O, "fig_numbers.json"), "w"), indent=1)
print("ok", {k: v for k, v in nums.items() if k.startswith("g5") or k.startswith("tierE")})
