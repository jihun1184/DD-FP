#!/usr/bin/env python3
"""Generate provenance-coupled Fig A/B from one exported DRIVE or CREMI case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qualitative_common import command_string, sha256_file, software_versions, write_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dpi", type=int, default=240)
    p.add_argument("--selection-json", type=Path, default=None,
                   help="selected_cases.json; adds percentile/rank provenance for frozen figures")
    return p.parse_args()


def load(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def take(a: np.ndarray, axis: int | None, index: int | None) -> np.ndarray:
    return a if a.ndim == 2 or axis is None else np.take(a, int(index), axis=int(axis))


def display_image(a: np.ndarray) -> np.ndarray:
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        return a
    finite = a[np.isfinite(a)]
    if not finite.size:
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.percentile(finite, [1, 99])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0, 1)


def contour_if_present(ax, mask, **kwargs):
    if np.any(mask) and np.any(~mask):
        ax.contour(mask.astype(float), levels=[0.5], **kwargs)


def roi_box(ax, roi, color="cyan"):
    y0, y1, x0, x1 = roi
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "--", color=color, linewidth=1.1)


def main() -> int:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prov_path = case_dir / "case_provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    dataset, case_id = prov["dataset"], prov["case_id"]
    is_drive = dataset == "DRIVE"
    raw = load(case_dir / "raw_no_interp_expanded.npy")
    seq = load(case_dir / "sequential_fp.npy")
    dd = load(case_dir / "corrected_ddfp.npy")
    diff = load(case_dir / "seq_vs_ddfp_difference.npy")
    axis, index = prov.get("axis"), prov.get("slice_index")
    raw2, seq2, dd2, diff2 = (take(a, axis, index) for a in (raw, seq, dd, diff))
    if is_drive:
        source = load(case_dir / "fundus_expanded_rgb.npy")
    else:
        em = case_dir / "raw_em_patch_expanded.npy"
        source = take(load(em), axis, index) if em.exists() else raw2
    source_display = display_image(source)
    roi = [int(x) for x in prov["roi_y0_y1_x0_x1"]]
    y0, y1, x0, x1 = roi
    selection = None
    if args.selection_json:
        records = json.loads(args.selection_json.read_text(encoding="utf-8"))
        selection = next((r for r in records if r.get("case_id") == case_id), None)
        if selection is None:
            raise RuntimeError(f"case {case_id} absent from {args.selection_json}")

    # Fig A: same spatial grid and same deterministic ROI in every panel.
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.2), dpi=args.dpi)
    source_cmap = None if source_display.ndim == 3 else "gray"
    panels = (
        (source_display, "Source image", source_cmap),
        (raw2, "Raw / no-interp", "gray"),
        (seq2, "Sequential FP", "gray"),
        (dd2, "Corrected DD-FP", "gray"),
        (diff2, "Thresholded difference", "magma"),
    )
    for ax, (arr, title, cmap) in zip(axes.flat[:5], panels):
        ax.imshow(arr, cmap=cmap, interpolation="nearest" if cmap else None)
        roi_box(ax, roi)
        ax.set_title(title)
        ax.axis("off")
    ax = axes.flat[5]
    ax.imshow(source_display[y0:y1, x0:x1], cmap=None if source_display.ndim == 3 else "gray")
    contour_if_present(ax, (seq2[y0:y1, x0:x1] >= 0.5), colors="gold", linestyles="--", linewidths=1.6)
    contour_if_present(ax, (dd2[y0:y1, x0:x1] >= 0.5), colors="cyan", linestyles="-", linewidths=1.6)
    ax.set_title("Deterministic ROI overlay")
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], color="gold", linestyle="--", lw=1.8, label="Sequential FP"),
        plt.Line2D([0], [0], color="cyan", linestyle="-", lw=1.8, label="Corrected DD-FP"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2)
    fig.suptitle(f"{dataset} — {case_id}")
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig_a = args.output_dir / f"{case_id}_FigA.png"
    fig.savefig(fig_a, bbox_inches="tight")
    plt.close(fig)

    # Fig B: anatomical/source context plus raw and corrected contours.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.1), dpi=args.dpi)
    axes[0].imshow(source_display, cmap=None if source_display.ndim == 3 else "gray")
    contour_if_present(axes[0], raw2 >= 0.5, colors="gold", linestyles="--", linewidths=1.1)
    contour_if_present(axes[0], dd2 >= 0.5, colors="cyan", linestyles="-", linewidths=1.1)
    roi_box(axes[0], roi, color="red")
    axes[0].set_title("Full context")
    axes[0].axis("off")
    axes[1].imshow(source_display[y0:y1, x0:x1], cmap=None if source_display.ndim == 3 else "gray")
    contour_if_present(axes[1], raw2[y0:y1, x0:x1] >= 0.5, colors="gold", linestyles="--", linewidths=1.8)
    contour_if_present(axes[1], dd2[y0:y1, x0:x1] >= 0.5, colors="cyan", linestyles="-", linewidths=1.8)
    axes[1].set_title("Deterministic ROI")
    axes[1].axis("off")
    handles = [
        plt.Line2D([0], [0], color="gold", linestyle="--", lw=1.8, label="Raw / no-interp contour"),
        plt.Line2D([0], [0], color="cyan", linestyle="-", lw=1.8, label="Corrected DD-FP contour"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2)
    fig.suptitle(f"{dataset} — {case_id}")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig_b = args.output_dir / f"{case_id}_FigB.png"
    fig.savefig(fig_b, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "schema": "ddfp-qualitative-figure-v1",
        "dataset": dataset,
        "case_id": case_id,
        "source_case_provenance": str(prov_path),
        "source_case_provenance_sha256": sha256_file(prov_path),
        "axis": axis,
        "slice_index": index,
        "roi_y0_y1_x0_x1": roi,
        "selection_metric": selection.get("selection_metric") if selection else prov["metrics"]["selection_metric"],
        "selection_score": selection.get("selection_score") if selection else prov["metrics"]["selection_score"],
        "percentile_target": selection.get("target_percentile") if selection else None,
        "selection_label": selection.get("selection_label") if selection else None,
        "actual_rank_one_based": selection.get("actual_rank_one_based") if selection else None,
        "total_audited_cases": selection.get("total_cases") if selection else None,
        "figure_status": "frozen_representative" if selection else "pilot_unfrozen",
        "source_paths": prov.get("source_paths"),
        "input_sha256": prov.get("input_sha256"),
        "r3_source_sha256": prov.get("r3_source_sha256"),
        "generation_command": command_string(),
        "software_versions": software_versions({"matplotlib": matplotlib.__version__}),
        "outputs": {
            "fig_a": {"path": str(fig_a.resolve()), "sha256": sha256_file(fig_a)},
            "fig_b": {"path": str(fig_b.resolve()), "sha256": sha256_file(fig_b)},
        },
        "interpretation_scope": "Visual comparison, anatomical/source alignment, and difference localization only.",
    }
    write_json(args.output_dir / f"{case_id}_figure_provenance.json", metadata)
    print(f"[PASS] {fig_a}")
    print(f"[PASS] {fig_b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
