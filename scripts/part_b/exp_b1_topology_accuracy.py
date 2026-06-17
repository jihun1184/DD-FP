"""
scripts/part_b/exp_b1_topology_accuracy.py
-------------------------------------------
Part B Experiment 1: Topology Measurement Accuracy on 2D Drive/Cremi Dataset.

Changes from original:
  [FIX-1] TSI threshold set: 5-point → 9-point (matching paper eq:tsi)
           old: (0.30, 0.40, 0.50, 0.60, 0.70)
           new: (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
  [FIX-2] Added b0_consistency metric  (b0_cons = β₀⁸ / β₀⁴)
  [FIX-3] Added chi_sign_flip metric
  [FIX-4] Wilcoxon signed-rank test (p<0.001, W=0 verification)
  [FIX-5] All values written to CSV; p-values printed in summary

Confirmed paper value: naive_interp DRIVE TSI ≈ 222.74 (9-point thresholds, historical reference only).

Usage:
    python scripts/part_b/exp_b1_topology_accuracy.py \
        --drive-gt data/DRIVE/training/1st_manual \
        --output results/part_b/exp_b1_v2_results.csv

    # DRIVE + CREMI
    python scripts/part_b/exp_b1_topology_accuracy.py \
        --drive-gt data/DRIVE/training/1st_manual \
        --cremi-gt data/CREMI/masks --cremi-limit 50 \
        --output results/part_b/exp_b1_v2_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label, generate_binary_structure

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.preprocessor import (
    NoInterpPreprocessor, NaiveInterpPreprocessor, DDFPPreprocessor,
    SeqFPPreprocessor,
)
from src.utils.benchmark_utils import verify_dwc
from types import SimpleNamespace


TSI_THRESHOLDS: tuple = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)


def _b0(binary: np.ndarray, conn: int) -> int:
    """β₀: number of connected components (4- or 8-connectivity)."""
    struct = generate_binary_structure(2, 1 if conn == 4 else 2)
    _, n = label(binary.astype(np.int32), structure=struct)
    return int(n)


def _chi(binary: np.ndarray) -> int:
    """Euler characteristic (pixel parity / CW complex formula)."""
    b = binary.astype(np.int32)
    return (int(b.sum())
            - int((b[:-1, :] & b[1:, :]).sum())
            - int((b[:, :-1] & b[:, 1:]).sum())
            + int((b[:-1, :-1] & b[1:, :-1] & b[:-1, 1:] & b[1:, 1:]).sum()))


def _tsi(interp: np.ndarray) -> float:
    """
    TSI = std[β₀⁸(θ)] for θ ∈ TSI_THRESHOLDS (9-point, 0.05 spacing).

    [FIX-1] Original code used 5 points (0.3-0.7, step 0.1) — inconsistent with paper.
    This function uses 9 points matching paper eq:tsi.
    """
    return float(np.std([
        _b0((interp >= t).astype(np.uint8), 8)
        for t in TSI_THRESHOLDS
    ]))


def _count_holes(binary: np.ndarray) -> int:
    """β₁ (number of holes) = β₀ - χ."""
    return max(0, _b0(binary, 8) - _chi(binary))


def topology_metrics(
    interp: np.ndarray,
    ref_chi: int,
    threshold: float = 0.5,
) -> dict:
    """
    Full topology metric suite.

    Parameters
    ----------
    interp    : float32 interpolated image (continuous for naive; binary for no_interp/ddfp)
    ref_chi   : Euler characteristic of the original mask (needed for chi_sign_flip)
    threshold : binarisation threshold (default 0.5)

    Returns (all at threshold=0.5 except TSI which averages over 9 thresholds)
    -------
    beta0_4conn, beta0_8conn, beta1, chi
    cc              = |β₀⁴ − β₀⁸|  (Cross-Connectivity)
    cc_zero         = int(cc == 0)
    tsi             = std[β₀⁸(θ)] over 9 thresholds  [FIX-1]
    b0_consistency  = β₀⁸ / β₀⁴   [FIX-2]
    chi_sign_flip   = 1 if sign(χ_interp) ≠ sign(χ_orig)  [FIX-3]
    is_binary       = all pixels in {0,1}
    """
    bin_img = (interp >= threshold).astype(np.uint8)
    b0_4    = _b0(bin_img, 4)
    b0_8    = _b0(bin_img, 8)
    b1      = _count_holes(bin_img)
    chi     = _chi(bin_img)
    cc      = abs(b0_4 - b0_8)

    # [FIX-2] b0_consistency: resolution-invariant topology coherence
    # DWC → β₀⁴ = β₀⁸ → ratio = 1.0 (perfect)
    # DWC violation → 8-conn merges diagonal components → ratio << 1.0
    b0_cons = round(b0_8 / b0_4, 6) if b0_4 > 0 else 1.0

    # [FIX-3] chi_sign_flip: topology inversion indicator
    # naive bilinear creates spurious face-cells → χ flips negative
    chi_flip = int((ref_chi != 0) and (chi * ref_chi < 0))

    # [FIX-1] TSI: 9-point threshold set
    tsi = _tsi(interp)

    is_bin = bool(np.all((interp == 0) | (interp == 1)))

    return {
        "beta0_4conn":    b0_4,
        "beta0_8conn":    b0_8,
        "beta1":          b1,
        "chi":            chi,
        "cc":             cc,
        "cc_zero":        int(cc == 0),
        "tsi":            round(tsi, 4),
        "b0_consistency": b0_cons,      # [FIX-2]
        "chi_sign_flip":  chi_flip,     # [FIX-3]
        "is_binary":      int(is_bin),
    }


def load_mask(path: Path) -> np.ndarray:
    """Load an arbitrary image file as a binary uint8 mask (0 or 1)."""
    try:
        from PIL import Image
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
    except ImportError:
        import imageio
        arr = imageio.imread(str(path)).astype(np.float32) / 255.0
        if arr.ndim == 3:
            arr = arr.mean(axis=2)
    return (arr > 0.5).astype(np.uint8)


def get_preprocessors(cfg: SimpleNamespace) -> dict:
    return {
        "no_interp":    NoInterpPreprocessor(cfg),
        "naive_interp": NaiveInterpPreprocessor(cfg),
        "ddfp":         DDFPPreprocessor(cfg),
        "seq_fp":       SeqFPPreprocessor(cfg),  # Sequential FP correctness reference
    }


def run_experiment(
    mask_paths: list,
    dataset_name: str,
    preprocessors: dict,
    output_rows: list,
) -> None:
    """
    Apply 3 preprocessing methods to each mask and measure topology metrics.

    [FIX-1,2,3] Pass ref_chi to topology_metrics so b0_cons and chi_flip are included.
    """
    for path in mask_paths:
        mask = load_mask(path)
        h, w = mask.shape

        fg_rate = mask.mean()
        if fg_rate < 0.001 or fg_rate > 0.999:
            print(f"  [{dataset_name}] {path.name}  SKIP "
                  f"(fg_rate={fg_rate:.3f})", flush=True)
            continue

        ref_chi = _chi(mask)

        print(f"  [{dataset_name}] {path.name}  "
              f"shape={h}×{w}  fg={fg_rate:.2f}  "
              f"ref_chi={ref_chi}", end="", flush=True)

        for prep_name, prep in preprocessors.items():
            interp_img, _ = prep(
                mask.astype(np.float32), label=None, sample_id=path.stem
            )

            topo = topology_metrics(interp_img, ref_chi=ref_chi)

            dwc = verify_dwc(mask, interp_img.astype(np.float32))

            output_rows.append({
                "dataset":        dataset_name,
                "sample":         path.name,
                "preprocessing":  prep_name,
                "orig_h":         h,
                "orig_w":         w,
                "interp_h":       interp_img.shape[0],
                "interp_w":       interp_img.shape[1],
                "fg_rate":        round(float(fg_rate), 4),
                "ref_chi":        ref_chi,
                **topo,
                "dwc_violations": dwc["n_violations"],
                "dwc_viol_rate":  round(dwc["violation_rate"], 6),
            })

        print("  done")


def wilcoxon_test(x: list, y: list, label: str = "") -> dict:
    """
    One-sided Wilcoxon signed-rank test: x < y (ddfp < baseline).

    Returns: {statistic (W), pvalue, n, significant}
    Uses scipy.stats.wilcoxon (zero_method='wilcox', alternative='less').
    """
    try:
        from scipy.stats import wilcoxon
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)

        if np.allclose(x_arr, y_arr):
            return {"W": None, "pvalue": None, "n": len(x),
                    "significant": False, "note": "all_ties"}

        result = wilcoxon(x_arr, y_arr, zero_method="wilcox",
                          alternative="less")
        return {
            "W":           float(result.statistic),
            "pvalue":      float(result.pvalue),
            "n":           len(x),
            "significant": bool(result.pvalue < 0.001),
            "note":        "",
        }
    except ImportError:
        return {"W": None, "pvalue": None, "n": len(x),
                "significant": False, "note": "scipy_not_installed"}
    except Exception as e:
        return {"W": None, "pvalue": None, "n": len(x),
                "significant": False, "note": str(e)}


# Main

def main(args: argparse.Namespace) -> None:
    cfg = SimpleNamespace(preprocessing=SimpleNamespace(
        type="ddfp",
        naive_mode="bilinear",
        ddfp_overlap=1,
        ddfp_cache_dir=None,
        no_interp_cache_dir=None,
        naive_interp_cache_dir=None,
        seq_fp_cache_dir=None,
    ))
    preps = get_preprocessors(cfg)
    rows: list = []

    print(f"\n[config] TSI thresholds ({len(TSI_THRESHOLDS)}-point): "
          f"{TSI_THRESHOLDS}")

    # ── DRIVE ────────────────────────────────────────────────────────────
    drive_dir = Path(args.drive_gt)
    if drive_dir.exists():
        masks = sorted(drive_dir.glob("*.gif")) or sorted(drive_dir.glob("*.png"))
        if masks:
            print(f"\nDRIVE: {len(masks)} masks  ({drive_dir})")
            run_experiment(masks, "drive", preps, rows)
        else:
            print(f"  [WARNING] no .gif/.png found: {drive_dir}")
    else:
        print(f"  [WARNING] DRIVE directory not found: {drive_dir}")

    # ── CREMI ────────────────────────────────────────────────────────────
    cremi_dir = Path(args.cremi_gt)
    if cremi_dir.exists():
        masks = sorted(cremi_dir.glob("*.png"))[:args.cremi_limit]
        if masks:
            print(f"\nCREMI: {len(masks)} masks  ({cremi_dir})")
            run_experiment(masks, "cremi", preps, rows)
        else:
            print(f"  [WARNING] no .png found: {cremi_dir}")
    else:
        print(f"  [INFO] CREMI not found ({cremi_dir}) — skipping")

    if not rows:
        print("\nNo masks processed.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved → {out_path}  ({len(rows)} rows)")

    _print_summary(rows)


def _print_summary(rows: list) -> None:
    import collections

    datasets = sorted(set(r["dataset"] for r in rows))

    print("\n" + "=" * 70)
    print("SUMMARY  —  Part B Exp 1 v2: Topology Measurement Accuracy")
    print(f"TSI thresholds: {len(TSI_THRESHOLDS)}-point  {TSI_THRESHOLDS}")
    print("=" * 70)

    for ds in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        n_samples = len(set(r["sample"] for r in ds_rows))
        print(f"\n──── Dataset: {ds.upper()}  ({n_samples} masks) ────")

        by_prep: dict[str, dict[str, list]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        for r in ds_rows:
            p = r["preprocessing"]
            by_prep[p]["cc"].append(r["cc"])
            by_prep[p]["tsi"].append(r["tsi"])
            by_prep[p]["b0_cons"].append(r["b0_consistency"])
            by_prep[p]["chi_flip"].append(r["chi_sign_flip"])
            by_prep[p]["dwc_rate"].append(r["dwc_viol_rate"])

        prep_order = ["no_interp", "naive_interp", "ddfp", "seq_fp"]

        # [1] CC
        print(f"\n  CC = |β₀⁴ − β₀⁸|  (thr=0.5)")
        for p in prep_order:
            vals = by_prep[p]["cc"]
            if not vals:
                continue
            zero_pct = 100.0 * sum(v == 0 for v in vals) / len(vals)
            print(f"    {p:<15}  mean={np.mean(vals):7.2f}  "
                  f"CC=0: {zero_pct:5.1f}%  (n={len(vals)})")

        # [2] TSI (9-point, [FIX-1])
        print(f"\n  TSI = std[β₀⁸(θ)]  ({len(TSI_THRESHOLDS)}-point, 0.05 spacing)  ← [FIX-1]")
        for p in prep_order:
            vals = by_prep[p]["tsi"]
            if not vals:
                continue
            print(f"    {p:<15}  mean={np.mean(vals):8.2f}  "
                  f"max={max(vals):8.2f}  std={np.std(vals):.2f}")

        # [3] b0_consistency ([FIX-2])
        print(f"\n  b0_consistency = β₀⁸ / β₀⁴  ← [FIX-2]")
        for p in prep_order:
            vals = by_prep[p]["b0_cons"]
            if not vals:
                continue
            perfect = 100.0 * sum(abs(v - 1.0) < 1e-5 for v in vals) / len(vals)
            print(f"    {p:<15}  mean={np.mean(vals):.6f}  "
                  f"=1.0: {perfect:5.1f}%")

        # [4] chi_sign_flip ([FIX-3])
        print(f"\n  chi_sign_flip = sign(χ_interp) ≠ sign(χ_orig)  ← [FIX-3]")
        for p in prep_order:
            vals = by_prep[p]["chi_flip"]
            if not vals:
                continue
            flip_pct = 100.0 * sum(vals) / len(vals)
            print(f"    {p:<15}  flip: {sum(vals)}/{len(vals)}  "
                  f"({flip_pct:.1f}%)")

        # [5] DWC violation rate
        print(f"\n  DWC violation rate")
        for p in prep_order:
            vals = by_prep[p]["dwc_rate"]
            if not vals:
                continue
            print(f"    {p:<15}  mean={np.mean(vals):.2e}  "
                  f"max={max(vals):.2e}")

        if ds == "drive" and len(by_prep["ddfp"]["tsi"]) >= 5:
            print(f"\n  Wilcoxon signed-rank (one-sided less: x<y)  ← [FIX-4]")
            print(f"  direction: all tests use \"worse side x < better side y\" (less)")
            print(f"  → CC/TSI/DWC/χ_flip: x=ddfp/seq_fp, y=baseline")
            print(f"  → b0_cons:           x=baseline, y=ddfp/seq_fp  (higher b0_cons is better)")
            print(f"  → W=0 = all pairs differ in the same direction (complete separation)")
            print(f"  → ddfp vs seq_fp: numerical equivalence check (expect all_ties)")
            print()

            # (met_name, col_key, x_prep, y_prep)
            metrics_to_test = [
                ("CC",       "cc",         "ddfp",     None),        # ddfp < baseline
                ("TSI",      "tsi",        "ddfp",     None),        # ddfp < baseline
                ("b0_cons",  "b0_cons",    None,       "ddfp"),      # baseline < ddfp
                ("DWC",      "dwc_rate",   "ddfp",     None),        # ddfp < baseline
                ("chi_flip", "chi_flip",   "ddfp",     None),        # ddfp < baseline
            ]
            for met_name, key, x_fixed, y_fixed in metrics_to_test:
                for baseline in ["no_interp", "naive_interp"]:
                    ddfp_vals = by_prep["ddfp"][key]
                    base_vals = by_prep[baseline][key]
                    if not base_vals or not ddfp_vals:
                        continue

                    # x_fixed=None → x=ddfp, y=baseline
                    # y_fixed=None → x=baseline, y=ddfp (b0_cons)
                    if x_fixed is not None:
                        x_vals, y_vals = ddfp_vals, base_vals
                    else:
                        x_vals, y_vals = base_vals, ddfp_vals

                    res = wilcoxon_test(x_vals, y_vals)   # always less
                    sig  = "***" if res["significant"] else "n.s."
                    pval = f"{res['pvalue']:.2e}" if res["pvalue"] is not None else "N/A"
                    w    = f"{res['W']:.1f}"       if res["W"] is not None else "N/A"
                    note = f"  [{res['note']}]"    if res["note"] else ""
                    match = "✅" if res["W"] == 0 and res["significant"] else (
                            "—" if res["note"] == "all_ties" else "⚠")
                    print(f"    {met_name:<9} ddfp vs {baseline:<15} "
                          f"W={w:>5}  p={pval}  {sig}  {match}{note}")

            # seq_fp vs baseline (same comparison as ddfp)
            if by_prep["seq_fp"]["tsi"]:
                print()
                print(f"  seq_fp vs baselines (should match ddfp results above):")
                for met_name, key, x_fixed, y_fixed in metrics_to_test:
                    for baseline in ["no_interp", "naive_interp"]:
                        seq_vals  = by_prep["seq_fp"][key]
                        base_vals = by_prep[baseline][key]
                        if not base_vals or not seq_vals:
                            continue
                        if x_fixed is not None:
                            x_vals, y_vals = seq_vals, base_vals
                        else:
                            x_vals, y_vals = base_vals, seq_vals
                        res = wilcoxon_test(x_vals, y_vals)
                        sig  = "***" if res["significant"] else "n.s."
                        pval = f"{res['pvalue']:.2e}" if res["pvalue"] is not None else "N/A"
                        w    = f"{res['W']:.1f}"       if res["W"] is not None else "N/A"
                        note = f"  [{res['note']}]"    if res["note"] else ""
                        match = "✅" if res["W"] == 0 and res["significant"] else (
                                "—" if res["note"] == "all_ties" else "⚠")
                        print(f"    {met_name:<9} seq_fp vs {baseline:<15} "
                              f"W={w:>5}  p={pval}  {sig}  {match}{note}")

                # ddfp vs seq_fp equivalence check
                print()
                print(f"  ddfp ↔ seq_fp numerical equivalence (Theorem 4.1 empirical check):")
                for met_name, key in [("CC", "cc"), ("TSI", "tsi"),
                                       ("b0_cons", "b0_cons"), ("DWC", "dwc_rate")]:
                    ddfp_vals = by_prep["ddfp"][key]
                    seq_vals  = by_prep["seq_fp"][key]
                    if not ddfp_vals or not seq_vals:
                        continue
                    arr_d = np.array(ddfp_vals, dtype=float)
                    arr_s = np.array(seq_vals,  dtype=float)
                    max_diff = float(np.max(np.abs(arr_d - arr_s))) if len(arr_d) == len(arr_s) else float("nan")
                    identical = np.allclose(arr_d, arr_s, atol=1e-5)
                    flag = "✅" if identical else "⚠ DIFFER"
                    print(f"    {met_name:<9}  max|ddfp-seq_fp|={max_diff:.2e}  {flag}")

    print("\n" + "─" * 70)
    print("─" * 70)
    drive_rows = [r for r in rows if r["dataset"] == "drive"]
    if drive_rows:
        naive_tsi_vals = [r["tsi"] for r in drive_rows
                          if r["preprocessing"] == "naive_interp"]
        ddfp_tsi_vals  = [r["tsi"] for r in drive_rows
                          if r["preprocessing"] == "ddfp"]
        no_cc_vals     = [r["cc"]  for r in drive_rows
                          if r["preprocessing"] == "no_interp"]
        naive_b0       = [r["b0_consistency"] for r in drive_rows
                          if r["preprocessing"] == "naive_interp"]
        naive_flip     = [r["chi_sign_flip"] for r in drive_rows
                          if r["preprocessing"] == "naive_interp"]

        # Print summary table
        ref_vals: dict = {}
        def _ref_str(key, fmt=".2f"):
            return "N/A"

        has_ref = False
        hdr_ref = ""
        print(f"  {'metric':<35} {'this run':>12}")
        print(f"  {'-'*35} {'-'*12}")

        def _row(label, current_val, ref_key, fmt=".2f", pct=False):
            cur_str = f"{current_val:{fmt}}" + ("%" if pct else "")
            print(f"  {label:<35} {cur_str:>12}")

        _row("naive_interp TSI (DRIVE mean)",   np.mean(naive_tsi_vals),        "naive_tsi")
        _row("ddfp TSI (DRIVE mean)",           np.mean(ddfp_tsi_vals),         "ddfp_tsi", fmt=".4f")
        _row("no_interp CC (DRIVE mean)",       np.mean(no_cc_vals),            "no_cc")
        _row("naive_interp b0_cons (DRIVE mean)", np.mean(naive_b0),            "naive_b0", fmt=".6f")
        _row("naive_interp chi_flip (DRIVE, %)", 100*np.mean(naive_flip),       "naive_flip", fmt=".1f", pct=True)


# CLI

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Part B Exp 1: Topology accuracy (9-point TSI + full metrics)"
    )
    p.add_argument(
        "--drive-gt",
        default="data/DRIVE/training/1st_manual",
        help="DRIVE GT mask directory (.gif or .png)",
    )
    p.add_argument(
        "--cremi-gt",
        default="data/CREMI/masks",
        help="CREMI GT mask directory (.png)",
    )
    p.add_argument(
        "--cremi-limit",
        type=int,
        default=50,
    )
    p.add_argument(
        "--output",
        default="results/part_b/exp_b1_results.csv",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())