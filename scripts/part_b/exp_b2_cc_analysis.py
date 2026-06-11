"""
scripts/part_b/exp_b2_cc_analysis.py
-------------------------------------
Part B Experiment 2: Connectivity Consistency Analysis on 2D Drive/Cremi Dataset.

Provides controlled validation of topology metric behaviour across:
  (a) Synthetic images — KNOWN topology, no GT ambiguity
  (b) DRIVE GT masks   — reference = original mask beta0/chi
  (c) CREMI GT masks   — reference = original mask beta0/chi

Three-way distinction (critical for paper):
  no_interp   : binary, DWC violations  → CC > 0, TSI = 0
  naive_interp: continuous, no violations → CC = 0 (trivial), TSI > 0
  ddfp        : binary, DWC guaranteed  → CC = 0 AND TSI = 0

The distinction between naive and ddfp is the key contribution:
  naive satisfies CC=0 because bilinear output is non-binary (no
  4- vs 8-connectivity ambiguity at pixel boundaries).
  ddfp  satisfies CC=0 by construction (DWC guarantee, Boutry 2015),
  and additionally produces truly binary output → TSI=0.

Metrics per row:
  CC              = |beta0_4conn - beta0_8conn|         [Rosenfeld & Kak 1982]
  TSI             = std( beta0_8conn(thr) ) over thr    [threshold stability]
  b0_consistency  = beta0_8conn / beta0_4conn           [topology coherence]
                      DWC ⟹ b0_4=b0_8 ⟹ ratio=1.0
                      DWC violation ⟹ 8-conn diagonal merge ⟹ ratio << 1.0
  chi_sign_flip   = 1 if sign(chi_interp) ≠ sign(chi_original)
                      naive bilinear creates spurious loops ⟹ chi → negative
  is_binary       = output ∈ {0,1}

No training required. Runtime: ~minutes.

Usage (from project root):
    # Synthetic + DRIVE
    python scripts/part_b/exp_b2_cc_analysis.py \\
        --drive-gt data/DRIVE/training/1st_manual \\
        --output results/part_b/exp_b2_results.csv

    # Synthetic + DRIVE + CREMI
    python scripts/part_b/exp_b2_cc_analysis.py \\
        --drive-gt data/DRIVE/training/1st_manual \\
        --cremi-gt data/CREMI/masks \\
        --cremi-limit 50 \\
        --output results/part_b/exp_b2_results.csv
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
)
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------

def _b0(binary: np.ndarray, conn: int) -> int:
    """Connected component count: conn=4 or conn=8."""
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


def _tsi(
    interp: np.ndarray,
    thresholds: tuple = (0.3, 0.4, 0.5, 0.6, 0.7),
) -> float:
    """
    Threshold Stability Index = std( beta0_8conn(thr) ).

    naive_interp: continuous values → TSI > 0 (topology changes with thr)
    ddfp/no_interp: binary → TSI = 0 always
    """
    return float(np.std([_b0((interp >= t).astype(np.uint8), 8)
                          for t in thresholds]))


def full_metrics(
    interp: np.ndarray,
    ref_b0: int,
    ref_chi: int,
    threshold: float = 0.5,
) -> dict:
    """
    Full metric suite at one binarisation threshold.

    Parameters
    ----------
    interp    : float32 interpolated image (may be continuous for naive)
    ref_b0    : reference beta0_8conn from original binary mask (unused for
                b0_consistency, kept for API compatibility)
    ref_chi   : reference Euler characteristic from original binary mask
    threshold : binarisation threshold (default 0.5)

    New metrics replacing the previous (broken) beta0_error / chi_error
    -----------------------------------------------------------------------
    b0_consistency  = beta0_8conn / beta0_4conn
        DWC ⟹ beta0_4conn = beta0_8conn ⟹ ratio = 1.0  (perfect)
        DWC violation ⟹ 8-conn merges diagonal components
                       ⟹ beta0_8 << beta0_4 ⟹ ratio << 1.0
        DRIVE no_interp: ~0.008   ddfp: 1.000   (173× difference)

        Previous beta0_error = |beta0_8(interp) - beta0(original)| was
        meaningless because interp and original have different resolutions
        (H×W vs (2H-1)×(2W-1)). b0_consistency is resolution-invariant.

    chi_sign_flip   = 1 if sign(chi_interp) ≠ sign(chi_original), else 0
        chi = beta0 - beta1 (Euler characteristic).
        Healthy vascular/membrane masks have chi > 0 (few holes).
        naive bilinear interpolation creates spurious face-cells at diagonal
        crossings ⟹ chi flips negative ⟹ chi_sign_flip = 1.
        DRIVE: naive 19/20 flip, no_interp 0/20, ddfp 0/20.

        Previous chi_error = |chi_interp - chi_original| suffered from the
        same resolution mismatch. chi_sign_flip is scale-free and directly
        interpretable as a topological anomaly indicator.
    """
    bin_img   = (interp >= threshold).astype(np.uint8)
    b0_4      = _b0(bin_img, 4)
    b0_8      = _b0(bin_img, 8)
    chi       = _chi(bin_img)
    cc        = abs(b0_4 - b0_8)

    # b0_consistency: 1.0 = perfect (DWC), <1.0 = diagonal merging present
    b0_cons   = round(b0_8 / b0_4, 6) if b0_4 > 0 else 1.0

    # chi_sign_flip: 1 if interpolation inverted the Euler characteristic sign
    chi_flip  = int((ref_chi != 0) and (chi * ref_chi < 0))

    tsi       = _tsi(interp)
    is_bin    = bool(np.all((interp == 0) | (interp == 1)))

    return {
        "threshold":      threshold,
        "beta0_4conn":    b0_4,
        "beta0_8conn":    b0_8,
        "chi":            chi,
        "cc":             cc,
        "cc_zero":        int(cc == 0),
        "b0_consistency": b0_cons,
        "chi_sign_flip":  chi_flip,
        "tsi":            round(tsi, 4),
        "is_binary":      int(is_bin),
    }


# ---------------------------------------------------------------------------
# Preprocessor factory
# ---------------------------------------------------------------------------

def make_preprocessors() -> dict:
    cfg = SimpleNamespace(preprocessing=SimpleNamespace(
        type="ddfp",
        naive_mode="bilinear",
        ddfp_overlap=1,
        ddfp_cache_dir=None,
        no_interp_cache_dir=None,
        naive_interp_cache_dir=None,
    ))
    return {
        "no_interp":    NoInterpPreprocessor(cfg),
        "naive_interp": NaiveInterpPreprocessor(cfg),
        "ddfp":         DDFPPreprocessor(cfg),
    }


# ---------------------------------------------------------------------------
# Synthetic benchmark images
# ---------------------------------------------------------------------------

def make_synthetic_images() -> dict:
    """
    Build binary images with analytically known topology.

    Returns dict of name → (image, ref_b0_8conn, ref_chi)

    Chosen to stress-test specific topology failure modes:
      disk/ring/two_disks → canonical shapes (sanity check)
      thin_line           → 1-px vessels (DWC violation risk)
      cross_gap           → anti-diagonal crossing (CC stress test)
      diagonal_stripe     → pure 8-conn structure (known DD-FP limitation)
    """
    def _ref(arr):
        b = arr.astype(np.uint8)
        return _b0(b, 8), _chi(b)

    images = {}

    # 1. Single disk (beta0=1, beta1=0)
    h, w = 64, 64
    y, x = np.mgrid[-h//2:h//2, -w//2:w//2]
    disk = ((x**2 + y**2) < 20**2).astype(np.uint8)
    images["disk_1"] = (disk, *_ref(disk))

    # 2. Ring / annulus (beta0=1, beta1=1)
    ring = (((x**2 + y**2) < 25**2) & ((x**2 + y**2) > 12**2)).astype(np.uint8)
    images["ring_1"] = (ring, *_ref(ring))

    # 3. Two separate disks (beta0=2, beta1=0)
    d = np.zeros((64, 96), dtype=np.uint8)
    yy, xx = np.mgrid[0:64, 0:96]
    d |= (((xx-22)**2 + (yy-32)**2) < 15**2).astype(np.uint8)
    d |= (((xx-70)**2 + (yy-32)**2) < 15**2).astype(np.uint8)
    images["two_disks"] = (d, *_ref(d))

    # 4. Thin horizontal vessel (1-px wide — DWC violation risk)
    v = np.zeros((32, 64), dtype=np.uint8)
    v[16, 5:59] = 1
    images["thin_line"] = (v, *_ref(v))

    # 5. Cross with gap at junction
    #    Centre pixel removed → anti-diagonal crossing
    c = np.zeros((32, 32), dtype=np.uint8)
    c[16, 4:28] = 1
    c[4:28, 16] = 1
    c[16, 16] = 0
    images["cross_gap"] = (c, *_ref(c))

    # 6. Diagonal stripe — pure 8-connected structure
    #    Exposes DD-FP limitation: Khalimsky grid does not preserve
    #    pure-diagonal (8-conn only) connectivity.  Used to motivate
    #    the Discussion paragraph on structural assumptions.
    diag = np.zeros((24, 32), dtype=np.uint8)
    for i in range(min(24, 32)):
        diag[i, i % 32] = 1
    images["diagonal_stripe"] = (diag, *_ref(diag))

    return images


# ---------------------------------------------------------------------------
# Real-data processing (dataset-agnostic)
# ---------------------------------------------------------------------------

def process_real_masks(
    mask_paths: list,
    source: str,
    preps: dict,
    thresholds: list,
    rows: list,
) -> None:
    """
    Process real GT masks and append rows.

    Parameters
    ----------
    mask_paths : list of Path objects
    source     : "drive" or "cremi" — recorded in CSV
    preps      : preprocessor dict
    thresholds : list of binarisation thresholds to test
    rows       : output list (mutated in place)
    """
    for path in mask_paths:
        try:
            from PIL import Image
            mask = (np.array(Image.open(path).convert("L"),
                             dtype=np.float32) / 255.0 > 0.5).astype(np.uint8)
        except Exception as e:
            print(f"  Skip {path.name}: {e}")
            continue

        # Skip degenerate masks
        fg_rate = mask.mean()
        if fg_rate < 0.001 or fg_rate > 0.999:
            print(f"  [{source}] {path.name}  SKIP (fg_rate={fg_rate:.3f})")
            continue

        ref_b0  = _b0(mask, 8)
        ref_chi = _chi(mask)
        print(f"  [{source}] {path.name}  "
              f"ref_b0={ref_b0}  ref_chi={ref_chi}", flush=True)

        for prep_name, prep in preps.items():
            interp, _ = prep(mask.astype(np.float32), sample_id=path.stem)
            for thr in thresholds:
                m = full_metrics(interp, ref_b0=ref_b0,
                                 ref_chi=ref_chi, threshold=thr)
                rows.append({
                    "source":        source,
                    "sample":        path.name,
                    "preprocessing": prep_name,
                    "ref_beta0":     ref_b0,
                    "ref_chi":       ref_chi,
                    **m,
                })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    preps      = make_preprocessors()
    thresholds = [0.3, 0.4, 0.5]
    rows: list = []

    # ── Part 1: Synthetic images ──────────────────────────────────────────
    print("\n=== Synthetic images (known topology) ===")
    synthetics = make_synthetic_images()
    for name, (mask, ref_b0, ref_chi) in synthetics.items():
        h, w = mask.shape
        print(f"  {name}: shape={h}x{w}  ref_b0={ref_b0}  ref_chi={ref_chi}")

        for prep_name, prep in preps.items():
            interp, _ = prep(mask.astype(np.float32), sample_id=name)
            for thr in thresholds:
                m = full_metrics(interp, ref_b0=ref_b0,
                                 ref_chi=ref_chi, threshold=thr)
                rows.append({
                    "source":        "synthetic",
                    "sample":        name,
                    "preprocessing": prep_name,
                    "ref_beta0":     ref_b0,
                    "ref_chi":       ref_chi,
                    **m,
                })

    # ── Part 2: DRIVE GT masks ────────────────────────────────────────────
    drive_dir = Path(args.drive_gt)
    if drive_dir.exists():
        masks = sorted(drive_dir.glob("*.gif")) or sorted(drive_dir.glob("*.png"))
        if masks:
            print(f"\n=== DRIVE GT masks ({len(masks)} images) ===")
            process_real_masks(masks, "drive", preps, thresholds, rows)
        else:
            print(f"  [WARNING] No masks in {drive_dir}")
    else:
        print(f"  [INFO] DRIVE directory not found ({drive_dir}) — skipping.")

    # ── Part 3: CREMI GT masks ────────────────────────────────────────────
    cremi_dir = Path(args.cremi_gt)
    if cremi_dir.exists():
        masks = sorted(cremi_dir.glob("*.png"))[:args.cremi_limit]
        if masks:
            print(f"\n=== CREMI GT masks ({len(masks)} patches, "
                  f"limit={args.cremi_limit}) ===")
            process_real_masks(masks, "cremi", preps, thresholds, rows)
        else:
            print(f"  [WARNING] No .png masks in {cremi_dir}")
    else:
        print(f"  [INFO] CREMI directory not found ({cremi_dir}) — skipping. "
              f"Run scripts/prepare_cremi.py to prepare data.")

    if not rows:
        print("No data processed.")
        return

    # ── Save CSV ──────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved → {out}  ({len(rows)} rows)")

    _print_summary(rows)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(rows: list) -> None:
    import collections

    sources = ["synthetic", "drive", "cremi"]

    print("\n" + "=" * 65)
    print("SUMMARY  —  Part B Exp 2: CC Deep Analysis")
    print("=" * 65)

    for source in sources:
        src_rows = [r for r in rows if r["source"] == source]
        if not src_rows:
            continue

        thr05 = [r for r in src_rows if float(r["threshold"]) == 0.5]
        n_samples = len(set(r["sample"] for r in src_rows))
        print(f"\n──── Source: {source.upper()}  ({n_samples} samples, "
              f"thr=0.5) ────")

        # CC
        print("\n  [CC]")
        by_p = collections.defaultdict(list)
        for r in thr05:
            by_p[r["preprocessing"]].append(r["cc"])
        for p in ["no_interp", "naive_interp", "ddfp"]:
            v = by_p[p]
            if v:
                z = 100.0 * sum(x == 0 for x in v) / len(v)
                print(f"    {p:<15}  mean CC={np.mean(v):6.2f}  "
                      f"CC=0: {z:5.1f}%")

        # TSI
        print("\n  [TSI]")
        by_p2 = collections.defaultdict(list)
        for r in thr05:
            by_p2[r["preprocessing"]].append(r["tsi"])
        for p in ["no_interp", "naive_interp", "ddfp"]:
            v = by_p2[p]
            if v:
                print(f"    {p:<15}  mean={np.mean(v):.4f}  "
                      f"max={max(v):.4f}")

        # b0_consistency
        print("\n  [b0_consistency = β₀^{8-conn} / β₀^{4-conn}]")
        print("    DWC ⟹ b0_4=b0_8 ⟹ ratio=1.0  |  "
              "DWC violation ⟹ diagonal merge ⟹ ratio<<1.0")
        by_p3 = collections.defaultdict(list)
        for r in thr05:
            by_p3[r["preprocessing"]].append(float(r["b0_consistency"]))
        for p in ["no_interp", "naive_interp", "ddfp"]:
            v = by_p3[p]
            if v:
                print(f"    {p:<15}  mean={np.mean(v):.4f}  "
                      f"min={min(v):.4f}  median={float(np.median(v)):.4f}")

        # chi_sign_flip
        print("\n  [chi_sign_flip: sign(χ_interp) ≠ sign(χ_original)]")
        print("    naive bilinear creates spurious loops ⟹ χ → negative")
        by_p4 = collections.defaultdict(list)
        for r in thr05:
            by_p4[r["preprocessing"]].append(int(r["chi_sign_flip"]))
        for p in ["no_interp", "naive_interp", "ddfp"]:
            v = by_p4[p]
            if v:
                pct = 100.0 * sum(v) / len(v)
                print(f"    {p:<15}  flip rate={pct:.0f}%  "
                      f"({sum(v)}/{len(v)})")

    # Threshold sensitivity table
    synth_rows = [r for r in rows if r["source"] == "synthetic"]
    if synth_rows:
        print("\n──── Threshold sensitivity (synthetic) ────")
        print("  CC=0 rate across thresholds — "
              "naive drops at extreme thresholds\n")
        for thr in [0.3, 0.4, 0.5]:
            t_rows = [r for r in synth_rows if float(r["threshold"]) == thr]
            by_p = collections.defaultdict(list)
            for r in t_rows:
                by_p[r["preprocessing"]].append(r["cc_zero"])
            parts = []
            for p in ["no_interp", "naive_interp", "ddfp"]:
                v = by_p[p]
                if v:
                    parts.append(f"{p}: {100*sum(v)/len(v):.0f}%")
            print(f"  thr={thr}  |  " + "  ".join(parts))

    print()
    print("Expected (paper Table 4):")
    print("  ddfp:         CC=0 100%,  TSI=0.00,  b0_cons=1.00,  chi_flip=0%")
    print("  naive_interp: CC=0 100%,  TSI>0   ,  b0_cons=1.00,  chi_flip>0%")
    print("  no_interp:    CC>0 (thin/diagonal),  b0_cons<<1.0,  TSI=0.00")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Part B Exp 2: CC deep analysis + TSI (synthetic + DRIVE + CREMI)"
    )
    p.add_argument(
        "--drive-gt",
        default="data/DRIVE/training/1st_manual",
        help="Path to DRIVE GT mask directory (.gif or .png)",
    )
    p.add_argument(
        "--cremi-gt",
        default="data/CREMI/masks",
        help="Path to CREMI GT mask directory (.png 128×128 patches)",
    )
    p.add_argument(
        "--cremi-limit",
        type=int,
        default=50,
        help="Max number of CREMI patches to process (default: 50)",
    )
    p.add_argument(
        "--output",
        default="results/part_b/exp_b2_results.csv",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
