"""
scripts/part_b/verify_wilcoxon.py
-----------------------------------
DD-FP Part B — Wilcoxon signed-rank verification script.

Reproduces all topology-consistency Wilcoxon claims from the paper
using a results CSV only; no preprocessing pipeline re-run required.

Test design:
  - One-sided Wilcoxon signed-rank, zero_method='wilcox'
  - Direction: ddfp is always the "smaller" side
    (lower-is-better metrics: ddfp < baseline;
     higher-is-better metrics: baseline < ddfp)
  - W=0 means all differences point in the same direction (complete separation)

Usage:
    python scripts/part_b/verify_wilcoxon.py \
        --csv results/part_b/exp_b1_v2_results.csv
    python scripts/part_b/verify_wilcoxon.py \
        --csv exp_b1_results.csv --dataset drive --alpha 0.001
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


#   wilcoxon(x, y, alternative='less') → H1: median(x-y) < 0 → x < y

TESTS: list[dict] = [
    # ── CC ──────────────────────────────────────────────────────────────────
    {
        "label":     "CC  ddfp < no_interp",
        "metric":    "CC",
        "col":       "cc",
        "x_prep":    "ddfp",
        "y_prep":    "no_interp",      # y = baseline
        "alt":       "less",           # H1: ddfp < no_interp
        "note":      "anti-diagonal CC violations in no_interp",
    },
    # ── TSI ─────────────────────────────────────────────────────────────────
    {
        "label":     "TSI ddfp < naive_interp",
        "metric":    "TSI",
        "col":       "tsi",
        "x_prep":    "ddfp",
        "y_prep":    "naive_interp",
        "alt":       "less",
        "note":      "threshold-sensitive β₀ in naive_interp",
    },
    # ── b0_consistency ──────────────────────────────────────────────────────
    {
        "label":     "b0_cons no_interp < ddfp",
        "metric":    "b0_consistency",
        "col":       "b0_consistency",
        "x_prep":    "no_interp",
        "y_prep":    "ddfp",
        "alt":       "less",           # H1: no_interp < ddfp
        "note":      "β₀⁸/β₀⁴ ratio; DWC → ratio=1, violation → ratio<<1",
    },
    # ── DWC violation rate ───────────────────────────────────────────────────
    {
        "label":     "DWC  ddfp < no_interp",
        "metric":    "DWC violation rate",
        "col":       "dwc_viol_rate",
        "x_prep":    "ddfp",
        "y_prep":    "no_interp",
        "alt":       "less",
        "note":      "pixel-level DWC violation rate; ddfp=0 by construction",
    },
    # ── chi_sign_flip ────────────────────────────────────────────────────────
    # ddfp: 0/20 (0%), naive_interp: 19/20 (95%)
    # ddfp < naive_interp
    {
        "label":     "χ_flip ddfp < naive_interp",
        "metric":    "chi_sign_flip",
        "col":       "chi_sign_flip",
        "x_prep":    "ddfp",
        "y_prep":    "naive_interp",
        "alt":       "less",
        "note":      "sign(χ_interp)≠sign(χ_orig); naive: 95% flip, ddfp: 0%",
    },
]


def run_wilcoxon(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "less",
    zero_method: str = "wilcox",
) -> dict:
    """
    One-sided Wilcoxon signed-rank test.

    Parameters
    ----------
    x, y        : paired observations (same sample, different preprocessing)
    alternative : 'less' | 'greater' | 'two-sided'
    zero_method : 'wilcox' (scipy default; ties at 0 excluded)

    Returns
    -------
    dict with keys: W, pvalue, n, n_effective, significant, note
    """
    from scipy.stats import wilcoxon as sp_wilcoxon

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    if np.allclose(x, y):
        return {"W": None, "pvalue": None, "n": n,
                "n_effective": 0, "significant": False,
                "note": "all_ties (x≡y)"}

    diffs = x - y
    n_eff = int(np.sum(diffs != 0))

    try:
        result = sp_wilcoxon(x, y,
                             zero_method=zero_method,
                             alternative=alternative)
        return {
            "W":           float(result.statistic),
            "pvalue":      float(result.pvalue),
            "n":           n,
            "n_effective": n_eff,
            "significant": bool(result.pvalue < 0.001),
            "note":        "",
        }
    except Exception as e:
        return {"W": None, "pvalue": None, "n": n,
                "n_effective": n_eff, "significant": False,
                "note": str(e)}


def descriptive_stats(df: pd.DataFrame, col: str, dataset: str) -> pd.DataFrame:
    """Descriptive statistics for a given metric, grouped by preprocessing."""
    sub = df[df["dataset"] == dataset] if "dataset" in df.columns else df
    print("sub:", sub.head())
    rows = []
    for prep in ["no_interp", "naive_interp", "ddfp"]:
        vals = sub[sub["preprocessing"] == prep][col].values
        if len(vals) == 0:
            continue
        rows.append({
            "preprocessing": prep,
            "n":    len(vals),
            "mean": np.mean(vals),
            "std":  np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
            "min":  np.min(vals),
            "max":  np.max(vals),
            "=0 %": f"{100 * np.mean(vals == 0):.1f}%",
        })
    return pd.DataFrame(rows).set_index("preprocessing")


def run_verification(
    csv_path: Path,
    dataset: str = "drive",
    alpha: float = 0.001,
) -> bool:
    """
    Load CSV → descriptive statistics → Wilcoxon tests.

    Returns
    -------
    df with W, p-value, etc.
    """
    df = pd.read_csv(csv_path)
    if "dataset" in df.columns:
        sub = df[df["dataset"] == dataset].copy()
    else:
        sub = df.copy()
        sub["dataset"] = dataset

    n_samples = sub["sample"].nunique() if "sample" in sub.columns else len(sub) // 3
    preps_avail = set(sub["preprocessing"].unique())

    print("=" * 68)
    print("DD-FP Part B — Wilcoxon Signed-Rank Verification Report")
    print("=" * 68)
    print(f"  CSV        : {csv_path}")
    print(f"  Dataset    : {dataset.upper()}  (N={n_samples} samples)")
    print(f"  α threshold: {alpha}")
    print(f"  zero_method: wilcox  (ties excluded from ranking)")
    print(f"  scipy.stats.wilcoxon — one-sided paired test")
    print()

    print("── Descriptive Statistics ──────────────────────────────────────")
    for col, label in [
        ("cc",             "CC  (Cross-Connectivity, ↓ better)"),
        ("tsi",            "TSI (Threshold Sensitivity Index,  ↓ better)"),
        ("b0_consistency", "b0_cons = β₀⁸/β₀⁴              (↑ better)"),
        ("dwc_viol_rate",  "DWC violation rate               (↓ better)"),
        ("chi_sign_flip",  "χ sign-flip rate                 (↓ better)"),
    ]:
        if col not in sub.columns:
            continue
        print(f"\n  {label}")
        stats = descriptive_stats(sub, col, dataset)
        print(stats.to_string())
    print()

    print("── Wilcoxon Signed-Rank Tests ──────────────────────────────────")
    print(f"  H₁: x < y  (one-sided less, paired by sample)")
    print()

    all_pass = True
    results_rows = []

    for t in TESTS:
        col = t["col"]
        if col not in sub.columns:
            print(f"  [SKIP] {t['label']} — column '{col}' not in CSV")
            continue
        if t["x_prep"] not in preps_avail or t["y_prep"] not in preps_avail:
            print(f"  [SKIP] {t['label']} — preprocessing not found")
            continue

        x_df = (sub[sub["preprocessing"] == t["x_prep"]]
                .sort_values("sample" if "sample" in sub.columns else sub.index.name)
                [[col]].rename(columns={col: "x"}))
        y_df = (sub[sub["preprocessing"] == t["y_prep"]]
                .sort_values("sample" if "sample" in sub.columns else sub.index.name)
                [[col]].rename(columns={col: "y"}))

        x_vals = x_df["x"].values
        y_vals = y_df["y"].values

        res = run_wilcoxon(x_vals, y_vals, alternative=t["alt"])

        _pass = (
            res["significant"] and
            res["pvalue"] is not None and
            res["pvalue"] < alpha
        )
        verdict = "✅ PASS" if _pass else "⚠  FAIL"
        W_str = f"{res['W']:.1f}" if res["W"] is not None else "N/A"
        p_str = f"{res['pvalue']:.2e}" if res["pvalue"] is not None else "N/A"
        neff  = res["n_effective"]

        print(f"  {t['label']}")
        print(f"    Metric : {t['metric']}")
        print(f"    N pairs: {res['n']}  (N_eff={neff}, ties excluded)")
        print(f"    Result : W={W_str},  p={p_str}  [{verdict}]")
        if res["note"]:
            print(f"    Note   : {res['note']}")
        print()

        results_rows.append({
            "label":       t["label"],
            "metric":      t["metric"],
            "x_prep":      t["x_prep"],
            "y_prep":      t["y_prep"],
            "alternative": t["alt"],
            "N":           res["n"],
            "N_eff":       neff,
            "W":           res["W"],
            "pvalue":      res["pvalue"],
            "significant": res["significant"],
            "note":        res["note"],
        })

    return pd.DataFrame(results_rows)


# CLI

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DD-FP Wilcoxon signed-rank verification (topology consistency)"
    )
    p.add_argument(
        "--csv",
        default="results/part_b/exp_b1_results.csv",
        help="path to exp_b1 results CSV (default: results/part_b/exp_b1_results.csv)",
    )
    p.add_argument(
        "--dataset",
        default="drive",
        help="dataset to verify (default: drive)",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.001,
        help="significance level alpha (default: 0.001)",
    )
    p.add_argument(
        "--csv-out",
        default=None,
        help="path to output CSV (default: None)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)

    df = run_verification(csv_path, dataset=args.dataset, alpha=args.alpha)
    if args.csv_out:
        df.to_csv(args.csv_out, index=False)
        print(f"[INFO] Results saved to: {args.csv_out}")   