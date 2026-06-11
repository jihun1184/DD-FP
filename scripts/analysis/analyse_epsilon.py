"""
scripts/analysis/analyse_epsilon.py
==================================
Block C (Tier-1) — IBI convergence tolerance ε sensitivity analysis.

Input
-----
  JSON output of run_epsilon_sensitivity.py  (epsilon_sensitivity*.json)
  OR pre-aggregated CSV: subject, K, delta, eps, violations, rounds [, t_s]

Usage
-----
    # JSON produced by run_epsilon_sensitivity.py
    python analyse_epsilon.py --input epsilon_sensitivity_k16_delta1.csv

    # Plain CSV (subject,K,delta,eps,violations,rounds)
    python analyse_epsilon.py --input epsilon_sensitivity.csv

    # Explicit K / delta filter (default: analyse all present)
    python analyse_epsilon.py --input epsilon_sensitivity_k16_delta1.csv --K 16 --delta 1

Outputs
-------
  - Console table (ε × metric)
  - epsilon_summary.csv      (mean ± std per ε, for LaTeX)
  - epsilon_sensitivity.pdf  (two-panel figure: violations + rounds)
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not found — Kruskal-Wallis skipped. pip install scipy")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_input(path: Path) -> pd.DataFrame:
    """
    Auto-detect format: JSON (run_epsilon_sensitivity output) or CSV.
    Both produce a DataFrame with columns:
        subject, K, delta, eps, violations, rounds
    """
    text = path.read_text()
    try:
        data = json.loads(text)
        # JSON from run_epsilon_sensitivity.py
        records = data["records"]
        df = pd.DataFrame(records)
        # keep only the columns analyse needs
        keep = ["subject", "K", "delta", "eps", "violations", "rounds"]
        df = df[[c for c in keep if c in df.columns]]
        print(f"Loaded JSON → {len(df)} records  "
              f"(meta: K={data['meta'].get('K')}, "
              f"delta={data['meta'].get('delta')}, "
              f"seed={data['meta'].get('seed','?')}, "
              f"n_subjects={data['meta'].get('n_subjects','?')})")
        return df
    except (json.JSONDecodeError, KeyError):
        pass

    # Plain CSV fallback
    df = pd.read_csv(path)
    required = {"subject", "K", "delta", "eps", "violations", "rounds"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: input CSV missing columns: {missing}")
    print(f"Loaded CSV → {len(df)} records")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def summarise(df: pd.DataFrame, K: int, delta: int) -> pd.DataFrame:
    """Per-ε summary statistics for the given (K, delta) configuration."""
    sub = df[(df["K"] == K) & (df["delta"] == delta)].copy()
    if sub.empty:
        sys.exit(f"ERROR: No data for K={K}, δ={delta}.")

    records = []
    for eps in sorted(sub["eps"].unique()):
        s = sub[sub["eps"] == eps]
        n = len(s)
        viol = s["violations"]
        rnd  = s["rounds"]
        records.append({
            "eps":         eps,
            "n_subjects":  n,
            "viol_mean":   viol.mean(),
            "viol_std":    viol.std(ddof=1) if n > 1 else 0.0,
            "viol_min":    viol.min(),
            "viol_max":    viol.max(),
            "pass_rate":   (viol == 0).mean(),
            "rounds_mean": rnd.mean(),
            "rounds_std":  rnd.std(ddof=1) if n > 1 else 0.0,
            "rounds_min":  rnd.min(),
            "rounds_max":  rnd.max(),
        })
    return pd.DataFrame(records)


def kruskal_wallis(df: pd.DataFrame, metric: str,
                   K: int, delta: int) -> dict:
    """Kruskal-Wallis H-test across ε levels."""
    if not HAS_SCIPY:
        return {"H": float("nan"), "p": float("nan")}

    sub = df[(df["K"] == K) & (df["delta"] == delta)]
    groups = [sub[sub["eps"] == e][metric].values
              for e in sorted(sub["eps"].unique())]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return {"H": float("nan"), "p": float("nan")}

    all_vals = np.concatenate(groups)
    if np.all(all_vals == all_vals[0]):
        return {"H": 0.0, "p": 1.0,
                "note": "all values identical — ε has no effect"}
    try:
        H, p = sp_stats.kruskal(*groups)
        return {"H": H, "p": p}
    except ValueError as e:
        return {"H": 0.0, "p": 1.0, "note": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def plot_sensitivity(summary: pd.DataFrame, out_path: Path,
                     K: int, delta: int) -> None:
    eps_vals = summary["eps"].values
    x        = np.arange(len(eps_vals))
    labels   = [f"$\\varepsilon={e}$" for e in eps_vals]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    fig.suptitle(f"$K={K},\\ \\delta={delta}$", fontsize=12)

    # panel A: violations
    ax = axes[0]
    ax.bar(x, summary["viol_mean"], yerr=summary["viol_std"],
           capsize=4, color="#4C72B0", edgecolor="black",
           linewidth=0.6, error_kw={"linewidth": 0.8})
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Boundary DWC violations (mean ± std)", fontsize=10)
    ax.set_xlabel("Convergence tolerance $\\varepsilon$", fontsize=10)
    ax.set_title("(a) DWC violations", fontsize=11)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)
    for i, row in summary.iterrows():
        ax.text(i, row["viol_mean"] + row["viol_std"] + 0.05,
                f"{row['pass_rate']*100:.0f}% pass",
                ha="center", va="bottom", fontsize=8, color="#333333")

    # panel B: rounds
    ax = axes[1]
    ax.bar(x, summary["rounds_mean"], yerr=summary["rounds_std"],
           capsize=4, color="#DD8452", edgecolor="black",
           linewidth=0.6, error_kw={"linewidth": 0.8})
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("IBI rounds used $R^*$ (mean ± std)", fontsize=10)
    ax.set_xlabel("Convergence tolerance $\\varepsilon$", fontsize=10)
    ax.set_title("(b) IBI convergence rounds", fontsize=11)
    ax.set_ylim(0, summary["rounds_max"].max() + 1.5)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Figure → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ε sensitivity analysis for DD-FP IBI (sec:enew4)."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="JSON (run_epsilon_sensitivity output) or CSV "
             "(subject,K,delta,eps,violations,rounds)"
    )
    parser.add_argument("--K",     type=int, default=None,
                        help="K to analyse (default: use value from data)")
    parser.add_argument("--delta", type=int, default=None,
                        help="δ to analyse (default: use value from data)")
    parser.add_argument("--out_dir", type=Path, default=Path("."),
                        help="Output directory (default: .)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── load ─────────────────────────────────────────────────────────────────
    df = load_input(args.input)
    df["eps"]        = df["eps"].astype(float)
    df["violations"] = df["violations"].astype(int)
    df["rounds"]     = df["rounds"].astype(int)

    # ── resolve K / delta ────────────────────────────────────────────────────
    K     = args.K     if args.K     is not None else int(df["K"].mode()[0])
    delta = args.delta if args.delta is not None else int(df["delta"].mode()[0])
    print(f"Analysing: K={K}, δ={delta}")

    # ── save normalised CSV ───────────────────────────────────────────────────
    csv_out = args.out_dir / f"epsilon_sensitivity_K{K}_d{delta}.csv"
    df[df["K"] == K][df["delta"] == delta].to_csv(csv_out, index=False)
    print(f"Normalised CSV → {csv_out}")

    # ── summarise ────────────────────────────────────────────────────────────
    summary = summarise(df, K=K, delta=delta)
    print(f"\n── Per-ε summary (K={K}, δ={delta}) ────────────────────")
    print(summary.to_string(index=False, float_format="{:.3f}".format))

    summary_path = args.out_dir / f"epsilon_summary_K{K}_d{delta}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary CSV → {summary_path}")

    # ── Kruskal-Wallis ───────────────────────────────────────────────────────
    for metric in ["violations", "rounds"]:
        res = kruskal_wallis(df, metric, K=K, delta=delta)
        note = res.get("note", "")
        print(f"Kruskal-Wallis ({metric}): H={res['H']:.3f}, p={res['p']:.4f}"
              + (f"  [{note}]" if note else ""))

    # ── figure ───────────────────────────────────────────────────────────────
    fig_path = args.out_dir / f"epsilon_sensitivity_K{K}_d{delta}.pdf"
    plot_sensitivity(summary, fig_path, K=K, delta=delta)

    print("\nDone.")


if __name__ == "__main__":
    main()