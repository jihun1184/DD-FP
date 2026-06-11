"""
scripts/extras/run_epsilon_sensitivity.py
==========================================
E-NEW-4: IBI convergence tolerance ε sensitivity experiment.

Sweeps K ∈ {2,4,8,16}, δ ∈ {1,2}, ε ∈ {0.1,0.5,1.0,2.0} on N=20
randomly sampled BraTS 2021 FLAIR subjects (seed=42).
Reproduces sec:enew4 (Kruskal-Wallis H=0, p=1.0).

Full experiment (paper default):
  K_list    : 2 4 8 16      (4 values)
  delta_list: 1 2            (2 values)
  eps_list  : 0.1 0.5 1.0 2.0 (4 values)
  n_subjects: 20
  → 4 × 2 × 4 × 20 = 640 runs, saved to a single JSON

Single-config run (e.g. K=16, δ=1 only):
  --K_list 16 --delta_list 1

Usage:
  # Full sweep (all K, both delta)
  python scripts/analysis/run_epsilon_sensitivity.py \\
      --brats /path/to/BraTS2021_Training_Data

  # Partial (K=16 only, both delta)
  python scripts/analysis/run_epsilon_sensitivity.py \\
      --brats /path/to/BraTS2021_Training_Data \\
      --K_list 16 --delta_list 1 2

  # Resume / append: completed results are loaded and skipped
  python scripts/analysis/run_epsilon_sensitivity.py \\
      --brats /path/to/BraTS2021_Training_Data \\
      --out epsilon_sensitivity_full.json   # existing file → auto-skip

Output JSON (analyse_epsilon.py compatible):
  {
    "meta": {"K_list":[2,4,8,16], "delta_list":[1,2], ...},
    "records": [
      {"subject":"BraTS2021_00077","K":16,"delta":1,"eps":0.1,
       "violations":0,"rounds":2,"t_s":15.93},
      ...
    ]
  }
"""
from __future__ import annotations
import glob, json, os, random, sys, time, warnings
from itertools import product
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts" / "ddfp"))

try:
    import nibabel as nib
except ImportError:
    sys.exit("nibabel required: pip install nibabel")


# ── path helper ───────────────────────────────────────────────────────────────

def _add_exp_dir(exp_dir: str) -> None:
    p = str(Path(exp_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


# ── ε-aware IBI (identical to run_ibi_v10 except convergence criterion) ──────

def run_ibi_eps(
    vol_u8: np.ndarray,
    K: int,
    delta: int,
    eps: float,
    max_rounds: int = 8,
) -> dict:
    """
    DD-FP IBI with configurable convergence threshold `eps`.
    Returns R* (rounds used) and boundary DWC violation count.
    """
    from experiment_DDFP_all import (
        split_domain_1d,
        _pin_and_seed_expanded,
        _assemble_boundary,
        count_boundary_violations,
    )
    import cupy as cp
    from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu

    W, H, D = vol_u8.shape
    W2, H2, D2 = 2*W - 1, 2*H - 1, 2*D - 1
    W2s, H2s = W2, H2
    slices_z = split_domain_1d(D, K)
    boundary_z_orig = [z1 for _, z1 in slices_z[:-1]]

    t0 = time.time()
    l_inf = 0.0

    # ── Round 0 ───────────────────────────────────────────────────────────────
    u_assembled = cp.zeros((W2, H2, D2), dtype=cp.float32)
    sub_meta = []

    for k, (z0, z1) in enumerate(slices_z):
        z0_ext = max(0, z0 - delta)
        z1_ext = min(D, z1 + delta)
        sub_vol = vol_u8[:, :, z0_ext:z1_ext]

        U_lo, U_hi, _ = build_ispan_gpu(sub_vol)
        u_sub_pad = front_propagation_gpu(U_lo, U_hi, l_inf, verbose=False)
        u_sub = u_sub_pad[1:-1, 1:-1, 1:-1]

        inner_loc_z0 = z0 - z0_ext
        inner_loc_z1 = inner_loc_z0 + (z1 - z0)
        ez_s = 2 * inner_loc_z0 if k > 0 else 0
        ez_e = 2 * inner_loc_z1 - 1 if k < K - 1 else u_sub.shape[2]
        gz_s = 2 * z0 if k > 0 else 0
        gz_e = min(gz_s + (ez_e - ez_s), D2)
        actual = gz_e - gz_s
        if actual > 0 and ez_s + actual <= u_sub.shape[2]:
            u_assembled[:, :, gz_s:gz_e] = u_sub[:, :, ez_s:ez_s + actual]
        if k < K - 1:
            _assemble_boundary(u_assembled, u_sub, z1, z0_ext, D2)

        sub_meta.append({"k": k, "z0": z0, "z1": z1,
                          "z0_ext": z0_ext, "z1_ext": z1_ext})

    # ── IBI rounds ────────────────────────────────────────────────────────────
    R_star = 0
    for r in range(1, max_rounds + 1):
        u_prev = u_assembled.copy()
        u_new  = u_assembled.copy()

        for m in sub_meta:
            k      = m["k"];  z0 = m["z0"];  z1 = m["z1"]
            z0_ext = m["z0_ext"];  z1_ext = m["z1_ext"]
            sub_vol = vol_u8[:, :, z0_ext:z1_ext]

            U_lo, U_hi, _ = build_ispan_gpu(sub_vol)
            seeds = cp.full(U_lo.shape, cp.nan, dtype=cp.float32)

            if k > 0:
                _pin_and_seed_expanded(
                    U_lo, U_hi, seeds, u_assembled, z0_ext,
                    g_ez_start=2 * z0_ext,
                    g_ez_end  =2 * z0 + 1,
                    W2s=W2s, H2s=H2s,
                )
            if k < K - 1:
                _pin_and_seed_expanded(
                    U_lo, U_hi, seeds, u_assembled, z0_ext,
                    g_ez_start=2 * z1,
                    g_ez_end  =2 * z1_ext,
                    W2s=W2s, H2s=H2s,
                )

            u_sub_pad = front_propagation_gpu(
                U_lo, U_hi, l_inf,
                boundary_seeds=seeds,
                verbose=False,
            )
            u_sub = u_sub_pad[1:-1, 1:-1, 1:-1]

            inner_loc_z0 = z0 - z0_ext
            inner_loc_z1 = inner_loc_z0 + (z1 - z0)
            ez_s = 2 * inner_loc_z0 if k > 0 else 0
            ez_e = 2 * inner_loc_z1 - 1 if k < K - 1 else u_sub.shape[2]
            gz_s = 2 * z0 if k > 0 else 0
            gz_e = min(gz_s + (ez_e - ez_s), D2)
            actual = gz_e - gz_s
            if actual > 0 and ez_s + actual <= u_sub.shape[2]:
                u_new[:, :, gz_s:gz_e] = u_sub[:, :, ez_s:ez_s + actual]
            if k < K - 1:
                _assemble_boundary(u_new, u_sub, z1, z0_ext, D2)

        max_change = float(cp.abs(u_new - u_prev).max().item())
        u_assembled = u_new
        R_star = r

        if max_change < eps:
            break

    u_out = cp.asnumpy(u_assembled)
    bdry_viol = count_boundary_violations(u_out, boundary_z_orig)

    return {
        "R_star":     R_star,
        "violations": bdry_viol,
        "t_s":        round(time.time() - t0, 2),
    }


# ── subject sampling ──────────────────────────────────────────────────────────

def get_subject_paths(brats_dir: str,
                      n_subjects: int = 20,
                      seed: int = 42) -> list[tuple[str, str]]:
    flair_files = sorted(glob.glob(
        os.path.join(brats_dir, "*", "*_flair.nii*")))
    if not flair_files:
        sys.exit(f"ERROR: No FLAIR files found under {brats_dir}")

    path_map: dict[str, str] = {}
    for fp in flair_files:
        path_map[Path(fp).parent.name] = fp

    all_names = sorted(path_map.keys())
    if len(all_names) < n_subjects:
        warnings.warn(f"Only {len(all_names)} subjects found "
                      f"(requested {n_subjects}).")
        n_subjects = len(all_names)

    random.seed(seed)
    selected = sorted(random.sample(all_names, n_subjects))
    return [(name, path_map[name]) for name in selected]


# ── resume helper ─────────────────────────────────────────────────────────────

def load_existing(out_path: Path) -> list[dict]:
    """Load completed records from a previous run (for resume)."""
    if not out_path.exists():
        return []
    try:
        data = json.loads(out_path.read_text())
        records = data.get("records", [])
        print(f"[resume] Loaded {len(records)} existing records from {out_path}")
        return records
    except Exception as e:
        warnings.warn(f"Could not load existing output ({e}); starting fresh.")
        return []


def _done_key(r: dict) -> tuple:
    return (r["subject"], r["K"], r["delta"], r["eps"])


# ── inline summary ────────────────────────────────────────────────────────────

def print_summary(records: list[dict],
                  K_list: list[int],
                  delta_list: list[int],
                  eps_list: list[float],
                  max_rounds: int) -> None:
    n_subj = len(set(r["subject"] for r in records))
    print(f"\n{'='*70}")
    print(f"  Summary  (N={n_subj} subjects, R_max={max_rounds})")
    print(f"{'='*70}")
    hdr = f"  {'K':>3}  {'δ':>2}  {'ε':>4}  {'Pass':>6}  "
    hdr += f"{'Viol mean±std':>16}  {'R* mean±std':>14}  {'R* range':>10}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for K, delta in product(K_list, delta_list):
        for eps in eps_list:
            sub = [r for r in records
                   if r["K"] == K and r["delta"] == delta and r["eps"] == eps]
            if not sub:
                continue
            viols = [r["violations"] for r in sub]
            rds   = [r["rounds"]     for r in sub]
            n     = len(sub)
            passes = sum(1 for v in viols if v == 0)
            std_v = np.std(viols, ddof=1) if n > 1 else 0.0
            std_r = np.std(rds,   ddof=1) if n > 1 else 0.0
            print(
                f"  {K:>3}  {delta:>2}  {eps:>4.1f}  "
                f"{passes:>2}/{n:<3}  "
                f"{np.mean(viols):>6.2f} ± {std_v:>5.2f}  "
                f"{np.mean(rds):>6.2f} ± {std_r:>5.2f}  "
                f"[{min(rds)}, {max(rds)}]"
            )
    print(f"{'='*70}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description="E-NEW-4: ε sensitivity sweep for DD-FP IBI convergence"
    )
    ap.add_argument("--brats",       required=True,
                    help="BraTS 2021 Training root directory")
    ap.add_argument("--seed",        type=int, default=42,
                    help="Random seed for subject sampling (default: 42)")
    ap.add_argument("--n_subjects",  type=int, default=20,
                    help="Subjects to sample (default: 20)")
    ap.add_argument("--K_list",      type=int, nargs="+",
                    default=[2, 4, 8, 16],
                    help="K values to sweep (default: 2 4 8 16)")
    ap.add_argument("--delta_list",  type=int, nargs="+",
                    default=[1, 2],
                    help="δ values to sweep (default: 1 2)")
    ap.add_argument("--eps_list",    type=float, nargs="+",
                    default=[0.1, 0.5, 1.0, 2.0],
                    help="ε values to sweep (default: 0.1 0.5 1.0 2.0)")
    ap.add_argument("--max_rounds",  type=int, default=8)
    ap.add_argument("--exp_dir",     default=".",
                    help="Directory containing experiment_DDFP_all.py")
    ap.add_argument("--out",         default="epsilon_sensitivity_full.json",
                    help="Output JSON path (existing file → auto-resume)")
    args = ap.parse_args()

    _add_exp_dir(args.exp_dir)
    from experiment_DDFP_all import preprocess

    K_list     = sorted(args.K_list)
    delta_list = sorted(args.delta_list)
    eps_list   = sorted(args.eps_list)
    out_path   = Path(args.out)

    # ── subjects ──────────────────────────────────────────────────────────────
    subjects = get_subject_paths(args.brats,
                                 n_subjects=args.n_subjects,
                                 seed=args.seed)
    print(f"Subjects: {len(subjects)}  (seed={args.seed})")
    for name, _ in subjects:
        print(f"  {name}")

    # ── resume ────────────────────────────────────────────────────────────────
    records  = load_existing(out_path)
    done_set = {_done_key(r) for r in records}

    configs   = list(product(K_list, delta_list))
    n_total   = len(subjects) * len(configs) * len(eps_list)
    n_skip    = sum(1 for subj, K, d, e
                    in product([n for n,_ in subjects], K_list, delta_list, eps_list)
                    if (subj, K, d, e) in done_set)
    n_remain  = n_total - n_skip

    print(f"\nConfigs: K={K_list} × δ={delta_list} × ε={eps_list}")
    print(f"Total runs: {n_total}  |  skip (done): {n_skip}  |  remaining: {n_remain}")

    if n_remain == 0:
        print("All runs already complete.")
        print_summary(records, K_list, delta_list, eps_list, args.max_rounds)
        return

    # ── sweep ─────────────────────────────────────────────────────────────────
    done_count = n_skip
    for K, delta in configs:
        print(f"\n{'─'*60}")
        print(f"  K={K}, δ={delta}")
        print(f"{'─'*60}")

        header = f"  {'Subject':<30}" + "".join(
            f"  ε={e}" for e in eps_list)
        print(header)

        for name, flair_path in subjects:
            # skip if all eps already done for this (name, K, delta)
            if all((name, K, delta, e) in done_set for e in eps_list):
                print(f"  {name:<30}  [skip — all ε done]")
                continue

            vol_u8 = preprocess(
                nib.load(flair_path).get_fdata(dtype=np.float32))

            row = f"  {name:<30}"
            for eps in eps_list:
                if (name, K, delta, eps) in done_set:
                    row += f"  [skip]"
                    continue

                res = run_ibi_eps(vol_u8, K=K, delta=delta,
                                  eps=eps, max_rounds=args.max_rounds)
                done_count += 1

                record = {
                    "subject":    name,
                    "K":          K,
                    "delta":      delta,
                    "eps":        eps,
                    "violations": res["violations"],
                    "rounds":     res["R_star"],
                    "t_s":        res["t_s"],
                }
                records.append(record)
                done_set.add((name, K, delta, eps))

                tag = "✓" if res["violations"] == 0 else f"✗{res['violations']}"
                row += f"  R={res['R_star']}{tag}"

                # incremental save after every record
                _save(records, out_path, K_list, delta_list, eps_list,
                      args.max_rounds, args.seed, args.n_subjects)

            print(row)

    # ── final summary ─────────────────────────────────────────────────────────
    print_summary(records, K_list, delta_list, eps_list, args.max_rounds)
    print(f"\nSaved → {out_path}")
    print("Next: python scripts/extras/analyse_epsilon.py "
          f"--input {out_path}")


def _save(records, out_path, K_list, delta_list, eps_list,
          max_rounds, seed, n_subjects):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "meta": {
            "K_list":      K_list,
            "delta_list":  delta_list,
            "eps_list":    eps_list,
            "max_rounds":  max_rounds,
            "n_subjects":  n_subjects,
            "seed":        seed,
        },
        "records": records,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()