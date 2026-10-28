from __future__ import annotations

import argparse
import csv
import gc
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


# ----------------------------------------------------------------------
# Frozen production provenance expected for P5
# ----------------------------------------------------------------------

EXPECTED_HEAD = "b5be086b9388a607ff70c2ef360c001c217cc238"
BASE_HEAD = "8fdfe6233dfe53f996ed2f09e8810bca53d5323b"
P3_PATCH_PATHS = {
    "src/ddfp/cpu_fp.py",
    "src/ddfp/gpu_immersion.py",
    "src/ddfp/gpu_native_worklist.py",
    "src/ddfp/gpu_seed_aware_worklist.py",
    "test/test_smoke.py",
}

EXPECTED_EXP_B3_SHA256 = "6be95f2c698d1fc5d3aa841bcf1e7c2d612d6ff9603a826449f3c2fa566e16d3"


EXPECTED_SHA256 = {
    "scripts/ddfp/experiment_DDFP_all.py":
        "b6a9b109f4105e7b77ebed880f669ba670bbd87123deca9509ba5b00993e561e",
    "src/ddfp/gpu_immersion.py":
        "b784a376d3466a1d4ded5089a2eee9f27df35006b4926301dd06106b4ed262a4",
    "src/ddfp/gpu_native_worklist.py":
        "853bf33a6bbd9c91f5304a018f0a51ff53e8418360fe1477208cf81df17163ac",
    "src/ddfp/gpu_seed_aware_worklist.py":
        "c80f56b9ba7afd63196332d51894168d36f6e9f3e2a51b6b966786311e4eecad",
}


# Historical full-N references.
# These are comparison targets, NOT execution pass/fail criteria.
HISTORICAL = {
    (2, 1): {
        "dwc_pass": 1251,
        "r_eq_2": 1147,
    },
    (2, 2): {
        "dwc_pass": 1251,
        "r_gt_2": 1060,
        "r_mean_2dp": 6.65,
    },
    (4, 1): {
        "dwc_pass": 1194,
    },
    (8, 1): {
        "dwc_pass": 42,
    },
    (16, 1): {
        "dwc_pass": 1251,
    },
    (16, 2): {
        "dwc_pass": 1251,
    },
}


FIELDS = [
    "subject_id",
    "K",
    "delta",
    "R_star",
    "bdry_viol",
    "dwc_ok",
    "time_s",
    "D",
    "D_sub",
    "sigma_cover",
    "rho_pct",
    "kappa_med",
    "kappa_max",
    "max_rounds",
    "ref_chi",
    "ddfp_chi",
    "chi_sign_flip",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_provenance(repo: Path) -> dict:
    head = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    # On the original workstation P5 ran at the local P4 freeze commit.
    # On a fresh server the equivalent production tree is reconstructed by
    # checking out BASE_HEAD and applying the frozen P3 five-path patch.
    if head not in {EXPECTED_HEAD, BASE_HEAD}:
        raise RuntimeError(
            f"Unexpected repo HEAD: {head}\n"
            f"Allowed: local P4 freeze {EXPECTED_HEAD} or public base {BASE_HEAD}"
        )

    actual = {}
    for rel, expected in EXPECTED_SHA256.items():
        p = repo / rel
        if not p.exists():
            raise FileNotFoundError(p)

        digest = sha256(p)
        actual[rel] = digest

        if digest != expected:
            raise RuntimeError(
                f"SHA mismatch: {rel}\n"
                f"actual   {digest}\n"
                f"expected {expected}"
            )

    status_lines = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--short"], text=True
    ).splitlines()
    changed_paths = set()
    for line in status_lines:
        if not line.strip():
            continue
        changed_paths.add(line[3:].strip().split(" -> ")[-1])

    if head == BASE_HEAD and changed_paths != P3_PATCH_PATHS:
        raise RuntimeError(
            "Fresh-server tree must contain exactly the five frozen P3 patch paths.\n"
            f"actual={sorted(changed_paths)}\nexpected={sorted(P3_PATCH_PATHS)}"
        )

    return {
        "repo": str(repo),
        "head": head,
        "server_equivalent_tree": head == BASE_HEAD,
        "changed_paths": sorted(changed_paths),
        "sha256": actual,
    }


def load_ref_chi_map(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"subject_id", "ref_chi"}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(
            f"Reference chi CSV must contain columns {sorted(required)}: {path}"
        )
    out = {}
    for r in rows:
        sid = r["subject_id"].strip()
        if not sid:
            raise RuntimeError("Blank subject_id in reference chi CSV")
        if sid in out:
            raise RuntimeError(f"Duplicate subject_id in reference chi CSV: {sid}")
        out[sid] = int(float(r["ref_chi"]))
    return out


def import_frozen_topology_metric(repo: Path):
    src = repo / "scripts" / "part_b" / "exp_b3_brats_3d_all.py"
    if not src.is_file():
        raise FileNotFoundError(
            f"Missing frozen BraTS topology metric source: {src}"
        )
    digest = sha256(src)
    if digest != EXPECTED_EXP_B3_SHA256:
        raise RuntimeError(
            "Frozen exp_b3 source hash mismatch.\n"
            f"actual   {digest}\n"
            f"expected {EXPECTED_EXP_B3_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("ddfp_exp_b3_chiflip_frozen", src)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {src}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    metric = getattr(mod, "topology_metrics_3d", None)
    if not callable(metric):
        raise RuntimeError("Frozen exp_b3 does not expose topology_metrics_3d")
    # χ is threshold-0.5 topology; avoid the unrelated nine-threshold TSI sweep.
    mod.tsi_3d = lambda *args, **kwargs: 0.0
    return metric, src, digest


def to_numpy_for_metric(arr):
    if hasattr(arr, "get") and callable(arr.get):
        arr = arr.get()
    return np.asarray(arr)


def measure_chi(metric_fn, u_dd, ref_chi: int, *, contract_check: bool):
    u_np = to_numpy_for_metric(u_dd)
    m0 = metric_fn(u_np, 0, int(ref_chi))
    if not isinstance(m0, dict) or "chi" not in m0 or "chi_sign_flip" not in m0:
        raise RuntimeError(
            "Frozen topology_metrics_3d did not return chi/chi_sign_flip"
        )
    chi = int(m0["chi"])
    flip = int(m0["chi_sign_flip"])
    expected_flip = int((int(ref_chi) != 0) and (chi * int(ref_chi) < 0))
    if flip != expected_flip:
        raise RuntimeError(
            f"chi_sign_flip formula mismatch: metric={flip}, expected={expected_flip}, "
            f"ref_chi={ref_chi}, ddfp_chi={chi}"
        )
    if contract_check:
        # ref_b0 is a legacy argument for this metric family. Verify empirically that
        # χ/χ-flip are invariant to it before trusting the χ-only measurement path.
        m1 = metric_fn(u_np, 123456789, int(ref_chi))
        if int(m1["chi"]) != chi or int(m1["chi_sign_flip"]) != flip:
            raise RuntimeError(
                "Frozen topology_metrics_3d χ outputs unexpectedly depend on ref_b0"
            )
    return chi, flip


def parse_configs(text: str):
    out = []
    seen = set()

    for item in text.split(","):
        k, d = item.strip().split(":")
        pair = (int(k), int(d))

        if pair not in seen:
            seen.add(pair)
            out.append(pair)

    return out


def write_atomic(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    os.replace(tmp, path)


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []

    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_paths(brats_dir: Path, n: int, seed: int):
    paths = sorted(
        glob.glob(str(brats_dir / "*" / "*_flair.nii*"))
    )

    total = len(paths)
    if total == 0:
        raise RuntimeError(f"No FLAIR files under {brats_dir}")

    if n == -1 or n >= total:
        return paths, total

    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=n, replace=False)

    return [paths[i] for i in sorted(idx)], total


def cleanup_gpu():
    gc.collect()
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def make_summary(
    rows: list[dict],
    configs,
    n_target: int,
    max_rounds: int,
    provenance: dict,
):
    groups = defaultdict(list)

    for r in rows:
        groups[(int(r["K"]), int(r["delta"]))].append(r)

    summary = {
        "target_subjects": n_target,
        "expected_rows": n_target * len(configs),
        "actual_rows": len(rows),
        "max_rounds": max_rounds,
        "configs": [],
        "provenance": provenance,
    }

    for K, delta in configs:
        rr = groups[(K, delta)]

        R = np.array([int(r["R_star"]) for r in rr], dtype=int)
        B = np.array([int(r["bdry_viol"]) for r in rr], dtype=int)
        F = np.array([int(r["chi_sign_flip"]) for r in rr], dtype=int)
        REFCHI = np.array([int(r["ref_chi"]) for r in rr], dtype=int)

        n = len(rr)
        n_pass = int((B == 0).sum())

        item = {
            "K": K,
            "delta": delta,
            "n": n,

            "dwc_pass": n_pass,
            "dwc_fail": int(n - n_pass),
            "dwc_pass_pct": (
                round(100.0 * n_pass / n, 4) if n else None
            ),

            "R_eq_1": int((R == 1).sum()) if n else 0,
            "R_eq_2": int((R == 2).sum()) if n else 0,
            "R_le_2": int((R <= 2).sum()) if n else 0,
            "R_gt_2": int((R > 2).sum()) if n else 0,

            "R_eq_2_pct": (
                round(100.0 * (R == 2).sum() / n, 4)
                if n else None
            ),
            "R_le_2_pct": (
                round(100.0 * (R <= 2).sum() / n, 4)
                if n else None
            ),

            "R_mean": round(float(R.mean()), 6) if n else None,
            "R_median": float(np.median(R)) if n else None,
            "R_min": int(R.min()) if n else None,
            "R_max": int(R.max()) if n else None,

            # run_ibi_v10 does not distinguish "converged exactly at cap"
            # from "hit cap"; therefore this is intentionally neutral.
            "R_eq_max_rounds": int((R == max_rounds).sum()) if n else 0,

            "bdry_viol_sum": int(B.sum()) if n else 0,
            "bdry_viol_max": int(B.max()) if n else None,
            "chi_flip_n": int(F.sum()) if n else 0,
            "chi_flip_pct": round(100.0 * float(F.sum()) / n, 6) if n else None,
            "ref_chi_zero_n": int((REFCHI == 0).sum()) if n else 0,
        }

        hist = HISTORICAL.get((K, delta))
        comparison = {}

        if hist and n == 1251:
            if "dwc_pass" in hist:
                comparison["dwc_pass"] = {
                    "historical": hist["dwc_pass"],
                    "corrected": item["dwc_pass"],
                    "status": (
                        "UNCHANGED"
                        if item["dwc_pass"] == hist["dwc_pass"]
                        else "CHANGED"
                    ),
                }

            if "r_eq_2" in hist:
                comparison["R_eq_2"] = {
                    "historical": hist["r_eq_2"],
                    "corrected": item["R_eq_2"],
                    "status": (
                        "UNCHANGED"
                        if item["R_eq_2"] == hist["r_eq_2"]
                        else "CHANGED"
                    ),
                }

            if "r_gt_2" in hist:
                comparison["R_gt_2"] = {
                    "historical": hist["r_gt_2"],
                    "corrected": item["R_gt_2"],
                    "status": (
                        "UNCHANGED"
                        if item["R_gt_2"] == hist["r_gt_2"]
                        else "CHANGED"
                    ),
                }

            if "r_mean_2dp" in hist:
                corrected_2dp = round(item["R_mean"], 2)
                comparison["R_mean_2dp"] = {
                    "historical": hist["r_mean_2dp"],
                    "corrected": corrected_2dp,
                    "status": (
                        "UNCHANGED"
                        if corrected_2dp == hist["r_mean_2dp"]
                        else "CHANGED"
                    ),
                }

        item["historical_comparison"] = comparison
        summary["configs"].append(item)

    return summary


def print_summary(summary):
    print()
    print("=" * 92)
    print("P5-S4C corrected K×delta×R* summary")
    print("=" * 92)

    print(
        f"{'K':>3} {'d':>3} {'N':>5} "
        f"{'DWC pass':>12} {'pass%':>8} "
        f"{'R=2':>7} {'R<=2':>7} {'R>2':>7} "
        f"{'meanR':>8} {'maxR':>6}"
    )

    for x in summary["configs"]:
        print(
            f"{x['K']:>3} {x['delta']:>3} {x['n']:>5} "
            f"{x['dwc_pass']:>5}/{x['n']:<5} "
            f"{x['dwc_pass_pct']:>7.2f}% "
            f"{x['R_eq_2']:>7} "
            f"{x['R_le_2']:>7} "
            f"{x['R_gt_2']:>7} "
            f"{x['R_mean']:>8.3f} "
            f"{x['R_max']:>6}"
        )

        for metric, c in x["historical_comparison"].items():
            print(
                f"      historical {metric}: "
                f"{c['historical']} -> {c['corrected']} "
                f"[{c['status']}]"
            )

    print("=" * 92)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--repo", required=True)
    ap.add_argument("--brats-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--failures", required=True)
    ap.add_argument("--ref-chi-csv", required=True)

    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-rounds", type=int, default=8)

    ap.add_argument(
        "--configs",
        default="2:1,2:2,4:1,4:2,8:1,8:2,16:1,16:2",
    )
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-kappa", action="store_true")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)

    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    brats_dir = Path(args.brats_dir).resolve()
    out = Path(args.output).resolve()
    summary_path = Path(args.summary).resolve()
    failure_path = Path(args.failures).resolve()
    ref_chi_path = Path(args.ref_chi_csv).resolve()

    provenance = verify_provenance(repo)
    ref_chi_map = load_ref_chi_map(ref_chi_path)
    metric_fn, metric_src, metric_sha = import_frozen_topology_metric(repo)
    provenance["chi_measurement"] = {
        "metric_source": str(metric_src),
        "metric_sha256": metric_sha,
        "metric_function": "topology_metrics_3d",
        "tsi_stubbed_for_chi_only": True,
        "reference_chi_csv": str(ref_chi_path),
        "reference_chi_csv_sha256": sha256(ref_chi_path),
        "reference_chi_subjects": len(ref_chi_map),
    }

    sys.path.insert(0, str(repo))

    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CUDA GPU not available")

    from scripts.ddfp.experiment_DDFP_all import (
        preprocess,
        run_ibi_v10,
        count_boundary_violations,
        _compute_kappa,
    )

    configs = parse_configs(args.configs)
    paths, dataset_total = select_paths(
        brats_dir, args.n, args.seed
    )
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError("Require num_shards>=1 and 0<=shard_index<num_shards")
    selected_total_before_shard = len(paths)
    paths = paths[args.shard_index::args.num_shards]

    if args.resume:
        rows = load_rows(out)
    else:
        rows = []

    if rows:
        missing_checkpoint_columns = sorted(set(FIELDS) - set(rows[0]))
        if missing_checkpoint_columns:
            raise RuntimeError(
                "Existing checkpoint predates the chi-flip patch; use the new CHIFLIP "
                f"result directory. Missing columns={missing_checkpoint_columns}"
            )

    # Reject duplicates in checkpoint.
    keys = [
        (r["subject_id"], int(r["K"]), int(r["delta"]))
        for r in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate rows already present in checkpoint")

    done = set(keys)

    target_keys = []
    for p in paths:
        sid = Path(p).parent.name
        for K, delta in configs:
            target_keys.append((sid, K, delta))

    remaining = [
        k for k in target_keys if k not in done
    ]

    selected_subject_ids = {Path(p).parent.name for p in paths}
    missing_ref_chi = sorted(selected_subject_ids - set(ref_chi_map))
    if missing_ref_chi:
        raise RuntimeError(
            f"Reference chi mapping is missing {len(missing_ref_chi)} selected subjects; "
            f"first={missing_ref_chi[:5]}"
        )

    print("P5-S4C PREFLIGHT PASS")
    print("REPO_HEAD =", provenance["head"])
    print("DATASET_DISCOVERED =", dataset_total)
    print("SELECTED_TOTAL_BEFORE_SHARD =", selected_total_before_shard)
    print("SHARD_INDEX =", args.shard_index)
    print("NUM_SHARDS =", args.num_shards)
    print("TARGET_SUBJECTS =", len(paths))
    print("CONFIGS =", configs)
    print("EXPECTED_ROWS =", len(target_keys))
    print("RESUME_ROWS =", len(rows))
    print("REMAINING_CONFIG_RUNS =", len(remaining))
    print("MAX_ROUNDS =", args.max_rounds)
    print("REF_CHI_CSV =", ref_chi_path)
    print("REF_CHI_SUBJECTS =", len(ref_chi_map))
    print("CHI_METRIC_SHA256 =", metric_sha)

    failures = []
    chi_contract_checked = False

    pbar = tqdm(
        total=len(target_keys),
        initial=len(done),
        desc="P5-S4C",
        unit="cfg",
        dynamic_ncols=True,
        smoothing=0.1,
    )

    # Process one volume once, then all missing configs for it.
    for p in paths:
        sid = Path(p).parent.name

        missing = [
            (K, d)
            for K, d in configs
            if (sid, K, d) not in done
        ]

        if not missing:
            continue

        try:
            vol = preprocess(
                nib.load(p).get_fdata(dtype=np.float32)
            )

            if vol.size == 0:
                raise RuntimeError("empty volume")

            D = int(vol.shape[2])

            # κ does not depend on delta, so calculate only once per K.
            kappa_cache = {}

            for K, delta in missing:
                pbar.set_description(
                    f"{sid} K={K} d={delta}"
                )

                try:
                    t0 = time.perf_counter()

                    res = run_ibi_v10(
                        vol,
                        K=K,
                        delta=delta,
                        max_rounds=args.max_rounds,
                    )

                    b = int(
                        count_boundary_violations(
                            res["u_dd"],
                            res["boundary_z_orig"],
                        )
                    )

                    ref_chi = int(ref_chi_map[sid])
                    ddfp_chi, chi_flip = measure_chi(
                        metric_fn,
                        res["u_dd"],
                        ref_chi,
                        contract_check=not chi_contract_checked,
                    )
                    chi_contract_checked = True

                    elapsed = time.perf_counter() - t0

                    if args.skip_kappa:
                        kappa_med = np.nan
                        kappa_max = np.nan
                    else:
                        if K not in kappa_cache:
                            kappa_cache[K] = _compute_kappa(
                                vol,
                                res["boundary_z_orig"],
                                K,
                            )
                        kappa_med, kappa_max = kappa_cache[K]

                    D_sub = D // K
                    sigma_cover = D_sub // 2
                    rho_pct = (
                        round(delta / D_sub * 100.0, 1)
                        if D_sub > 0 else 0.0
                    )

                    row = {
                        "subject_id": sid,
                        "K": K,
                        "delta": delta,
                        "R_star": int(res["R_star"]),
                        "bdry_viol": b,
                        "dwc_ok": int(b == 0),
                        "time_s": round(elapsed, 6),
                        "D": D,
                        "D_sub": D_sub,
                        "sigma_cover": sigma_cover,
                        "rho_pct": rho_pct,
                        "kappa_med": (
                            ""
                            if np.isnan(kappa_med)
                            else round(float(kappa_med), 1)
                        ),
                        "kappa_max": (
                            ""
                            if np.isnan(kappa_max)
                            else round(float(kappa_max), 1)
                        ),
                        "max_rounds": args.max_rounds,
                        "ref_chi": ref_chi,
                        "ddfp_chi": ddfp_chi,
                        "chi_sign_flip": chi_flip,
                    }

                    rows.append(row)
                    done.add((sid, K, delta))

                    # Atomic checkpoint after every configuration.
                    write_atomic(out, rows)

                    pbar.set_postfix(
                        R=row["R_star"],
                        viol=b,
                        t=f"{elapsed:.1f}s",
                        chi=chi_flip,
                        done=len(done),
                        fail=len(failures),
                    )

                except Exception as e:
                    failures.append(
                        {
                            "subject_id": sid,
                            "K": K,
                            "delta": delta,
                            "error": f"{type(e).__name__}: {e}",
                        }
                    )
                    tqdm.write(
                        f"FAIL {sid} K={K} d={delta}: "
                        f"{type(e).__name__}: {e}"
                    )

                finally:
                    cleanup_gpu()
                    pbar.update(1)

            del vol

        except Exception as e:
            # Volume-load failure affects all missing configs.
            for K, delta in missing:
                failures.append(
                    {
                        "subject_id": sid,
                        "K": K,
                        "delta": delta,
                        "error": f"{type(e).__name__}: {e}",
                    }
                )
                pbar.update(1)

            tqdm.write(
                f"FAIL LOAD {sid}: {type(e).__name__}: {e}"
            )

        finally:
            cleanup_gpu()

    pbar.close()

    failure_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["subject_id", "K", "delta", "error"],
            delimiter="\t",
        )
        w.writeheader()
        w.writerows(failures)

    # Reload final checkpoint and independently aggregate it.
    final_rows = load_rows(out)

    summary = make_summary(
        final_rows,
        configs=configs,
        n_target=len(paths),
        max_rounds=args.max_rounds,
        provenance=provenance,
    )
    summary["dataset_total"] = dataset_total
    summary["selected_total_before_shard"] = selected_total_before_shard
    summary["shard_index"] = args.shard_index
    summary["num_shards"] = args.num_shards
    summary["timing_policy"] = "time_s is operational only; do not use for TIPS/runtime benchmark"
    summary["chi_policy"] = (
        "ref_chi comes from the bundled 1251-subject direct full-N baseline freeze; "
        "ddfp_chi and chi_sign_flip are measured immediately from res['u_dd'] using "
        "the frozen exp_b3 topology_metrics_3d with TSI disabled."
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print_summary(summary)

    final_keys = {
        (r["subject_id"], int(r["K"]), int(r["delta"]))
        for r in final_rows
    }

    complete = (
        len(final_rows) == len(target_keys)
        and final_keys == set(target_keys)
        and len(failures) == 0
    )

    print()
    print("EXPECTED_ROWS =", len(target_keys))
    print("FINAL_ROWS =", len(final_rows))
    print("UNIQUE_KEYS =", len(final_keys))
    print("FAILURES_THIS_RUN =", len(failures))

    if complete:
        print("P5_S4C_EXECUTION_PASS")
        print("P6_K16_CHIFLIP_SHARD_PASS")
    else:
        print("P5_S4C_EXECUTION_FAIL")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
