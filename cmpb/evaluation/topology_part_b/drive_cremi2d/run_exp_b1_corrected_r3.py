#!/usr/bin/env python3
"""
Fail-closed runner for the corrected DD-FP Part-B Exp-1 CSV.

Purpose
-------
Produce `exp_b1_corrected.csv` using the repository's original
`scripts/part_b/exp_b1_topology_accuracy.py`, while checking that:

1. the local repository contains the frozen public base commit
   8fdfe6233dfe53f996ed2f09e8810bca53d5323b;
2. the experiment script is byte-identical to that base-commit version;
3. the R3 production indexing/native-worklist correction is present in the
   current working tree;
4. the generated CSV reproduces the corrected DRIVE gates.

The R3 application kit changes the FP implementation, not the B1 experiment
script. Therefore the corrected CSV is obtained by running the unchanged B1
experiment script against the corrected core implementation.

Example
-------
python run_exp_b1_corrected_r3.py \
  --repo /path/to/DD-FP \
  --drive-gt /path/to/DRIVE/training/1st_manual \
  --cremi-gt /path/to/CREMI/masks \
  --output /path/to/exp_b1_corrected.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np


BASE_COMMIT = "8fdfe6233dfe53f996ed2f09e8810bca53d5323b"
EXP_SCRIPT = Path("scripts/part_b/exp_b1_topology_accuracy.py")

EXPECTED_PREPS = {"no_interp", "naive_interp", "ddfp", "seq_fp"}
EXPECTED_DRIVE_N = 20
EXPECTED_CREMI_N = 50


def die(msg: str) -> "NoReturn":
    raise SystemExit(f"[FAIL] {msg}")


def run(cmd: list[str], *, cwd: Path, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        check=True,
        text=True,
        capture_output=capture,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_repository(repo: Path) -> dict:
    if not (repo / ".git").exists():
        die(f"not a Git repository: {repo}")

    # The exact public base must exist locally.
    try:
        run(["git", "cat-file", "-e", f"{BASE_COMMIT}^{{commit}}"], cwd=repo)
    except subprocess.CalledProcessError:
        die(f"base commit {BASE_COMMIT} is not available in this clone")

    # Recover the exact original B1 script from the frozen base commit.
    base_script = run(
        ["git", "show", f"{BASE_COMMIT}:{EXP_SCRIPT.as_posix()}"],
        cwd=repo,
    ).stdout.encode("utf-8")

    current_path = repo / EXP_SCRIPT
    if not current_path.is_file():
        die(f"missing experiment script: {current_path}")
    current_script = current_path.read_bytes()

    if current_script != base_script:
        die(
            "current exp_b1_topology_accuracy.py differs from the frozen base "
            "version. Do not silently mix experiment-script revisions."
        )

    # Fail-closed markers for the R3 corrected production core.
    cpu = (repo / "src/ddfp/cpu_fp.py").read_text(encoding="utf-8")
    gpu = (repo / "src/ddfp/gpu_immersion.py").read_text(encoding="utf-8")

    markers = {
        "cpu_c_order_indexing": "(nx * Hp + ny) * Dp + nz" in cpu,
        "gpu_c_order_indexing": "(nx * Hp + ny) * Dp + nz" in gpu,
        "gpu_native_seed_aware_route":
            "seed_aware_boutry_bucket_gpu" in gpu
            and "tie_policy=\"lower\"" in gpu,
    }
    if not all(markers.values()):
        die(
            "R3 corrected core markers are incomplete. Apply the supplied "
            "DD_FP_GITHUB_APPLICATION_KIT_R3_20260906 first."
        )

    return {
        "base_commit": BASE_COMMIT,
        "experiment_script_sha256": sha256_bytes(current_script),
        **markers,
    }


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        die("generated CSV is empty")
    return rows


def f(row: dict, key: str) -> float:
    return float(row[key])


def validate_corrected_csv(path: Path, require_cremi: bool) -> dict:
    rows = load_csv(path)

    required = {
        "dataset", "sample", "preprocessing",
        "beta0_4conn", "beta0_8conn", "cc", "tsi",
        "b0_consistency", "chi_sign_flip",
        "dwc_violations", "dwc_viol_rate",
    }
    missing = required - set(rows[0])
    if missing:
        die(f"CSV missing required columns: {sorted(missing)}")

    drive = [r for r in rows if r["dataset"] == "drive"]
    drive_samples = sorted({r["sample"] for r in drive})
    if len(drive_samples) != EXPECTED_DRIVE_N:
        die(f"expected 20 DRIVE samples, found {len(drive_samples)}")
    if len(drive) != EXPECTED_DRIVE_N * 4:
        die(f"expected 80 DRIVE rows, found {len(drive)}")

    for sample in drive_samples:
        preps = {r["preprocessing"] for r in drive if r["sample"] == sample}
        if preps != EXPECTED_PREPS:
            die(f"{sample}: preprocessing set mismatch: {sorted(preps)}")

    cremi = [r for r in rows if r["dataset"] == "cremi"]
    cremi_samples = sorted({r["sample"] for r in cremi})
    if require_cremi:
        if len(cremi_samples) != EXPECTED_CREMI_N:
            die(f"expected 50 CREMI samples, found {len(cremi_samples)}")
        if len(cremi) != EXPECTED_CREMI_N * 4:
            die(f"expected 200 CREMI rows, found {len(cremi)}")
        if len(rows) != 280:
            die(f"expected historical full output size 280 rows, found {len(rows)}")

    def arm(name: str) -> list[dict]:
        return [r for r in drive if r["preprocessing"] == name]

    no = arm("no_interp")
    naive = arm("naive_interp")
    dd = arm("ddfp")
    seq = arm("seq_fp")

    gates = {
        "drive_n": len(drive_samples),
        "naive_tsi_mean": float(np.mean([f(r, "tsi") for r in naive])),
        "naive_tsi_max": float(np.max([f(r, "tsi") for r in naive])),
        "naive_tsi_sd_population": float(np.std([f(r, "tsi") for r in naive])),
        "naive_chi_flip_count": int(sum(int(r["chi_sign_flip"]) for r in naive)),
        "no_interp_cc_mean": float(np.mean([f(r, "cc") for r in no])),
        "no_interp_dwc_rate_mean": float(np.mean([f(r, "dwc_viol_rate") for r in no])),
        "ddfp_tsi_max": float(np.max([f(r, "tsi") for r in dd])),
        "ddfp_cc_max": float(np.max([f(r, "cc") for r in dd])),
        "ddfp_dwc_violations_sum": int(sum(int(r["dwc_violations"]) for r in dd)),
    }

    # Retained corrected endpoints. Use tolerances only where CSV rounding is involved.
    if round(gates["naive_tsi_mean"], 2) != 222.74:
        die(f"naive DRIVE mean TSI mismatch: {gates['naive_tsi_mean']}")
    if round(gates["naive_tsi_max"], 2) != 523.74:
        die(f"naive DRIVE max TSI mismatch: {gates['naive_tsi_max']}")
    if round(gates["naive_tsi_sd_population"], 2) != 144.15:
        die(f"naive DRIVE TSI SD mismatch: {gates['naive_tsi_sd_population']}")
    if gates["naive_chi_flip_count"] != 19:
        die(f"naive DRIVE chi-flip mismatch: {gates['naive_chi_flip_count']}/20")
    if abs(gates["no_interp_cc_mean"] - 448.25) > 1e-12:
        die(f"no_interp DRIVE mean CC mismatch: {gates['no_interp_cc_mean']}")
    if abs(gates["no_interp_dwc_rate_mean"] - 3.63e-4) > 5e-7:
        die(
            "no_interp DRIVE mean DWC-rate mismatch: "
            f"{gates['no_interp_dwc_rate_mean']}"
        )
    if gates["ddfp_tsi_max"] != 0:
        die(f"DD-FP TSI is not identically zero: max={gates['ddfp_tsi_max']}")
    if gates["ddfp_cc_max"] != 0:
        die(f"DD-FP CC is not identically zero: max={gates['ddfp_cc_max']}")
    if gates["ddfp_dwc_violations_sum"] != 0:
        die(
            "DD-FP DWC violations are nonzero: "
            f"sum={gates['ddfp_dwc_violations_sum']}"
        )

    # Reported metric-level DD-FP / sequential-FP equality.
    for key in ("cc", "tsi", "b0_consistency", "dwc_viol_rate"):
        a = np.array([f(r, key) for r in dd], dtype=float)
        b = np.array([f(r, key) for r in seq], dtype=float)
        if not np.array_equal(a, b):
            die(f"DD-FP / SeqFP mismatch in {key}")

    gates["csv_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    gates["total_rows"] = len(rows)
    gates["cremi_n"] = len(cremi_samples)
    return gates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--drive-gt", type=Path, required=True)
    ap.add_argument("--cremi-gt", type=Path, required=True)
    ap.add_argument("--cremi-limit", type=int, default=50)
    ap.add_argument("--output", type=Path, default=Path("exp_b1_corrected.csv"))
    ap.add_argument(
        "--drive-only",
        action="store_true",
        help="Allow missing CREMI/full-280-row gate. Figure 5 itself only needs DRIVE.",
    )
    ap.add_argument(
        "--recover-original-source",
        type=Path,
        default=None,
        help="Optional path to save the exact base-commit B1 script recovered via git show.",
    )
    args = ap.parse_args()

    repo = args.repo.resolve()
    out = args.output.resolve()

    provenance = verify_repository(repo)
    print("[PASS] repository/source gate")
    for k, v in provenance.items():
        print(f"  {k}: {v}")

    if not args.drive_gt.is_dir():
        die(f"DRIVE directory not found: {args.drive_gt}")
    if not args.drive_only and not args.cremi_gt.is_dir():
        die(f"CREMI directory not found: {args.cremi_gt}")

    if args.recover_original_source is not None:
        src = run(
            ["git", "show", f"{BASE_COMMIT}:{EXP_SCRIPT.as_posix()}"],
            cwd=repo,
        ).stdout
        target = args.recover_original_source.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(src, encoding="utf-8")
        print(f"[PASS] exact original B1 source recovered -> {target}")

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo / EXP_SCRIPT),
        "--drive-gt", str(args.drive_gt.resolve()),
        "--cremi-gt", str(args.cremi_gt.resolve()),
        "--cremi-limit", str(args.cremi_limit),
        "--output", str(out),
    ]

    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(repo))
    if proc.returncode != 0:
        die(f"experiment exited with code {proc.returncode}")

    gates = validate_corrected_csv(out, require_cremi=not args.drive_only)

    print("\nEXP_B1_CORRECTED_GATE_PASS")
    for k, v in gates.items():
        print(f"  {k}: {v}")
    print(f"\nRESULT: {out}")


if __name__ == "__main__":
    main()
