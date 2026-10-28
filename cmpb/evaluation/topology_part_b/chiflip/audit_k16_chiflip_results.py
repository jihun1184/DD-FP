from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_ref(path: Path) -> dict[str, int]:
    rows = read_rows(path)
    out = {}
    for r in rows:
        sid = r["subject_id"]
        if sid in out:
            raise SystemExit(f"Duplicate reference subject: {sid}")
        out[sid] = int(float(r["ref_chi"]))
    return out


def count_failures(pattern: str) -> tuple[list[str], int]:
    files = sorted(glob.glob(pattern))
    count = 0
    for name in files:
        with open(name, encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index == 0 and line.lower().startswith("subject_id"):
                    continue
                count += int(bool(line.strip()))
    return files, count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--failure-glob", required=True)
    parser.add_argument("--expected-subjects", type=int, required=True)
    parser.add_argument("--ref-chi-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    ref_path = Path(args.ref_chi_csv).resolve()
    rows = read_rows(csv_path)
    ref_map = read_ref(ref_path)
    failure_files, failure_count = count_failures(args.failure_glob)

    required = {
        "subject_id", "K", "delta", "R_star", "bdry_viol", "dwc_ok",
        "D", "D_sub", "sigma_cover", "max_rounds",
        "ref_chi", "ddfp_chi", "chi_sign_flip",
    }
    columns = set(rows[0]) if rows else set()
    missing_columns = sorted(required - columns)

    keys = [(r["subject_id"], int(r["K"]), int(r["delta"])) for r in rows] if not missing_columns else []
    subjects = {k[0] for k in keys}
    duplicate_keys = len(keys) - len(set(keys))
    wrong_configs = sum((k, d) != (16, 1) for _, k, d in keys)

    boundary = [int(r["bdry_viol"]) for r in rows] if not missing_columns else []
    dwc = [int(r["dwc_ok"]) for r in rows] if not missing_columns else []
    rstar = [int(r["R_star"]) for r in rows] if not missing_columns else []
    max_rounds = [int(r["max_rounds"]) for r in rows] if not missing_columns else []
    refs = [int(r["ref_chi"]) for r in rows] if not missing_columns else []
    chis = [int(r["ddfp_chi"]) for r in rows] if not missing_columns else []
    flips = [int(r["chi_sign_flip"]) for r in rows] if not missing_columns else []

    inconsistent_dwc = sum(ok != int(viol == 0) for ok, viol in zip(dwc, boundary))
    invalid_rstar = sum(not (1 <= r <= cap) for r, cap in zip(rstar, max_rounds))
    invalid_flip = sum(f not in (0, 1) for f in flips)
    formula_mismatch = sum(
        f != int((ref != 0) and (chi * ref < 0))
        for f, ref, chi in zip(flips, refs, chis)
    )
    missing_ref_subjects = sorted(subjects - set(ref_map))
    ref_value_mismatch = sum(
        int(r["ref_chi"]) != ref_map.get(r["subject_id"], int(r["ref_chi"]))
        for r in rows
    ) if not missing_columns else 0

    integrity_checks = {
        "required_columns_present": not missing_columns,
        "row_count_exact": len(rows) == args.expected_subjects,
        "unique_subject_count_exact": len(subjects) == args.expected_subjects,
        "duplicate_key_count_zero": duplicate_keys == 0,
        "config_exactly_k16_delta1": wrong_configs == 0,
        "failure_record_count_zero": failure_count == 0,
        "dwc_field_consistent": inconsistent_dwc == 0,
        "rstar_within_recorded_cap": invalid_rstar == 0,
        "chi_flip_binary": invalid_flip == 0,
        "chi_flip_formula_exact": formula_mismatch == 0,
        "reference_subjects_present": not missing_ref_subjects,
        "reference_chi_values_exact": ref_value_mismatch == 0,
    }
    integrity_pass = all(integrity_checks.values())

    dwc_all = bool(rows) and sum(boundary) == 0 and all(dwc)
    rstar2_all = bool(rows) and all(r == 2 for r in rstar)
    chi_flip_n = sum(flips)
    result = {
        "scope": {"K": 16, "delta": 1, "expected_subjects": args.expected_subjects},
        "execution_integrity": {
            "pass": integrity_pass,
            "checks": integrity_checks,
            "observed_rows": len(rows),
            "observed_subjects": len(subjects),
            "duplicate_keys": duplicate_keys,
            "wrong_config_rows": wrong_configs,
            "failure_records": failure_count,
            "missing_columns": missing_columns,
            "inconsistent_dwc_rows": inconsistent_dwc,
            "invalid_rstar_rows": invalid_rstar,
            "invalid_chi_flip_rows": invalid_flip,
            "chi_formula_mismatch_rows": formula_mismatch,
            "missing_reference_subjects": missing_ref_subjects[:20],
            "reference_chi_mismatch_rows": ref_value_mismatch,
        },
        "scientific_results": {
            "dwc_pass": sum(dwc),
            "dwc_fail": len(dwc) - sum(dwc),
            "boundary_violation_sum": sum(boundary),
            "boundary_violation_max": max(boundary, default=None),
            "rstar_counts": dict(sorted(Counter(rstar).items())),
            "rstar_min": min(rstar, default=None),
            "rstar_max": max(rstar, default=None),
            "rstar_mean": sum(rstar) / len(rstar) if rstar else None,
            "all_subjects_dwc": dwc_all,
            "all_subjects_rstar2": rstar2_all,
            "dwc_verdict": "PASS" if dwc_all else "FAIL",
            "rstar2_verdict": "PASS" if rstar2_all else "FAIL",
            "chi_flip_n": chi_flip_n,
            "chi_flip_pct": (100.0 * chi_flip_n / len(flips)) if flips else None,
            "ref_chi_zero_n": sum(ref == 0 for ref in refs),
            "ddfp_chi_min": min(chis, default=None),
            "ddfp_chi_max": max(chis, default=None),
        },
        "source_files": {
            "csv": csv_path.name,
            "reference_chi_csv": ref_path.name,
            "failure_files": [Path(name).name for name in failure_files],
        },
        "timing_policy": "Any time_s field is execution provenance only and is excluded from runtime, speedup, and TIPS claims.",
    }

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print("P6_K16_CHIFLIP_EXECUTION_INTEGRITY_" + ("PASS" if integrity_pass else "FAIL"))
    print("P6_K16_DWC_" + result["scientific_results"]["dwc_verdict"])
    print("P6_K16_RSTAR2_" + result["scientific_results"]["rstar2_verdict"])
    if not integrity_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
