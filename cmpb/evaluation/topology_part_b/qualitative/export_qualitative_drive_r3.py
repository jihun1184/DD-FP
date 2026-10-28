#!/usr/bin/env python3
"""Export lossless DRIVE qualitative inputs from the audited R3 Part-B code."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from scipy.ndimage import zoom as scipy_zoom

from qualitative_common import (
    add_repo_to_path,
    boundary_difference,
    command_string,
    ensure_new_or_empty,
    git_head,
    roi_from_support,
    safe_case_id,
    save_array,
    sha256_file,
    software_versions,
    verify_repo,
    write_csv,
    write_json,
    write_sha256_manifest,
)


REQUIRED_R3 = (
    "scripts/part_b/exp_b1_topology_accuracy.py",
    "src/ddfp/cpu_fp.py",
    "src/ddfp/gpu_immersion.py",
    "src/ddfp/gpu_native_worklist.py",
    "src/ddfp/gpu_seed_aware_worklist.py",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True, help="DD-FP repo after applying R3")
    p.add_argument("--drive-root", type=Path, required=True, help="DRIVE root containing training/")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--split", default="training", choices=("training", "test"))
    p.add_argument("--manual-dir", type=Path, default=None)
    p.add_argument("--image-dir", type=Path, default=None)
    p.add_argument("--fov-dir", type=Path, default=None)
    p.add_argument("--case-id", action="append", default=[], help="numeric id or exact manual stem; repeatable")
    p.add_argument("--include-naive", action="store_true")
    p.add_argument("--roi-pad", type=int, default=20)
    p.add_argument("--roi-min-size", type=int, default=96)
    p.add_argument("--roi-max-size", type=int, default=192)
    p.add_argument("--allow-existing", action="store_true")
    return p.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def find_by_id(directory: Path, case_number: str, role: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = sorted(p for p in directory.iterdir() if p.is_file() and p.name.startswith(case_number + "_"))
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous {role} for case {case_number}: {matches}")
    return matches[0] if matches else None


def requested(stem: str, case_number: str, filters: list[str]) -> bool:
    if not filters:
        return True
    return stem in filters or case_number in filters


def expanded_rgb(rgb: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = rgb.shape[:2]
    th, tw = target_hw
    out = scipy_zoom(rgb.astype(np.float32), (th / h, tw / w, 1.0), order=1, prefilter=False)
    if out.shape[:2] != (th, tw):
        raise RuntimeError(f"fundus expansion shape mismatch: {out.shape[:2]} vs {(th, tw)}")
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    drive = args.drive_root.resolve()
    out_root = args.output_root.resolve()
    ensure_new_or_empty(out_root, args.allow_existing)

    r3_hashes = verify_repo(repo, REQUIRED_R3)
    add_repo_to_path(repo)
    from scripts.part_b.exp_b1_topology_accuracy import get_preprocessors, load_mask
    from src.ddfp import get_backend

    manual_dir = (args.manual_dir or drive / args.split / "1st_manual").resolve()
    image_dir = (args.image_dir or drive / args.split / "images").resolve()
    fov_dir = (args.fov_dir or drive / args.split / "mask").resolve()
    if not manual_dir.is_dir():
        raise FileNotFoundError(f"manual mask directory not found: {manual_dir}")

    mask_paths = sorted(p for p in manual_dir.iterdir() if p.is_file() and p.suffix.lower() in {".gif", ".png", ".tif", ".tiff"})
    rows = []
    case_jsons = []
    cfg = SimpleNamespace(preprocessing=SimpleNamespace(
        type="ddfp", naive_mode="bilinear", ddfp_overlap=1,
        ddfp_cache_dir=None, no_interp_cache_dir=None,
        naive_interp_cache_dir=None, seq_fp_cache_dir=None,
    ))
    preps = get_preprocessors(cfg)

    for mask_path in mask_paths:
        match = re.match(r"^(\d+)", mask_path.stem)
        if not match:
            continue
        number = match.group(1)
        if not requested(mask_path.stem, number, args.case_id):
            continue
        case_id = safe_case_id(mask_path.stem)
        image_path = find_by_id(image_dir, number, "fundus image")
        if image_path is None:
            raise FileNotFoundError(f"fundus image not found for {mask_path.name} in {image_dir}")
        fov_path = find_by_id(fov_dir, number, "FOV mask")

        manual = load_mask(mask_path).astype(np.uint8)
        fundus = load_rgb(image_path)
        if fundus.shape[:2] != manual.shape:
            raise RuntimeError(f"native alignment failed for {case_id}: fundus={fundus.shape[:2]}, mask={manual.shape}")
        fg = float(manual.mean())
        if not 0.001 <= fg <= 0.999:
            raise RuntimeError(f"foreground gate failed for {case_id}: {fg}")

        timings = {}
        arrays = {}
        for name in ("no_interp", "naive_interp", "ddfp", "seq_fp"):
            if name == "naive_interp" and not args.include_naive:
                continue
            started = time.perf_counter()
            arrays[name], _ = preps[name](manual.astype(np.float32), label=None, sample_id=None)
            timings[name] = round(time.perf_counter() - started, 6)
        target_shape = arrays["no_interp"].shape
        if any(arr.shape != target_shape for arr in arrays.values()):
            raise RuntimeError(f"expanded-grid alignment failed for {case_id}")

        fundus_exp = expanded_rgb(fundus, target_shape)
        seq_bin = arrays["seq_fp"] >= 0.5
        dd_bin = arrays["ddfp"] >= 0.5
        raw_bin = arrays["no_interp"] >= 0.5
        diff = (seq_bin != dd_bin).astype(np.uint8)
        repair = (raw_bin != dd_bin)
        if np.any(diff):
            roi_support, roi_basis = diff.astype(bool), "seq_vs_ddfp_difference"
        elif np.any(repair):
            roi_support, roi_basis = repair, "raw_vs_ddfp_difference"
        else:
            roi_support, roi_basis = dd_bin, "corrected_ddfp_support"
        roi = roi_from_support(
            roi_support, pad=args.roi_pad, min_size=args.roi_min_size,
            max_size=args.roi_max_size,
        )

        case_dir = out_root / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "fundus_native": save_array(case_dir / "fundus_native_rgb.npy", fundus),
            "fundus_expanded": save_array(case_dir / "fundus_expanded_rgb.npy", fundus_exp),
            "manual_mask_native": save_array(case_dir / "manual_vessel_mask_native.npy", manual),
            "raw_no_interp": save_array(case_dir / "raw_no_interp_expanded.npy", arrays["no_interp"].astype(np.float32)),
            "sequential_fp": save_array(case_dir / "sequential_fp.npy", arrays["seq_fp"].astype(np.float32)),
            "corrected_ddfp": save_array(case_dir / "corrected_ddfp.npy", arrays["ddfp"].astype(np.float32)),
            "seq_vs_ddfp_difference": save_array(case_dir / "seq_vs_ddfp_difference.npy", diff),
        }
        if "naive_interp" in arrays:
            outputs["naive_interp"] = save_array(case_dir / "naive_interp.npy", arrays["naive_interp"].astype(np.float32))
        if fov_path:
            fov = load_mask(fov_path).astype(np.uint8)
            if fov.shape != manual.shape:
                raise RuntimeError(f"FOV alignment failed for {case_id}")
            outputs["fov_mask_native"] = save_array(case_dir / "fov_mask_native.npy", fov)

        metrics = {
            "seq_vs_ddfp_differing_pixels_threshold_0_5": int(diff.sum()),
            "seq_vs_ddfp_exact_scalar_differing_pixels": int(np.count_nonzero(arrays["seq_fp"] != arrays["ddfp"])),
            "boundary_differing_pixels_threshold_0_5": boundary_difference(seq_bin, dd_bin),
            "raw_vs_ddfp_change_pixels_threshold_0_5": int(repair.sum()),
            "selection_metric": "seq_vs_ddfp_differing_pixels_threshold_0_5",
            "selection_score": int(diff.sum()),
        }
        provenance = {
            "schema": "ddfp-qualitative-drive-export-v1",
            "dataset": "DRIVE",
            "case_id": case_id,
            "source_paths": {
                "fundus": str(image_path.resolve()),
                "manual_vessel_mask": str(mask_path.resolve()),
                "fov_mask": str(fov_path.resolve()) if fov_path else None,
            },
            "input_sha256": {
                "fundus": sha256_file(image_path),
                "manual_vessel_mask": sha256_file(mask_path),
                "fov_mask": sha256_file(fov_path) if fov_path else None,
            },
            "native_shape_hw": list(manual.shape),
            "expanded_shape_hw": list(target_shape),
            "binary_threshold": 0.5,
            "fundus_expansion": "scipy.ndimage.zoom order=1, target=(2H-1,2W-1), uint8 rounded",
            "axis": None,
            "slice_index": None,
            "roi_y0_y1_x0_x1": roi,
            "roi_basis": roi_basis,
            "roi_rule": "support bbox + pad; if either side exceeds roi-max-size, max-support fixed window; ties smallest y then x",
            "metrics": metrics,
            "timing_seconds": timings,
            "backend": get_backend(),
            "r3_repo_root": str(repo),
            "r3_git_head": git_head(repo),
            "r3_source_sha256": r3_hashes,
            "generation_command": command_string(),
            "software_versions": software_versions(),
            "outputs": outputs,
            "interpretation_scope": "Empirical visual comparison only; no universal correctness or equivalence claim.",
        }
        meta_path = case_dir / "case_provenance.json"
        write_json(meta_path, provenance)
        write_sha256_manifest(case_dir)
        case_jsons.append(provenance)
        rows.append({
            "dataset": "DRIVE", "case_id": case_id,
            "manual_path": str(mask_path.resolve()), "fundus_path": str(image_path.resolve()),
            "score": metrics["selection_score"], "selection_metric": metrics["selection_metric"],
            "seq_ddfp_diff": metrics["seq_vs_ddfp_differing_pixels_threshold_0_5"],
            "boundary_diff": metrics["boundary_differing_pixels_threshold_0_5"],
            "raw_ddfp_change": metrics["raw_vs_ddfp_change_pixels_threshold_0_5"],
            "roi": json.dumps(roi, separators=(",", ":")),
            "case_provenance": str(meta_path.resolve()),
        })
        print(f"[PASS] {case_id}: shape={target_shape}, diff={int(diff.sum())}, ROI={roi}")

    if not rows:
        raise RuntimeError("no DRIVE cases matched")
    write_csv(out_root / "cases.csv", rows)
    write_json(out_root / "cases.json", case_jsons)
    write_sha256_manifest(out_root)
    print(f"Exported {len(rows)} DRIVE case(s) to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
