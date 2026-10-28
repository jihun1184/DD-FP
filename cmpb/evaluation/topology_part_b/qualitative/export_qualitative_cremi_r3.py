#!/usr/bin/env python3
"""Export CREMI 3-D patch arrays with explicit padding/core-crop provenance."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
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
    "scripts/part_b/exp_b4_cremi_3d.py",
    "src/ddfp/cpu_fp.py",
    "src/ddfp/gpu_immersion.py",
    "src/ddfp/gpu_native_worklist.py",
    "src/ddfp/gpu_seed_aware_worklist.py",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--hdf5-dir", type=Path, default=None)
    p.add_argument("--hdf5-file", type=Path, action="append", default=[])
    p.add_argument("--n-volumes", type=int, default=3)
    p.add_argument("--n-patches", type=int, default=10)
    p.add_argument("--patch-size", type=int, nargs=3, default=(64, 128, 128), metavar=("D", "H", "W"))
    p.add_argument("--ddfp-pad", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--axis", type=int, choices=(0, 1, 2), default=0)
    p.add_argument("--roi-pad", type=int, default=20)
    p.add_argument("--roi-min-size", type=int, default=64)
    p.add_argument("--roi-max-size", type=int, default=192)
    p.add_argument("--raw-dataset", default="/volumes/raw")
    p.add_argument("--synthetic-only", action="store_true")
    p.add_argument("--case-id", action="append", default=[], help="exact generated patch name; repeatable")
    p.add_argument("--allow-existing", action="store_true")
    return p.parse_args()


def make_pad_plan(full_shape: tuple[int, int, int], coords: tuple[int, int, int], patch_size: tuple[int, int, int], delta: int) -> dict:
    starts = np.asarray(coords, dtype=int)
    sizes = np.asarray(patch_size, dtype=int)
    full = np.asarray(full_shape, dtype=int)
    padded_start = np.maximum(0, starts - delta)
    padded_end = np.minimum(full, starts + sizes + delta)
    low = starts - padded_start
    high = padded_end - (starts + sizes)
    expanded_start = 2 * low
    expanded_end = expanded_start + (2 * sizes - 1)
    return {
        "core_native_start_zyx": starts.tolist(),
        "core_native_end_zyx_exclusive": (starts + sizes).tolist(),
        "padded_native_start_zyx": padded_start.tolist(),
        "padded_native_end_zyx_exclusive": padded_end.tolist(),
        "requested_padding_voxels": int(delta),
        "actual_padding_low_zyx": low.tolist(),
        "actual_padding_high_zyx": high.tolist(),
        "expanded_core_crop_start_zyx": expanded_start.tolist(),
        "expanded_core_crop_end_zyx_exclusive": expanded_end.tolist(),
    }


def crop3(a: np.ndarray, start: list[int], end: list[int]) -> np.ndarray:
    return a[start[0]:end[0], start[1]:end[1], start[2]:end[2]]


def load_optional_raw(hdf_path: Path, dataset: str) -> np.ndarray | None:
    import h5py
    with h5py.File(hdf_path, "r") as f:
        key = dataset if dataset in f else dataset.lstrip("/")
        if key not in f:
            return None
        raw = f[key][:]
    if raw.ndim != 3:
        raise RuntimeError(f"raw EM dataset must be 3-D: {dataset} -> {raw.shape}")
    return raw


def expanded_scalar(a: np.ndarray, target: tuple[int, int, int], order: int) -> np.ndarray:
    factors = tuple(t / s for t, s in zip(target, a.shape))
    out = scipy_zoom(a.astype(np.float32), factors, order=order, prefilter=False)
    if out.shape != target:
        raise RuntimeError(f"scalar expansion mismatch: {out.shape} vs {target}")
    return out.astype(np.float32)


def select_slice_roi(primary: np.ndarray, secondary: np.ndarray, tertiary: np.ndarray, axis: int, pad: int, min_size: int, max_size: int):
    choices = (
        (primary, "seq_vs_ddfp_difference"),
        (secondary, "raw_vs_ddfp_difference"),
        (tertiary, "corrected_ddfp_support"),
    )
    reduce_axes = tuple(i for i in range(3) if i != axis)
    support, basis, scores = choices[-1][0], choices[-1][1], np.count_nonzero(choices[-1][0], axis=reduce_axes)
    for candidate, name in choices:
        candidate_scores = np.count_nonzero(candidate, axis=reduce_axes)
        if int(candidate_scores.max(initial=0)) > 0:
            support, basis, scores = candidate, name, candidate_scores
            break
    index = int(np.argmax(scores))
    support_2d = np.take(support, index, axis=axis)
    return index, roi_from_support(support_2d, pad=pad, min_size=min_size, max_size=max_size), basis, int(scores[index])


def export_case(
    *, args: argparse.Namespace, repo: Path, out_root: Path, r3_hashes: dict,
    case_name: str, source: str, membrane_full: np.ndarray, coords: tuple[int, int, int],
    patch_size: tuple[int, int, int], hdf_path: Path | None, raw_full: np.ndarray | None,
    no_interp_3d, seq_fp_3d, immersion_pipeline, backend: str,
) -> dict:
    pD, pH, pW = patch_size
    z0, y0, x0 = coords
    core = membrane_full[z0:z0+pD, y0:y0+pH, x0:x0+pW]
    if core.shape != patch_size:
        raise RuntimeError(f"core shape mismatch for {case_name}: {core.shape} vs {patch_size}")
    if core.sum() == 0:
        raise RuntimeError(f"empty membrane core for {case_name}")

    plan = make_pad_plan(membrane_full.shape, coords, patch_size, args.ddfp_pad)
    ps, pe = plan["padded_native_start_zyx"], plan["padded_native_end_zyx_exclusive"]
    padded = crop3(membrane_full, ps, pe)
    es, ee = plan["expanded_core_crop_start_zyx"], plan["expanded_core_crop_end_zyx_exclusive"]
    target = tuple(2 * s - 1 for s in patch_size)

    timings = {}
    started = time.perf_counter()
    raw_nointerp = no_interp_3d(core).astype(np.float32)
    timings["no_interp"] = round(time.perf_counter() - started, 6)

    started = time.perf_counter()
    seq_full = seq_fp_3d(padded)
    seq = crop3(seq_full, es, ee).astype(np.float32)
    timings["seq_fp"] = round(time.perf_counter() - started, 6)

    started = time.perf_counter()
    padded_whd = (padded.astype(np.uint8) * 255).transpose(2, 1, 0).copy()
    dd_full = immersion_pipeline(padded_whd, verbose=False).transpose(2, 1, 0).astype(np.float32)
    if dd_full.max(initial=0) > 1.0:
        dd_full /= 255.0
    dd = crop3(dd_full, es, ee).astype(np.float32)
    timings["ddfp"] = round(time.perf_counter() - started, 6)

    if raw_nointerp.shape != target or seq.shape != target or dd.shape != target:
        raise RuntimeError(f"expanded-grid gate failed for {case_name}: no={raw_nointerp.shape}, seq={seq.shape}, dd={dd.shape}, expected={target}")
    seq_bin, dd_bin, raw_bin = seq >= 0.5, dd >= 0.5, raw_nointerp >= 0.5
    diff = (seq_bin != dd_bin).astype(np.uint8)
    repair = raw_bin != dd_bin
    slice_idx, roi, roi_basis, slice_score = select_slice_roi(
        diff, repair, dd_bin, args.axis, args.roi_pad, args.roi_min_size,
        args.roi_max_size,
    )

    case_id = safe_case_id(case_name)
    case_dir = out_root / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "membrane_patch_native": save_array(case_dir / "membrane_patch_native.npy", core.astype(np.uint8)),
        "membrane_patch_padded_native": save_array(case_dir / "membrane_patch_padded_native.npy", padded.astype(np.uint8)),
        "raw_no_interp": save_array(case_dir / "raw_no_interp_expanded.npy", raw_nointerp),
        "sequential_fp": save_array(case_dir / "sequential_fp.npy", seq),
        "corrected_ddfp": save_array(case_dir / "corrected_ddfp.npy", dd),
        "seq_vs_ddfp_difference": save_array(case_dir / "seq_vs_ddfp_difference.npy", diff),
    }
    raw_info = None
    if raw_full is not None:
        if raw_full.shape != membrane_full.shape:
            raise RuntimeError(f"raw EM/label shape mismatch: {raw_full.shape} vs {membrane_full.shape}")
        raw_core = raw_full[z0:z0+pD, y0:y0+pH, x0:x0+pW]
        raw_exp = expanded_scalar(raw_core, target, order=1)
        outputs["raw_em_patch_native"] = save_array(case_dir / "raw_em_patch_native.npy", raw_core)
        outputs["raw_em_patch_expanded"] = save_array(case_dir / "raw_em_patch_expanded.npy", raw_exp)
        raw_info = {"hdf5_dataset": args.raw_dataset, "shape": list(raw_full.shape)}

    metrics = {
        "seq_vs_ddfp_differing_voxels_threshold_0_5": int(diff.sum()),
        "seq_vs_ddfp_exact_scalar_differing_voxels": int(np.count_nonzero(seq != dd)),
        "boundary_differing_voxels_threshold_0_5": boundary_difference(seq_bin, dd_bin),
        "raw_vs_ddfp_change_voxels_threshold_0_5": int(repair.sum()),
        "selection_metric": "seq_vs_ddfp_differing_voxels_threshold_0_5",
        "selection_score": int(diff.sum()),
    }
    provenance = {
        "schema": "ddfp-qualitative-cremi-export-v1",
        "dataset": "CREMI" if hdf_path else "CREMI-synthetic-fixture",
        "case_id": case_id,
        "sample_hdf5_id": hdf_path.stem if hdf_path else source,
        "source": source,
        "source_paths": {"hdf5": str(hdf_path.resolve()) if hdf_path else None},
        "input_sha256": {"hdf5": sha256_file(hdf_path) if hdf_path else None},
        "labels_dataset": "/volumes/labels/neuron_ids" if hdf_path else None,
        "raw_em": raw_info,
        "full_volume_shape_zyx": list(membrane_full.shape),
        "patch_size_zyx": list(patch_size),
        "patch_coordinates_z0_y0_x0": list(coords),
        "padding_and_crop": plan,
        "expanded_core_shape_zyx": list(target),
        "binary_threshold": 0.5,
        "axis": args.axis,
        "slice_index": slice_idx,
        "roi_y0_y1_x0_x1": roi,
        "roi_basis": roi_basis,
        "roi_rule": "max-burden slice; support bbox + pad; oversized bbox replaced by max-support fixed window; ties lowest indices",
        "slice_basis_score": slice_score,
        "selection_seed": args.seed,
        "patch_selection_rule": "R3 exp_b4.extract_subvolumes; RNG default_rng(seed); accepted in draw order at membrane fraction 0.01..0.30",
        "metrics": metrics,
        "timing_seconds": timings,
        "backend": backend,
        "r3_repo_root": str(repo),
        "r3_git_head": git_head(repo),
        "r3_source_sha256": r3_hashes,
        "generation_command": command_string(),
        "software_versions": software_versions(),
        "outputs": outputs,
        "interpretation_scope": "Empirical visual comparison and patch-local diagnostic only; no universal correctness, equality, convergence, or topology-proof claim.",
    }
    meta_path = case_dir / "case_provenance.json"
    write_json(meta_path, provenance)
    write_sha256_manifest(case_dir)
    print(f"[PASS] {case_id}: core={patch_size}, pad={plan['actual_padding_low_zyx']}/{plan['actual_padding_high_zyx']}, diff={int(diff.sum())}, slice={slice_idx}, ROI={roi}")
    return provenance


def main() -> int:
    args = parse_args()
    if args.ddfp_pad < 0 or args.n_patches < 1 or any(s < 2 for s in args.patch_size):
        raise ValueError("padding must be nonnegative; n-patches >=1; patch dimensions >=2")
    repo = args.repo_root.resolve()
    out_root = args.output_root.resolve()
    ensure_new_or_empty(out_root, args.allow_existing)
    r3_hashes = verify_repo(repo, REQUIRED_R3)
    add_repo_to_path(repo)
    from scripts.part_b.exp_b4_cremi_3d import (
        extract_subvolumes, load_cremi_membrane_3d, make_synthetic_membrane_3d,
        no_interp_3d, seq_fp_3d,
    )
    from src.ddfp import get_backend, immersion_pipeline

    jobs = []
    if args.synthetic_only:
        for name, vol in make_synthetic_membrane_3d():
            if args.case_id and name not in args.case_id:
                continue
            jobs.append((name, "synthetic", vol, (0, 0, 0), tuple(vol.shape), None, None))
    else:
        files = [p.resolve() for p in args.hdf5_file]
        if args.hdf5_dir:
            root = args.hdf5_dir.resolve()
            files.extend(sorted(root.glob("*.hdf")) + sorted(root.glob("*.h5")))
        files = sorted(dict.fromkeys(files))[:args.n_volumes]
        if not files:
            raise FileNotFoundError("no CREMI HDF5 files supplied")
        for hdf_path in files:
            membrane = load_cremi_membrane_3d(hdf_path)
            raw_full = load_optional_raw(hdf_path, args.raw_dataset)
            patches = extract_subvolumes(membrane, args.n_patches, tuple(args.patch_size), seed=args.seed)
            if len(patches) < args.n_patches:
                raise RuntimeError(f"only {len(patches)}/{args.n_patches} eligible patches found in {hdf_path}")
            for name, coords in patches:
                if args.case_id and name not in args.case_id:
                    continue
                jobs.append((name, f"cremi_{hdf_path.stem}", membrane, coords, tuple(args.patch_size), hdf_path, raw_full))

    results = []
    for name, source, full, coords, size, hdf_path, raw_full in jobs:
        results.append(export_case(
            args=args, repo=repo, out_root=out_root, r3_hashes=r3_hashes,
            case_name=name, source=source, membrane_full=full, coords=coords,
            patch_size=size, hdf_path=hdf_path, raw_full=raw_full,
            no_interp_3d=no_interp_3d, seq_fp_3d=seq_fp_3d,
            immersion_pipeline=immersion_pipeline, backend=get_backend(),
        ))
    if not results:
        raise RuntimeError("no CREMI cases matched")
    rows = []
    for p in results:
        rows.append({
            "dataset": p["dataset"], "case_id": p["case_id"], "sample_hdf5_id": p["sample_hdf5_id"],
            "z0": p["patch_coordinates_z0_y0_x0"][0], "y0": p["patch_coordinates_z0_y0_x0"][1], "x0": p["patch_coordinates_z0_y0_x0"][2],
            "patch_size": json.dumps(p["patch_size_zyx"], separators=(",", ":")),
            "score": p["metrics"]["selection_score"], "selection_metric": p["metrics"]["selection_metric"],
            "seq_ddfp_diff": p["metrics"]["seq_vs_ddfp_differing_voxels_threshold_0_5"],
            "boundary_diff": p["metrics"]["boundary_differing_voxels_threshold_0_5"],
            "raw_ddfp_change": p["metrics"]["raw_vs_ddfp_change_voxels_threshold_0_5"],
            "axis": p["axis"], "slice_index": p["slice_index"], "roi": json.dumps(p["roi_y0_y1_x0_x1"], separators=(",", ":")),
            "case_provenance": str((out_root / "cases" / p["case_id"] / "case_provenance.json").resolve()),
        })
    write_csv(out_root / "cases.csv", rows)
    write_json(out_root / "cases.json", results)
    write_sha256_manifest(out_root)
    print(f"Exported {len(results)} CREMI patch(es) to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
