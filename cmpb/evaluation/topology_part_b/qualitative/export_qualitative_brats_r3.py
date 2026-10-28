#!/usr/bin/env python3
"""Export one BraTS case for DD-FP qualitative-response reserve.

Run this only after DD_FP_GITHUB_APPLICATION_KIT_R3_20260906 has been
successfully applied to an exact 8fdfe623... clone and its release gate passes.

Outputs a common expanded-grid bundle:
  source_display_grid.npy       trilinear anatomical display image
  raw_or_predwc.npy             nearest-neighbour no-interpolation baseline
  sequential_fp.npy             measured CPU sequential FP
  corrected_production_ddfp.npy measured corrected domain-decomposed DD-FP/IBI
  owner_map.npy                 deterministic final production owner map
  source_resampling.json
  case_provenance.json
  source_native.nii[.gz]        immutable copied source

The script is an export/visualization helper only. It does not assert numerical
identity, convergence, or universal topology preservation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

BASE_COMMIT = "8fdfe6233dfe53f996ed2f09e8810bca53d5323b"
# Final R3 working tree hashes after step [5/6] documentation cleanup.
EXPECTED_ACTIVE_HASHES = {
    "src/ddfp/cpu_fp.py": "8eae554f7d3d688dee2285f41d20bfbf6e34af98016a3f1ad07f1ff5f8f36a88",
    "src/ddfp/gpu_immersion.py": "b784a376d3466a1d4ded5089a2eee9f27df35006b4926301dd06106b4ed262a4",
    "src/ddfp/gpu_native_worklist.py": "853bf33a6bbd9c91f5304a018f0a51ff53e8418360fe1477208cf81df17163ac",
    "src/ddfp/gpu_seed_aware_worklist.py": "c80f56b9ba7afd63196332d51894168d36f6e9f3e2a51b6b966786311e4eecad",
    "scripts/ddfp/experiment_DDFP_all.py": "4e52fd1499c54045c99bdb89af4658b82a0f3e143b3df105f48f53f4e6fe7f8e",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def verify_repo(repo: Path) -> dict[str, str]:
    if not (repo / ".git").is_dir():
        raise SystemExit(f"Not a git repository: {repo}")
    head = git(repo, "rev-parse", "HEAD")
    if head != BASE_COMMIT:
        raise SystemExit(f"HEAD mismatch: expected {BASE_COMMIT}, got {head}")
    hashes: dict[str, str] = {}
    for rel, expected in EXPECTED_ACTIVE_HASHES.items():
        p = repo / rel
        if not p.is_file():
            raise SystemExit(f"Missing corrected source: {rel}")
        got = sha256(p)
        if got != expected:
            raise SystemExit(f"R3 source hash mismatch for {rel}: {got} != {expected}")
        hashes[rel] = got
    return hashes


def build_owner_map_dhw(shape_dhw: tuple[int, int, int], K: int, split_domain_1d) -> np.ndarray:
    D, H, W = shape_dhw
    D2, H2, W2 = 2 * D - 1, 2 * H - 1, 2 * W - 1
    owner_z = np.full(D2, -1, dtype=np.int16)
    slices = split_domain_1d(D, K)
    # Mirror run_ibi_v10's sequential assembly/write order. The next tile's
    # interior write overwrites layer 2*z_b, leaving the final internal-boundary
    # pair owner(2*z_b-1)=k, owner(2*z_b)=k+1.
    for k, (z0, z1) in enumerate(slices):
        gz_s = 2 * z0 if k > 0 else 0
        gz_e = 2 * z1 - 1 if k < K - 1 else D2
        owner_z[gz_s:gz_e] = k
        if k < K - 1:
            for gz in (2 * z1 - 1, 2 * z1):
                if 0 <= gz < D2:
                    owner_z[gz] = k
    if np.any(owner_z < 0):
        raise RuntimeError("owner map has unassigned expanded-z layers")
    return np.broadcast_to(owner_z[:, None, None], (D2, H2, W2)).copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path, help="R3-applied DD-FP repo")
    ap.add_argument("--flair", required=True, type=Path, help="one BraTS FLAIR NIfTI")
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--out-root", required=True, type=Path)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--delta", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    repo = args.repo.resolve()
    flair = args.flair.resolve()
    if not flair.is_file():
        raise SystemExit(f"Missing FLAIR: {flair}")
    active_hashes = verify_repo(repo)

    sys.path.insert(0, str(repo))
    from scripts.part_b.exp_b3_brats_3d_all import (  # type: ignore
        load_brats_flair,
        naive_interp_3d,
        no_interp_3d,
        seq_fp_3d,
    )
    from scripts.ddfp.experiment_DDFP_all import (  # type: ignore
        count_boundary_violations,
        run_ibi_v10,
        split_domain_1d,
    )

    out = args.out_root.resolve() / args.case_id
    out.mkdir(parents=True, exist_ok=False)

    # Existing Part-B BraTS semantics: native FLAIR -> min-max [0,1], then
    # fixed threshold 0.5 for the binary topology/FP comparison.
    source_native_dhw = load_brats_flair(flair).astype(np.float32, copy=False)
    ref_bin_dhw = (source_native_dhw >= args.threshold).astype(np.uint8)
    if not np.any(ref_bin_dhw):
        raise SystemExit(f"Empty thresholded case at threshold={args.threshold}")

    # Common expanded display/comparator grid.
    source_display = naive_interp_3d(source_native_dhw).astype(np.float32, copy=False)
    raw_predwc = no_interp_3d(source_native_dhw).astype(np.float32, copy=False)

    print(f"[{args.case_id}] sequential FP ...", flush=True)
    t0 = time.perf_counter()
    seq_dhw = seq_fp_3d(ref_bin_dhw).astype(np.float32, copy=False)
    t_seq = time.perf_counter() - t0
    print(f"  done: {t_seq:.3f}s, shape={seq_dhw.shape}", flush=True)

    print(f"[{args.case_id}] corrected DD-FP K={args.K} delta={args.delta} ...", flush=True)
    # run_ibi_v10 convention is (W,H,D), uint8 [0,255].
    ref_u8_whd = (ref_bin_dhw.transpose(2, 1, 0) * 255).astype(np.uint8, copy=False)
    t0 = time.perf_counter()
    dd = run_ibi_v10(
        ref_u8_whd,
        K=args.K,
        delta=args.delta,
        max_rounds=args.max_rounds,
        verbose=False,
    )
    t_dd = time.perf_counter() - t0
    ddfp_dhw = np.asarray(dd["u_dd"], dtype=np.float32).transpose(2, 1, 0)
    if ddfp_dhw.max(initial=0) > 1.0:
        ddfp_dhw /= 255.0
    boundary_viol = count_boundary_violations(dd["u_dd"], dd["boundary_z_orig"])
    print(f"  done: {t_dd:.3f}s, R*={dd['R_star']}, boundary_viol={boundary_viol}", flush=True)

    shapes = {tuple(a.shape) for a in (source_display, raw_predwc, seq_dhw, ddfp_dhw)}
    if len(shapes) != 1:
        raise RuntimeError(f"common-grid shape mismatch: {shapes}")

    owner = build_owner_map_dhw(tuple(ref_bin_dhw.shape), args.K, split_domain_1d)
    if owner.shape != seq_dhw.shape:
        raise RuntimeError(f"owner shape mismatch: {owner.shape} != {seq_dhw.shape}")

    outputs = {
        "source_display_grid.npy": source_display,
        "raw_or_predwc.npy": raw_predwc,
        "sequential_fp.npy": seq_dhw,
        "corrected_production_ddfp.npy": ddfp_dhw,
        "owner_map.npy": owner,
    }
    for name, array in outputs.items():
        np.save(out / name, array, allow_pickle=False)

    native_name = "source_native.nii.gz" if flair.name.lower().endswith(".nii.gz") else "source_native.nii"
    shutil.copy2(flair, out / native_name)

    resampling = {
        "source_native": native_name,
        "source_display_grid": "source_display_grid.npy",
        "source_display_rule": "scripts.part_b.exp_b3_brats_3d_all.naive_interp_3d; scipy.ndimage.zoom order=1, prefilter=False",
        "raw_or_predwc_rule": "scripts.part_b.exp_b3_brats_3d_all.no_interp_3d; scipy.ndimage.zoom order=0, prefilter=False",
        "native_normalization": "load_brats_flair min-max normalization to [0,1]",
        "threshold": args.threshold,
        "axis_convention": "D,H,W for exported arrays; reserve axis=0/1/2 may be selected later, default requested BraTS axis=2 only after orientation validation",
    }
    (out / "source_resampling.json").write_text(json.dumps(resampling, indent=2), encoding="utf-8")

    file_hashes = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    provenance = {
        "dataset": "BraTS2021",
        "case_id": args.case_id,
        "source_run_id": args.run_id,
        "repo_head": git(repo, "rev-parse", "HEAD"),
        "repo_branch": git(repo, "branch", "--show-current"),
        "r3_active_source_hashes": active_hashes,
        "input_flair_original_path": str(flair),
        "input_flair_sha256": sha256(flair),
        "array_semantics": {
            "source_display_grid.npy": "trilinear display-grid anatomical FLAIR, normalized [0,1]",
            "raw_or_predwc.npy": "nearest-neighbour no-interpolation baseline on expanded grid, normalized [0,1]",
            "sequential_fp.npy": "measured CPU sequential FP on thresholded binary source, normalized [0,1]",
            "corrected_production_ddfp.npy": "measured corrected R3 domain-decomposed DD-FP/IBI output, normalized [0,1]",
            "owner_map.npy": "deterministic final production tile owner index per expanded-grid voxel",
        },
        "foreground_threshold": args.threshold,
        "K": args.K,
        "delta": args.delta,
        "max_rounds": args.max_rounds,
        "R_star": int(dd["R_star"]),
        "boundary_violations": int(boundary_viol),
        "timing_operational_only_s": {"sequential_fp": t_seq, "ddfp": t_dd},
        "export_shape_dhw": list(seq_dhw.shape),
        "file_sha256": file_hashes,
        "interpretation_boundary": "Qualitative empirical reserve only; no universal equality, fixed-point, convergence, assembly, or topology-preservation claim.",
    }
    (out / "case_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    # Recompute final hashes including provenance/resampling/native copy.
    final_hashes = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "SHA256SUMS.txt").write_text(
        "".join(f"{h}  {name}\n" for name, h in sorted(final_hashes.items())),
        encoding="utf-8",
    )
    print(f"EXPORT_OK {out}")


if __name__ == "__main__":
    main()
