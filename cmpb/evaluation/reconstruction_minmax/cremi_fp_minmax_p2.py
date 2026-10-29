#!/usr/bin/env python3
"""CREMI held-out z-slice FP/Min/Max/Linear comparison (P2).

RUN ON A CUDA GPU SERVER with the DD-FP v1.0.0 repository on PYTHONPATH.
Uses original CREMI training HDF5 volume/raw (z,y,x), no labels/GT segmentation.
Protocol: 3 fixed non-overlapping 128x128x33 ROIs per volume (A/B/C),
observed even z indices, held-out odd z indices, all 128x128x16 voxels
scored in native uint8 intensity, no percentile normalization.
This is a reconstruction experiment, NOT a full-grid DWC test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import cupy as cp
from src.ddfp.gpu_immersion import build_ispan_gpu, front_propagation_gpu

# (z, y, x). Fully fixed before running P2 results. All fits in 125x1250x1250.
ROIS = {
    "R1": (8, 160, 160),
    "R2": (48, 512, 512),
    "R3": (84, 896, 896),
}
CROP_ZYX = (33, 128, 128)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(pred, gt):
    # This experiment evaluates all positions (CREMI has no MRI-like air mask).
    err = pred.astype(np.float32) - gt.astype(np.float32)
    return {
        "mae": float(np.mean(np.abs(err), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(err.astype(np.float64)**2))),
        "bias": float(np.mean(err, dtype=np.float64)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cremi-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--samples", default="A,B,C", help="Comma-separated A,B,C")
    ap.add_argument("--max-rois", type=int, choices=(1, 2, 3), default=3)
    args = ap.parse_args()
    samples = [x.strip().upper() for x in args.samples.split(",")]
    if not samples or len(samples) != len(set(samples)) or any(x not in "ABC" for x in samples):
        ap.error("--samples must contain unique letters A, B, C")
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("CUDA GPU not found")
    if not args.cremi_root.is_dir():
        raise FileNotFoundError(args.cremi_root)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "cremi_p2_results.jsonl"
    completed = set()
    if output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                completed.add((r["sample"], r["roi"]))

    for sample in samples:
        path = args.cremi_root / f"sample_{sample}_20160501.hdf"
        if not path.is_file():
            alt = args.cremi_root / f"sample_{sample}.hdf"
            if alt.is_file():
                path = alt
            else:
                raise FileNotFoundError(f"Missing CREMI training file: {path}")
        file_sha = sha256(path)
        with h5py.File(path, "r") as h5:
            ds = h5["volumes/raw"]
            if ds.ndim != 3 or ds.dtype != np.uint8:
                raise ValueError(f"Unexpected CREMI raw array: {ds.shape}, {ds.dtype}")
            for roi_name, (z0, y0, x0) in list(ROIS.items())[:args.max_rois]:
                if (sample, roi_name) in completed:
                    print(f"SKIP {sample} {roi_name} (already recorded)", flush=True)
                    continue
                dz, dy, dx = CROP_ZYX
                if z0 + dz > ds.shape[0] or y0 + dy > ds.shape[1] or x0 + dx > ds.shape[2]:
                    raise ValueError(f"ROI out of range: {sample} {roi_name} {ds.shape}")
                crop_zyx = ds[z0:z0+dz, y0:y0+dy, x0:x0+dx]
                # P1 GPU code expects (x, y, z), not HDF5 (z, y, x).
                raw_xyz = np.ascontiguousarray(crop_zyx.transpose(2, 1, 0))
                # 33 z-slices: even 0,2,...,32 -> 17 observed;
                # odd 1,3,...,31 -> 16 held out (exactly 16 neighbor pairs).
                src = np.ascontiguousarray(raw_xyz[:, :, ::2])
                gt = raw_xyz[:, :, 1::2]
                a = src[:, :, :-1].astype(np.float32)
                b = src[:, :, 1:].astype(np.float32)
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                U_lo, U_hi, level = build_ispan_gpu(src)
                u_gpu = front_propagation_gpu(U_lo, U_hi, level, verbose=False)
                fp = cp.asnumpy(u_gpu[1:-1, 1:-1, 1:-1][::2, ::2, 1::2])
                if fp.shape != gt.shape or fp.shape != lo.shape:
                    raise RuntimeError(f"Shape mismatch fp {fp.shape} gt {gt.shape} lo {lo.shape}")
                record = {
                    "dataset": "CREMI", "sample": sample, "roi": roi_name,
                    "file": path.name, "source_sha256": file_sha,
                    "crop_origin_zyx": [z0, y0, x0],
                    "crop_size_zyx": list(CROP_ZYX),
                    "evaluation": "held_out_odd_z", "axis_input": "xyz",
                    "intensity": "native_uint8_no_normalization",
                    "mask": "all_voxels", "roi_voxels": int(gt.size),
                    "fp": metrics(fp, gt), "min": metrics(lo, gt),
                    "max": metrics(hi, gt),
                    "linear": metrics((a+b)/2.0, gt),
                    "fp_interior_fraction": float(np.mean((fp > lo) & (fp < hi))),
                    "fp_below_min_fraction": float(np.mean(fp < lo)),
                    "fp_above_max_fraction": float(np.mean(fp > hi)),
                }
                with output.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                print(f"{sample}/{roi_name}: FP={record['fp']['mae']:.4f}, "
                      f"Min={record['min']['mae']:.4f}, Max={record['max']['mae']:.4f}, "
                      f"Linear={record['linear']['mae']:.4f}", flush=True)
                del crop_zyx, raw_xyz, src, gt, a, b, lo, hi, U_lo, U_hi, level, u_gpu, fp
                cp.get_default_memory_pool().free_all_blocks()
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    main()
