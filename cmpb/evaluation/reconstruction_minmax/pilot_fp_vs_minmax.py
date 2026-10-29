
import os
import json
import hashlib
from pathlib import Path

import numpy as np
import nibabel as nib
import cupy as cp

from src.ddfp.gpu_immersion import (
    build_ispan_gpu,
    front_propagation_gpu,
)

root = Path(os.environ["BRATS_ROOT"])
n = int(os.environ.get("PILOT_N", "5"))
paths = sorted(root.glob("BraTS2021_*/*_flair.nii.gz"))

assert len(paths) == 1251, f"Expected 1251, found {len(paths)}"
assert 1 <= n <= len(paths)

# A fixed permutation allows nested 5 -> 20 -> 1251 cohorts.
idx = np.random.default_rng(20261028).permutation(len(paths))[:n]
selected = [paths[int(i)] for i in idx]

output = Path("pilot_fp_minmax_results.jsonl")
completed = set()
if output.exists():
    for line in output.read_text().splitlines():
        if line.strip():
            completed.add(json.loads(line)["subject"])

for k, path in enumerate(selected, 1):
    subject = path.parent.name
    if subject in completed:
        continue

    raw = nib.load(str(path)).get_fdata(dtype=np.float32)

    # Fit normalization to observed slices only.
    observed = raw[:, :, ::2]
    positive = observed[observed > 0]
    p1, p99 = np.percentile(positive, [1, 99])

    scaled = np.zeros(raw.shape, dtype=np.uint8)
    valid = raw > 0
    scaled[valid] = (
        np.clip((raw[valid] - p1) / (p99 - p1 + 1e-9), 0, 1)
        * 255
    ).astype(np.uint8)

    src = np.ascontiguousarray(scaled[:, :, ::2])
    gt = scaled[:, :, 1::2].astype(np.float32)

    a = src[:, :, :-1].astype(np.float32)
    b = src[:, :, 1:].astype(np.float32)

    lo = np.minimum(a, b)
    hi = np.maximum(a, b)

    # Identical evaluation ROI for all methods.
    roi = (a > 0) | (b > 0)
    assert roi.any()

    U_lo, U_hi, level = build_ispan_gpu(src)
    u_gpu = front_propagation_gpu(
        U_lo, U_hi, level, verbose=False
    )

    # Even x/y positions, inserted odd z positions.
    fp = cp.asnumpy(
        u_gpu[1:-1, 1:-1, 1:-1][::2, ::2, 1::2]
    )

    assert fp.shape == gt.shape

    methods = {
        "fp": fp,
        "min": lo,
        "max": hi,
        "linear": (a + b) / 2,
    }

    result = {
        "subject": subject,
        "source_sha256": hashlib.sha256(
            path.read_bytes()
        ).hexdigest(),
        "roi_voxels": int(roi.sum()),
    }

    for name, prediction in methods.items():
        error = prediction[roi] - gt[roi]
        result[name] = {
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "bias": float(np.mean(error)),
        }

    result["fp_interior_fraction"] = float(
        np.mean((fp[roi] > lo[roi]) & (fp[roi] < hi[roi]))
    )

    with output.open("a") as f:
        f.write(json.dumps(result) + "\n")

    print(f"[{k}/{n}] {subject} "
          f"FP={result['fp']['mae']:.3f}, "
          f"Min={result['min']['mae']:.3f}, "
          f"Max={result['max']['mae']:.3f}, "
          f"Linear={result['linear']['mae']:.3f}",
          flush=True)

    del U_lo, U_hi, u_gpu, fp
    cp.get_default_memory_pool().free_all_blocks()

print("Saved:", output)

