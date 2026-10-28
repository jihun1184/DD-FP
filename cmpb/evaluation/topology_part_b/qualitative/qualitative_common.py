#!/usr/bin/env python3
"""Shared deterministic/provenance helpers for qualitative exporters."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shlex
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# Submitted repository after applying the R3 correction kit.
EXPECTED_R3_HASHES = {
    "scripts/part_b/exp_b1_topology_accuracy.py": "888633e4e9cd5452dfd71003cfe600d4eb161103517db495e62fb924c8f1227d",
    "scripts/part_b/exp_b4_cremi_3d.py": "ecc741374c5059b72f45f8cd2f16eb712af9582408ccd5799803f71f1cf8eb68",
    "src/ddfp/cpu_fp.py": "8eae554f7d3d688dee2285f41d20bfbf6e34af98016a3f1ad07f1ff5f8f36a88",
    "src/ddfp/gpu_immersion.py": "b784a376d3466a1d4ded5089a2eee9f27df35006b4926301dd06106b4ed262a4",
    "src/ddfp/gpu_native_worklist.py": "853bf33a6bbd9c91f5304a018f0a51ff53e8418360fe1477208cf81df17163ac",
    "src/ddfp/gpu_seed_aware_worklist.py": "c80f56b9ba7afd63196332d51894168d36f6e9f3e2a51b6b966786311e4eecad",
    "test/test_smoke.py": "254927679ef43bf0cb4befe83f150a614a6b03a9faaeb5e7dc24900cde1735b0",
    "scripts/ddfp/experiment_DDFP_all.py": "b6a9b109f4105e7b77ebed880f669ba670bbd87123deca9509ba5b00993e561e",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def verify_repo(repo_root: Path, required: Iterable[str]) -> dict[str, str]:
    repo_root = repo_root.resolve()
    observed: dict[str, str] = {}
    mismatches = []
    for rel in required:
        path = repo_root / rel
        if not path.is_file():
            mismatches.append(f"missing: {rel}")
            continue
        digest = sha256_file(path)
        observed[rel] = digest
        expected = EXPECTED_R3_HASHES[rel]
        if digest != expected:
            mismatches.append(f"hash mismatch: {rel}: expected {expected}, got {digest}")
    if mismatches:
        raise RuntimeError("R3 source gate failed:\n  " + "\n  ".join(mismatches))
    return observed


def add_repo_to_path(repo_root: Path) -> None:
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def safe_case_id(value: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)
    if not out or out in {".", ".."}:
        raise ValueError(f"invalid case id: {value!r}")
    return out


def software_versions(extra: dict[str, str] | None = None) -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except Exception:
        versions["scipy"] = "unavailable"
    try:
        from PIL import __version__ as pillow_version
        versions["pillow"] = pillow_version
    except Exception:
        versions["pillow"] = "unavailable"
    try:
        import h5py
        versions["h5py"] = h5py.__version__
    except Exception:
        versions["h5py"] = "unavailable"
    if extra:
        versions.update(extra)
    return versions


def command_string(argv: list[str] | None = None) -> str:
    values = [sys.executable, *(sys.argv if argv is None else argv)]
    return shlex.join(values)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_array(path: Path, array: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.min(array)) if array.size else None,
        "max": float(np.max(array)) if array.size else None,
    }


def boundary_difference(a: np.ndarray, b: np.ndarray) -> int:
    from scipy.ndimage import binary_erosion, generate_binary_structure

    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    structure = generate_binary_structure(a.ndim, 1)
    ba = a ^ binary_erosion(a, structure=structure, border_value=0)
    bb = b ^ binary_erosion(b, structure=structure, border_value=0)
    return int(np.count_nonzero((ba | bb) & (a != b)))


def roi_from_support(
    support: np.ndarray,
    pad: int = 20,
    min_size: int = 64,
    max_size: int | None = 192,
) -> list[int]:
    """Return deterministic [y0,y1,x0,x1] ROI for a 2-D support."""
    if support.ndim != 2:
        raise ValueError(f"ROI support must be 2-D, got {support.shape}")
    h, w = support.shape
    ys, xs = np.where(support)
    if len(ys):
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
    else:
        y0, y1, x0, x1 = h // 2, h // 2 + 1, w // 2, w // 2 + 1
    y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
    x0, x1 = max(0, x0 - pad), min(w, x1 + pad)

    # Dispersed supports (e.g. a retinal vessel network) can make a global
    # bounding box nearly full-frame. In that case choose the fixed-size window
    # with maximum support count. np.argmax resolves ties in row-major order,
    # hence smallest y then x.
    if max_size is not None and ((y1 - y0) > max_size or (x1 - x0) > max_size):
        wh, ww = min(max_size, h), min(max_size, w)
        sat = np.pad(np.asarray(support, dtype=np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        scores = sat[wh:, ww:] - sat[:-wh, ww:] - sat[wh:, :-ww] + sat[:-wh, :-ww]
        top, left = np.unravel_index(int(np.argmax(scores)), scores.shape)
        return [int(top), int(top + wh), int(left), int(left + ww)]

    def grow(lo: int, hi: int, limit: int) -> tuple[int, int]:
        missing = max(0, min(min_size, limit) - (hi - lo))
        lo -= missing // 2
        hi += missing - missing // 2
        if lo < 0:
            hi = min(limit, hi - lo)
            lo = 0
        if hi > limit:
            lo = max(0, lo - (hi - limit))
            hi = limit
        return lo, hi

    y0, y1 = grow(y0, y1, h)
    x0, x1 = grow(x0, x1, w)
    return [int(y0), int(y1), int(x0), int(x1)]


def deterministic_slice_and_roi(
    primary_support: np.ndarray,
    fallback_support: np.ndarray,
    axis: int,
    pad: int = 20,
    min_size: int = 64,
    max_size: int | None = 192,
) -> tuple[int, list[int], str, int]:
    """Pick max-burden slice; ties resolve to the lowest index."""
    if primary_support.shape != fallback_support.shape or primary_support.ndim != 3:
        raise ValueError("3-D support arrays must have identical shapes")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    reduce_axes = tuple(i for i in range(3) if i != axis)
    primary_scores = np.count_nonzero(primary_support, axis=reduce_axes)
    if int(primary_scores.max(initial=0)) > 0:
        support, basis, scores = primary_support, "seq_vs_ddfp_difference", primary_scores
    else:
        scores = np.count_nonzero(fallback_support, axis=reduce_axes)
        support, basis = fallback_support, "raw_vs_ddfp_difference"
        if int(scores.max(initial=0)) == 0:
            support = np.asarray(fallback_support, dtype=bool)
            basis = "fallback_support"
    idx = int(np.argmax(scores))
    sl = np.take(support, idx, axis=axis)
    return idx, roi_from_support(sl, pad=pad, min_size=min_size, max_size=max_size), basis, int(scores[idx])


def write_sha256_manifest(root: Path, output_name: str = "SHA256SUMS.txt") -> Path:
    root = root.resolve()
    target = root / output_name
    lines = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p != target):
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target


def git_head(repo_root: Path) -> str:
    head = repo_root / ".git" / "HEAD"
    if not head.is_file():
        return "unavailable"
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = repo_root / ".git" / value[5:]
        if ref.is_file():
            return ref.read_text(encoding="utf-8").strip()
    return value


def ensure_new_or_empty(path: Path, allow_existing: bool) -> None:
    if path.exists() and any(path.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"output directory is not empty: {path}; use --allow-existing to overwrite named case files"
        )
    path.mkdir(parents=True, exist_ok=True)
