"""
scripts/prepare_cremi.py
CREMI HDF5 volume -> 2D PNG slice conversion.

CREMI neuron_ids semantics:
  - neuron_ids == 0  : neuron membrane / boundary (the thin lines between cells)
  - neuron_ids  > 0  : neuron cell body interior (fills ~95% of area)

Segmentation target for topology-preserving tasks (DD-FP paper):
  Membrane detection -- thin tubular structures, topologically rich.
  mask = (neuron_ids == 0)  =>  white=membrane, black=cell interior

  Using (neuron_ids > 0) is WRONG: it produces ~95% white masks that
  look all-white and carry no topological signal for thin-structure tasks.

Usage:
    # Manual download then convert
    python scripts/prepare_cremi.py \\
        --hdf5-dir data/CREMI/raw/ \\
        --output-dir data/CREMI/ \\
        --max-slices 50

    # Auto-download + convert
    python scripts/prepare_cremi.py --download --output-dir data/CREMI/

Download URLs:
    https://cremi.org/static/data/sample_A_20160501.hdf
    https://cremi.org/static/data/sample_B_20160501.hdf
    https://cremi.org/static/data/sample_C_20160501.hdf
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image


CREMI_URLS = {
    "A": "https://cremi.org/static/data/sample_A_20160501.hdf",
    "B": "https://cremi.org/static/data/sample_B_20160501.hdf",
    "C": "https://cremi.org/static/data/sample_C_20160501.hdf",
}


def _check_h5py() -> None:
    try:
        import h5py  # noqa: F401
    except ImportError:
        print("[ERROR] h5py not installed. Run: pip install h5py")
        sys.exit(1)


def download_cremi(output_dir: Path) -> None:
    """Download three CREMI HDF5 files."""
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for vol, url in CREMI_URLS.items():
        dest = raw_dir / f"sample_{vol}.hdf"
        if dest.exists():
            print(f"[SKIP] already exists: {dest}")
            continue
        print(f"[DL]  {vol}: {url}")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"[WARN] download failed ({vol}): {e}")
            print(f"       Please download manually from https://cremi.org"
                  f" and place in '{raw_dir}/'.")


def convert_volume(
    hdf_path: Path,
    vol_name: str,
    img_dir:  Path,
    lbl_dir:  Path,
    max_slices: int | None = None,
    patch_size: int | None = 256,
    patch_stride: int | None = None,
) -> int:
    """
    Convert a single HDF5 volume to per-slice/patch PNGs.

    HDF5 layout (CREMI standard):
        /volumes/raw                [Z, Y, X] uint8
        /volumes/labels/neuron_ids  [Z, Y, X] uint64

    Mask convention (FIXED):
        neuron_ids == 0  =>  membrane / boundary  =>  WHITE (foreground)
        neuron_ids  > 0  =>  cell body interior   =>  BLACK (background)

        Rationale: the segmentation target for topology-preserving tasks is
        the thin membrane network between neurons (tubular, topologically rich).
        Using (neuron_ids > 0) was WRONG -- it labels ~95% of pixels as
        foreground, producing all-white masks with no topological structure.

    Patch size options (recommended: 256 -- Option A):
        256x256 -> DD-FP output 511x511, V100 batch=8 feasible  <- recommended
        512x512 -> DD-FP output 1023x1023, batch=2~4
        None    -> full 1250x1250 slice (memory-intensive)

    Patch count for 1250x1250 slice with 256x256 non-overlapping patches:
        floor(1250/256) x floor(1250/256) = 4x4 = 16 patches/slice
        3 volumes x 125 slices x 16 patches = 6,000 patches total

    Returns
    -------
    Number of patches (or slices) saved.
    """
    import h5py

    if not hdf_path.exists():
        print(f"[WARN] HDF5 not found, skipping: {hdf_path}")
        return 0

    with h5py.File(hdf_path, "r") as f:
        raw    = f["/volumes/raw"][:]
        labels = f["/volumes/labels/neuron_ids"][:]

    n_slices = raw.shape[0]
    if max_slices is not None:
        n_slices = min(n_slices, max_slices)

    H, W   = raw.shape[1], raw.shape[2]
    stride = patch_stride or patch_size  # default: non-overlapping

    count = 0
    for z in range(n_slices):
        img_arr = raw[z].astype(np.uint8)

        # Boundary extraction from dense instance segmentation map.
        #
        # CREMI neuron_ids is a DENSE label: every pixel belongs to a neuron (id > 0).
        # There are virtually no id==0 pixels, so (labels==0) gives all-black,
        # and (labels>0) gives all-white. Neither is useful.
        #
        # Correct approach: membrane = pixels where a horizontal or vertical
        # neighbor has a DIFFERENT neuron ID. This yields the thin boundary
        # network between cells (~5-15% foreground), which is the topologically
        # rich target for DD-FP validation.
        lbl_z = labels[z]
        boundary = np.zeros(lbl_z.shape, dtype=bool)
        boundary[:, :-1] |= (lbl_z[:, :-1] != lbl_z[:, 1:])   # right neighbor differs
        boundary[:, 1:]  |= (lbl_z[:, :-1] != lbl_z[:, 1:])   # left neighbor differs
        boundary[:-1, :] |= (lbl_z[:-1, :] != lbl_z[1:, :])   # bottom neighbor differs
        boundary[1:, :]  |= (lbl_z[:-1, :] != lbl_z[1:, :])   # top neighbor differs
        lbl_bin = boundary.astype(np.uint8) * 255

        if patch_size is None:
            # Save full slice
            sid = f"{vol_name}_{z:04d}"
            Image.fromarray(img_arr, "L").save(img_dir / f"{sid}.png")
            Image.fromarray(lbl_bin, "L").save(lbl_dir / f"{sid}.png")
            count += 1
        else:
            # Non-overlapping patch tiling
            patch_idx = 0
            for y in range(0, H - patch_size + 1, stride):
                for x in range(0, W - patch_size + 1, stride):
                    img_p = img_arr[y:y+patch_size, x:x+patch_size]
                    lbl_p = lbl_bin[y:y+patch_size, x:x+patch_size]
                    sid = f"{vol_name}_{z:04d}_p{patch_idx:04d}"
                    Image.fromarray(img_p, "L").save(img_dir / f"{sid}.png")
                    Image.fromarray(lbl_p, "L").save(lbl_dir / f"{sid}.png")
                    patch_idx += 1
                    count += 1

    return count


def main(args: argparse.Namespace) -> None:
    _check_h5py()

    output_dir = Path(args.output_dir)
    img_dir    = output_dir / "images"
    lbl_dir    = output_dir / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # patch_size=0 means full-slice mode
    if args.patch_size == 0:
        args.patch_size = None

    if args.download:
        download_cremi(output_dir)

    hdf_dir = Path(args.hdf5_dir) if args.hdf5_dir else output_dir / "raw"
    total   = 0

    for vol in ["A", "B", "C"]:
        candidates = [
            hdf_dir / f"sample_{vol}_20160501.hdf",
            hdf_dir / f"sample_{vol}.hdf",
            hdf_dir / f"{vol}.hdf",
        ]
        hdf_path = next((p for p in candidates if p.exists()), candidates[0])

        print(f"\n[CONVERT] volume {vol}: {hdf_path}")
        n = convert_volume(hdf_path, vol, img_dir, lbl_dir,
                           args.max_slices, args.patch_size, args.patch_stride)
        print(f"          -> {n} slices saved")
        total += n

    print(f"\nDone: {total} total slices/patches")
    print(f"  images: {img_dir}")
    print(f"  masks:  {lbl_dir}")

    imgs = sorted(img_dir.glob("*.png"))
    lbls = sorted(lbl_dir.glob("*.png"))
    if len(imgs) != len(lbls):
        print(f"[WARN] image count ({len(imgs)}) != mask count ({len(lbls)})!")
    else:
        print(f"  verify: {len(imgs)} pairs -- OK")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CREMI HDF5 -> PNG conversion")
    p.add_argument("--hdf5-dir",    default=None,
                   help="HDF5 source directory (default: output-dir/raw/)")
    p.add_argument("--output-dir",  default="data/CREMI",
                   help="PNG output root (default: data/CREMI)")
    p.add_argument("--max-slices",   type=int, default=None,
                   help="Max slices per volume (default: all 125)")
    p.add_argument("--patch-size",   type=int, default=256,
                   help="Patch size in pixels (default 256). 0=full slice.")
    p.add_argument("--patch-stride", type=int, default=None,
                   help="Patch stride (default=patch_size, non-overlapping)")
    p.add_argument("--download",     action="store_true",
                   help="Auto-download CREMI raw HDF5 files")
    return p.parse_args()


if __name__ == "__main__":
    main(_parse())
