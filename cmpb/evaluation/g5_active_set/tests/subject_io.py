"""subject_io.py -- backend adapters and BraTS subject streaming shared by run_g4_full_accuracy.py and closure_depth_trace.py.

Two environments with the same small interface:
  CupyEnv : real CUDA through the sealed R7 helper package (r7_common.make_backend): preprocessing, ispan builder and the frozen production K=1
            call (seed_aware_boutry_bucket_gpu(boundary_seeds=None) -> frozen v1.1) are exactly those of G3.  Nothing is loaded except one subject at a time.
  ShimEnv : CPU plumbing test (kernels executed on CPU threads through tests/fakecupy.py).  Synthetic volumes only.
"""
from __future__ import annotations

import glob
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tests"))


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_arr(a) -> str:
    return sha_bytes(np.ascontiguousarray(a).tobytes())


def sha_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------------------
class ShimEnv:
    name, cuda = "shim", False

    def __init__(self, seed=3, threads=4):
        from fakecupy import FakeCupy
        import harness as Hh
        self.cp = FakeCupy(seed=seed, threads=threads)
        self.Hh, self.F = Hh, Hh.FROZEN
        self.fault = None

    def device_info(self):
        return dict(backend="shim", cuda_executed=False)

    def preprocess(self, raw):
        return raw

    def prep(self, vol):
        return self.Hh.CPU_FP.build_ispan_cpu(vol)

    def frozen(self, cp, lo, hi, l):
        return self.F.boutry_bucket_gpu(cp, lo, hi, l, "lower")

    def to_host(self, a):
        return np.asarray(a)

    def free(self): pass
    def sync(self): pass
    def memstats(self): return {}


class CupyEnv:
    name, cuda = "cupy", True

    def __init__(self, runner_dir, repo, lib):
        sys.path.insert(0, str(runner_dir))
        import r7_common as C
        self.C = C
        self.be = C.make_backend("cupy", repo, lib)
        self.cp = self.be.cp

    def device_info(self):
        return self.be.device_info()

    def preprocess(self, raw):
        return self.be.preprocess(raw)

    def prep(self, vol):
        return self.be.prep(vol)

    def frozen(self, cp, lo, hi, l):  # exactly what front_propagation_gpu calls for boundary_seeds=None
        return self.be._tie(cp, lo, hi, l, None, "lower")

    def to_host(self, a):
        return self.cp.asnumpy(a)

    def free(self): self.be.free()
    def sync(self): self.be.sync()
    def memstats(self): return self.be.memstats()


# ---------------------------------------------------------------------------------------------------------------------------------
def list_brats_flair(root) -> list[tuple[str, Path]]:
    """Same discovery rule as the frozen R3 helper (select_cohort): sorted glob of <root>/*/*_flair.nii*."""
    paths = sorted(glob.glob(str(Path(root) / "*" / "*_flair.nii*")))
    return [(Path(p).parent.name, Path(p)) for p in paths]


def load_raw(path: Path) -> np.ndarray:
    import nibabel as nib
    return nib.load(str(path)).get_fdata(dtype=np.float32)


def synthetic_volume(kind: str, shape=(5, 4, 3)) -> np.ndarray:
    """Small uint8 volumes with a known closure geometry (used to validate the tracer and the G4 runner on the shim)."""
    v = np.zeros(shape, np.uint8)
    if kind == "const":
        return v
    if kind == "block":                      # compact bright block in the interior: convex obstacle for the background level
        v[1:-1, 1:-1, 1:-1] = 200
        return v
    if kind == "nozero":                     # no zero voxel at all: only the 1-cell padding ring contains level 0
        return np.random.default_rng(5).integers(1, 256, shape, dtype=np.uint8)
    if kind == "wall":                       # one-voxel wall across the whole cross-section with a hole in the far corner: forces a detour
        v[shape[0] // 2, :, :] = 255
        v[shape[0] // 2, -1, -1] = 0
        return v
    raise ValueError(kind)


def synthetic_intervals(kind: str, vol_shape=(5, 4, 3)):
    """Directly built (lo, hi, l_inf) lattices.  'wallfull': constant level-0 lattice with a full plane (ring included) of level-255 cells,
    one hole in the far corner of that plane -> every level-0 path from the corner to the far side must detour through the hole."""
    Wp, Hp, Dp = (2 * v - 1 + 2 for v in vol_shape)
    lo = np.zeros((Wp, Hp, Dp), np.float32)
    hi = np.zeros((Wp, Hp, Dp), np.float32)
    if kind == "wallfull":
        x = Wp // 2
        lo[x, :, :] = 255
        hi[x, :, :] = 255
        lo[x, Hp - 1, Dp - 1] = 0
        hi[x, Hp - 1, Dp - 1] = 0
        return lo, hi, 0.0
    raise ValueError(kind)
